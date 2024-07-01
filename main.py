import itertools
import math
import os
import torch

import tiktoken
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import wandb
import random
from datasets import load_dataset, DatasetDict
import tiktoken
import json


class GPTDatasetV1(Dataset):
    def __init__(self, texts, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        token_filename = f"tokens_{len(texts)}.json"

        if token_filename in os.listdir("./"):
            token_ids = json.load(open(token_filename))["token_ids"]
            print("Loaded from file", len(token_ids))
        else: 
            # Tokenize the entire text
            token_ids = tokenizer.encode(texts)

            print("freshly encoded and baked", len(token_ids))

            # Convert tokens to JSON format
            tokens_json = json.dumps({"token_ids":token_ids})

            # Save tokens to a file
            with open(token_filename, 'w') as f:
                f.write(tokens_json)
 

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(texts, batch_size=8, max_length=512,
                         stride=128, shuffle=True, drop_last=True):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(texts=texts,tokenizer=tokenizer,max_length=max_length, stride=stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    return dataloader


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, block_size, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(block_size, block_size), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head
        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # Unsqueeze the mask to match dimensions
        mask_unsqueezed = mask_bool.unsqueeze(0)
        # Use the unsqueezed mask to fill attention scores
        attn_scores.masked_fill_(mask_unsqueezed, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
            nn.Dropout(cfg["drop_rate"])
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            block_size=cfg["ctx_len"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_resid(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["ctx_len"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

def generate_text(model, device, prompt="hello I am"):
    model.eval()  # disable dropout

    model.to("cpu")
    start_context = prompt

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=50,
        context_size=TINYCODER_CONFIG_sub100MB["ctx_len"]
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    model.to(device)

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)

    return None


def calc_loss_loader(data_loader, model, device, num_iters=None):
    total_loss, num_batches = 0., 0
    if num_iters is None:
        num_iters = len(data_loader)
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_iters:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
            num_batches += 1
        else:
            break

    wandb.log({"total_loss/epoch": total_loss})
    return total_loss


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    logits = model(input_batch)
    logits = logits.view(-1, logits.size(-1))
    loss = torch.nn.functional.cross_entropy(logits, target_batch.view(-1))

    wandb.log({"loss/batch": loss})
    return loss


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_iters=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_iters=eval_iter)
    wandb.log({"train_loss/epoch": train_loss})
    wandb.log({"val_loss/epoch":val_loss})
    model.train()
    return train_loss, val_loss


def train_model(model, train_loader, val_loader, optimizer, device,
                n_epochs, eval_freq, eval_iter,
                encoded_start_context, warmup_iters=10,
                initial_lr=3e-05, min_lr=1e-6):
    global_step = 0

    max_lr = optimizer.param_groups[0]["lr"]

    # Calculate total number of iterations
    total_training_iters = len(train_loader) * n_epochs

    # Calculate the learning rate increment at each step during warmup
    lr_increment = (optimizer.param_groups[0]["lr"] - initial_lr) / warmup_iters
    loss = 0
    train_loss = 0
    val_loss = 0

    for epoch in range(n_epochs):
        model.train()

        for batch_id, (input_batch, target_batch) in enumerate(train_loader):
            optimizer.zero_grad()

            # Increment the global step at the beginning of the iteration
            global_step += 1

            # Warmup: adjust learning rate linearly
            if global_step < warmup_iters:
                lr = initial_lr + global_step * lr_increment
            # Cosine annealing phase
            else:
                progress = (global_step - warmup_iters) / (total_training_iters - warmup_iters)
                lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            # Apply the calculated learning rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()

            wandb.log({"training loss/batch": loss.item()})
            print(f"Epoch:{epoch}/{n_epochs} | Batch:{batch_id} | loss/batch: {loss.item()} | loss/epoch: {train_loss} | val_loss/epoch: {val_loss}")


            # Apply gradient clipping
            if global_step >= warmup_iters:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()


        train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)

        torch.save({
        'epoch': n_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item()
        },"./newrun.pt")


    return train_loss, val_loss

# Generate all combinations of hyperparameters
HPARAM_GRID = {
    "batch_size": [8],
    "drop_rate": [0.15],
    "warmup_iters": [15],
    "weight_decay": [0.04],
    "peak_lr": [0.001],
    "initial_lr": [0.00003],
    "min_lr": [0.00005],
    "n_epochs": [40],
}

TINYCODER_CONFIG_sub100MB =  {"vocab_size": 50257,  # Vocabulary size
                "ctx_len": 512,       # Context length -- shortened from original 1024 tokens
                "emb_dim": 256,       # Embedding dimension
                "n_heads": 4,        # Number of attention heads
                "n_layers": 4,       # Number of layers
                "drop_rate": 0.15,
                "qkv_bias": False,  
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Accept a prompt of text")
    parser.add_argument("--device", type=str, help="Enter your device")
    parser.add_argument("--epoch", type=int, help="Enter your epochs", default=2)
    parser.add_argument("--prompt", type=str, help="Enter your prompt", default=None)


    args = parser.parse_args()

    HPARAM_GRID["n_epochs"] = [args.epoch]
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="tinycoder-sub100MB",
        config=TINYCODER_CONFIG_sub100MB
    )


    hyperparameter_combinations = list(itertools.product(*HPARAM_GRID.values()))
    total_combinations = len(hyperparameter_combinations)
    print(f"Total hyperparameter configurations: {total_combinations}")
    # Placeholder for the best loss and best hyperparameters
    best_val_loss = float('inf')
    best_hparams = {}

    
    train_dataset = load_dataset("roneneldan/TinyStories", split="train")["text"][:200]
    valid_dataset = load_dataset("roneneldan/TinyStories", split="validation")["text"]

    train_dataset = " ".join(train_dataset)
    valid_dataset = " ".join(valid_dataset)

    device = args.device

    torch.manual_seed(123)

    interrupted = False
    current_config = 0
    for combination in hyperparameter_combinations:
        current_config += 1
        print(f"Evaluating configuration {current_config} of {total_combinations}")

        # Unpack the current combination of hyperparameters
        HPARAM_CONFIG = dict(zip(HPARAM_GRID.keys(), combination))

        torch.manual_seed(123)
        train_loader = create_dataloader_v1(
            texts=train_dataset,
            batch_size=HPARAM_CONFIG["batch_size"],
            max_length=TINYCODER_CONFIG_sub100MB["ctx_len"],
            stride=TINYCODER_CONFIG_sub100MB["ctx_len"],
            drop_last=True,
            shuffle=True
        )

        val_loader = create_dataloader_v1(
            texts=valid_dataset,
            batch_size=HPARAM_CONFIG["batch_size"],
            max_length=TINYCODER_CONFIG_sub100MB["ctx_len"],
            stride=TINYCODER_CONFIG_sub100MB["ctx_len"],
            drop_last=False,
            shuffle=False
            )

        model = GPTModel(TINYCODER_CONFIG_sub100MB)
        model.to(device)

        print(" Model Total Parameters", sum(p.numel() for p in model.parameters()))

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=HPARAM_CONFIG["peak_lr"],
            weight_decay=HPARAM_CONFIG["weight_decay"]
        )

        encoded_start_context = train_loader.dataset.tokenizer.encode("Nevertheless")
        encoded_tensor = torch.tensor(encoded_start_context).unsqueeze(0)

        if "oldrun.pt" in os.listdir("./"):
            checkpoint = torch.load("oldrun.pt")
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            epoch = checkpoint["epoch"]
            loss = checkpoint["loss"]
            #optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        model.train()

        train_loss, val_loss = train_model(
            model, train_loader, val_loader, optimizer, device,
            n_epochs=HPARAM_CONFIG["n_epochs"],
            eval_freq=5, eval_iter=1,
            encoded_start_context=encoded_tensor,
            warmup_iters=HPARAM_CONFIG["warmup_iters"],
            initial_lr=HPARAM_CONFIG["initial_lr"],
            min_lr=HPARAM_CONFIG["min_lr"]
        )

        print(f"Train loss:{train_loss} and Val loss:{val_loss}")

    wandb.finish()