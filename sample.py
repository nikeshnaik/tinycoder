import tiktoken
import torch
import os
from main import GPTModel, TINYCODER_CONFIG_sub100MB, generate_text_simple
import argparse

parser = argparse.ArgumentParser(description="Accept a prompt of text")
parser.add_argument("prompt", type=str, help="Enter your prompt text")
parser.add_argument("model_chkpt", type=str, help="model checkpoint location", default="./newrun.pt")


args = parser.parse_args()

model_chkpt = args.model_chkpt

model = GPTModel(TINYCODER_CONFIG_sub100MB)


if f"{model_chkpt}.pt" in os.listdir("./"):
    checkpoint = torch.load(f"{model_chkpt}.pt", map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

model.eval()  # disable dropout

start_context = prompt_text = args.prompt

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

print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
print("\nOutput:", out)
print("Output length:", len(out[0]))
print("Output text:", decoded_text)