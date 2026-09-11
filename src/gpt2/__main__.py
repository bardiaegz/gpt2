from .model import GPT
from .config import *
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

print(r"""
  ____                _  _____ _____ _______ 
 |  _ \              | |/ ____|  __ \__   __|
 | |_) | __ _ _ __ __| | |  __| |__) | | |   
 |  _ < / _` | '__/ _` | | |_ |  ___/  | |   
 | |_) | (_| | | | (_| | |__| | |      | |   
 |____/ \__,_|_|  \__,_|\_____|_|      |_|   
""")
dir_name = os.path.dirname(os.path.dirname(os.path.dirname((__file__))))
log_dir = os.path.join(dir_name, 'log')
log_path = os.path.join(log_dir, 'log.txt')
os.makedirs(log_dir, exist_ok=True)
log_file = open(log_path, 'w')

model = GPT(GPTConfig(vocab_size=50304))
model.to('mps')

enc = tiktoken.get_encoding('gpt2')


max_length = 30
num_return_sequences = 5

prompt = enc.encode('Hello, I\'m a language model,')
prompt = torch.tensor(prompt, dtype=torch.long)
prompt = prompt.unsqueeze(0)
prompt = prompt.repeat(num_return_sequences, 1)
prompt = prompt.to('mps')

model.eval()
with torch.inference_mode():
    log_file.write(f'\n{20 * '='} GENERATION {20 * '='}\n')
    while prompt.size(1) < max_length:
        logits, _ = model(prompt)
        logits = logits[:, -1, :]
        logits[:, enc.n_vocab:] = -float('inf')

        if temperature == 0:
            xcol = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indicies = torch.topk(probs, k=k, dim=-1)
            ix = torch.multinomial(topk_probs, num_samples=1)
            xcol = torch.gather(topk_indicies, dim=-1, index=ix)
        prompt = torch.cat((prompt, xcol), dim=-1)

for i in range(num_return_sequences):
    decoded = enc.decode(prompt[i, :max_length].tolist())
    log_file.write(f'\n SAMPLE {i} -> {decoded}\n')
    print(f'SAMPLE {i} -> {decoded}\n')
    print(decoded)

log_file.write(f'\n{'=' * 52}')
log_file.flush()
log_file.close()
