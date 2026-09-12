from .model import GPT
from .config import *
from .data import DataLoader
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm

BANNER = r"""
  ____                _  _____ _____ _______ 
 |  _ \              | |/ ____|  __ \__   __|
 | |_) | __ _ _ __ __| | |  __| |__) | | |   
 |  _ < / _` | '__/ _` | | |_ |  ___/  | |   
 | |_) | (_| | | | (_| | |__| | |      | |   
 |____/ \__,_|_|  \__,_|\_____|_|      |_|   
"""


def main() -> None:
    print(BANNER)

    dir_name = os.path.dirname(os.path.dirname(os.path.dirname((__file__))))
    log_dir = os.path.join(dir_name, 'log')
    log_path = os.path.join(log_dir, 'log.txt')
    os.makedirs(log_dir, exist_ok=True)
    log_file = open(log_path, 'w')

    model = GPT(GPTConfig(vocab_size=50304))
    model.to('mps')

    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)
    train_loader = DataLoader(B=B, T=T, device='mps')
    enc = tiktoken.get_encoding('gpt2')

    prompt = enc.encode('Hello, I\'m a language model,')
    prompt = torch.tensor(prompt, dtype=torch.long)
    prompt = prompt.unsqueeze(0)
    prompt = prompt.repeat(num_return_sequences, 1)
    prompt = prompt.to('mps')

    pbar = tqdm(range(max_steps), desc='Training BardGPT', colour='#7BC621', dynamic_ncols=True)
    for step in pbar:
        x_gen = prompt.clone()
        last_step = (step == max_steps - 1)
        if (step > 0 and step % 100 == 0) or last_step:
            model.eval()
            with torch.inference_mode():
                log_file.write(f'\n{20 * '='} GENERATION {20 * '='}\n')
                while x_gen.size(1) < max_length:
                    logits, _ = model(x_gen)
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
                    x_gen = torch.cat((x_gen, xcol), dim=-1)

            for i in range(num_return_sequences):
                decoded = enc.decode(x_gen[i, :max_length].tolist())
                log_file.write(f'\n SAMPLE {i} -> {decoded}\n')
                print(f'SAMPLE {i} -> {colors.OKGREEN}{decoded}{colors.ENDC}\n')

            log_file.write(f'\n{'=' * 52}')
            log_file.flush()
            model.train()
        optimizer.zero_grad()
        x, y = train_loader.next_batch()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=f'{loss.item():.6f}')

    log_file.close()
