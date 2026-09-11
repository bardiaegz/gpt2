from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 50_257
    block_size: int = 1_024
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    expansion_factor: int = 4

temperature = 0.8
k = 50
p = 0.95