import math

import torch
from torch import Tensor
from transformers import AutoTokenizer

from .model import CimiModel


DEFAULT_TOKENIZER = "HuggingFaceTB/SmolLM2-135M"


def load_tokenizer(name: str = DEFAULT_TOKENIZER):
    return AutoTokenizer.from_pretrained(name, use_fast=True)


@torch.inference_mode()
def generate(
    model: CimiModel,
    input_ids: Tensor,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    eos_token_id: int | None = None,
) -> Tensor:
    if input_ids.ndim != 2 or input_ids.size(0) != 1:
        raise ValueError("input_ids must have shape (1, sequence)")
    if not 1 <= input_ids.size(1) <= model.config.max_seq_len:
        raise ValueError(f"prompt length must be between 1 and {model.config.max_seq_len}")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    if eos_token_id is not None and not 0 <= eos_token_id < model.config.vocab_size:
        raise ValueError("eos_token_id must be within the model vocabulary")

    was_training = model.training
    model.eval()
    try:
        steps = min(max_new_tokens, model.config.max_seq_len - input_ids.size(1))
        for _ in range(steps):
            logits = model(input_ids).logits[:, -1].float()
            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probabilities = (logits / temperature).softmax(dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
        return input_ids
    finally:
        model.train(was_training)
