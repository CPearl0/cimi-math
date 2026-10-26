import argparse
import math
from pathlib import Path

import torch
from torch import Tensor
from transformers import AutoTokenizer

from .model import CimiModel, ModelConfig


DEFAULT_TOKENIZER = "HuggingFaceTB/SmolLM2-135M"


def load_tokenizer(name: str = DEFAULT_TOKENIZER, *, local_files_only: bool = False):
    return AutoTokenizer.from_pretrained(name, use_fast=True, local_files_only=local_files_only)


def load_checkpoint_model(path: str | Path, device: str | torch.device = "cpu"):
    # Memory mapping avoids reading the optimizer tensors into RAM for inference.
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    model = CimiModel(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    info = {key: checkpoint[key] for key in ("step", "train_config", "data_metadata")}
    del checkpoint
    return model.to(device).eval(), info


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", action="append", required=True, help="Repeat this option to try multiple prompts")
    parser.add_argument("--tokenizer", help="Local directory or Hugging Face repo; defaults to the saved data tokenizer")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("Supported devices are cpu and cuda")
    if args.max_new_tokens < 0 or not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("max-new-tokens and temperature must be nonnegative, and temperature must be finite")
    supports_bf16 = False
    if device.type == "cuda":
        device = torch.device("cuda", device.index if device.index is not None else torch.cuda.current_device())
        torch.cuda.set_device(device)
        supports_bf16 = torch.cuda.is_bf16_supported()
    dtype = args.dtype or ("bfloat16" if supports_bf16 else "float32")
    if device.type == "cuda" and dtype == "bfloat16" and not supports_bf16:
        parser.error("This GPU does not support BF16; use --dtype float32")

    model, info = load_checkpoint_model(args.checkpoint, device)
    tokenizer_name = args.tokenizer
    if tokenizer_name is None:
        tokenizer_path = Path(info["train_config"]["data_dir"]) / "tokenizer"
        if not tokenizer_path.is_dir():
            raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_path}; specify --tokenizer")
        tokenizer_name = str(tokenizer_path.resolve())
    tokenizer = load_tokenizer(tokenizer_name, local_files_only=args.tokenizer is None)
    if not tokenizer.is_fast or len(tokenizer) != model.config.vocab_size:
        raise ValueError("A fast tokenizer matching the model vocabulary is required")
    if tokenizer.eos_token_id != info["data_metadata"]["eos_token_id"]:
        raise ValueError("Tokenizer EOS token differs from the training data")
    torch.manual_seed(args.seed)
    print(f"Loaded {args.checkpoint}, step={info['step']}, device={device}, dtype={dtype}", flush=True)
    for prompt in args.prompt:
        input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        print(f"\nPrompt:\n{prompt}", flush=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=dtype == "bfloat16"):
            generated = generate(
                model, input_ids, max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, eos_token_id=tokenizer.eos_token_id,
            )
        continuation = generated[0, input_ids.size(1):].tolist()
        text = tokenizer.decode(continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(f"Continuation ({len(continuation)} tokens):\n{text}", flush=True)


if __name__ == "__main__":
    main()
