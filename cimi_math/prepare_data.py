import argparse
import gc
import json
from contextlib import closing
from itertools import islice
from pathlib import Path

import numpy as np
from datasets import load_dataset

from .inference import DEFAULT_TOKENIZER, load_tokenizer


def write_tokens(documents, tokenizer, output_dir: Path, train_tokens: int, val_tokens: int, source: dict):
    if min(train_tokens, val_tokens) < 2:
        raise ValueError("Each split must contain at least two tokens")
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define an EOS token")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any((output_dir / name).exists() for name in ("train.bin", "val.bin", "metadata.json")):
        raise FileExistsError(f"Token files already exist in {output_dir}")

    dtype = np.dtype("<u2" if len(tokenizer) <= 65536 else "<u4")
    targets = {"val": val_tokens, "train": train_tokens}
    counts = {"val": 0, "train": 0}
    documents = iter(documents)
    next_report = 1_000_000
    with (output_dir / "val.bin").open("wb") as val_file, (output_dir / "train.bin").open("wb") as train_file:
        files = {"val": val_file, "train": train_file}
        while counts["train"] < train_tokens:
            texts = list(islice(documents, 64))
            if not texts:
                raise ValueError(f"Dataset exhausted before reaching token targets: {counts}")
            texts = [text for text in texts if isinstance(text, str) and text.strip()]
            if not texts:
                continue
            encoded = tokenizer(texts, add_special_tokens=False, return_attention_mask=False)["input_ids"]
            for ids in encoded:
                # A document belongs to one split; discard its tail at a split boundary.
                split = "val" if counts["val"] < val_tokens else "train"
                ids = (ids + [tokenizer.eos_token_id])[: targets[split] - counts[split]]
                np.asarray(ids, dtype=dtype).tofile(files[split])
                counts[split] += len(ids)
                if sum(counts.values()) >= next_report:
                    print(f"Tokens written: train={counts['train']:,}, val={counts['val']:,}", flush=True)
                    next_report = sum(counts.values()) + 1_000_000
                if counts["train"] == train_tokens:
                    break

    tokenizer.save_pretrained(output_dir / "tokenizer")
    metadata = {
        "dtype": dtype.str, "vocab_size": len(tokenizer), "eos_token_id": tokenizer.eos_token_id,
        "train_tokens": counts["train"], "val_tokens": counts["val"], "source": source,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Data ready in {output_dir}: {counts}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--data-files", nargs="+", help="Local files, for example with --dataset json")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=Path("data/fineweb-edu"))
    parser.add_argument("--train-tokens", type=int, default=1_000_000_000)
    parser.add_argument("--val-tokens", type=int, default=1_000_000)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.shuffle_buffer <= 0:
        parser.error("--shuffle-buffer must be positive")
    tokenizer = load_tokenizer(args.tokenizer)
    dataset = load_dataset(
        args.dataset, name=None if args.data_files else args.dataset_config,
        data_files=args.data_files, split=args.split, streaming=True,
    ).shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    source = {key: value for key, value in vars(args).items() if key != "output_dir"}
    try:
        with closing(iter(dataset)) as rows, closing(row[args.text_column] for row in rows) as documents:
            write_tokens(documents, tokenizer, args.output_dir, args.train_tokens, args.val_tokens, source)
    finally:
        # Release streaming readers before Python starts finalizing native extensions.
        del dataset
        gc.collect()


if __name__ == "__main__":
    main()
