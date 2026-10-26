from pathlib import Path

import numpy as np
import torch


class TokenDataset:
    def __init__(self, path: str | Path, seq_len: int, dtype: str):
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        self.tokens = np.memmap(path, mode="r", dtype=dtype)
        self.seq_len = seq_len
        if len(self) == 0:
            raise ValueError(f"{path} must contain at least {seq_len + 1} tokens")

    def __len__(self):
        return (len(self.tokens) - 1) // self.seq_len

    def get_batch(self, indices):
        blocks = [
            self.tokens[index * self.seq_len : (index + 1) * self.seq_len + 1]
            for index in indices
        ]
        tokens = torch.from_numpy(np.array(blocks, dtype=np.int64))
        return tokens[:, :-1], tokens[:, 1:]


class BatchStream:
    def __init__(self, dataset: TokenDataset, batch_size: int, seed: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.position = 0
        self._shuffle()

    def _shuffle(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.order = torch.randperm(len(self.dataset), generator=generator)

    def next_batch(self):
        indices = []
        while len(indices) < self.batch_size:
            if self.position == len(self.dataset):
                self.epoch += 1
                self.position = 0
                self._shuffle()
            count = min(self.batch_size - len(indices), len(self.dataset) - self.position)
            indices.extend(self.order[self.position : self.position + count].tolist())
            self.position += count
        return self.dataset.get_batch(indices)

    def state_dict(self):
        return {"epoch": self.epoch, "position": self.position}

    def load_state_dict(self, state):
        if state["epoch"] < 0 or not 0 <= state["position"] <= len(self.dataset):
            raise ValueError("Invalid batch stream state")
        self.epoch, self.position = state["epoch"], state["position"]
        self._shuffle()
