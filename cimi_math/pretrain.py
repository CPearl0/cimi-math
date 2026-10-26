import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
from einops import rearrange
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from .data import BatchStream, TokenDataset
from .model import CimiModel, ModelConfig


@dataclass
class TrainConfig:
    data_dir: str = "data/fineweb-edu"
    output_dir: str = "runs/pretrain"
    device: str = "cuda"
    dtype: str = "bfloat16"
    seq_len: int = 1024
    batch_size: int = 2
    grad_accum_steps: int = 16
    max_steps: int = 30000
    warmup_steps: int = 1000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    aux_loss_weight: float = 0.01
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    log_interval: int = 10
    eval_interval: int = 250
    eval_batches: int = 20
    save_interval: int = 1000
    seed: int = 42
    resume: str | None = None

    def __post_init__(self):
        for name in ("learning_rate", "min_learning_rate", "weight_decay", "aux_loss_weight", "max_grad_norm"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        for name in (
            "seq_len", "batch_size", "grad_accum_steps", "max_steps", "log_interval",
            "eval_interval", "eval_batches", "save_interval",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.warmup_steps < self.max_steps:
            raise ValueError("warmup_steps must be between 0 and max_steps - 1")
        if not 0 <= self.min_learning_rate <= self.learning_rate or self.learning_rate <= 0:
            raise ValueError("Require 0 <= min_learning_rate <= learning_rate and learning_rate > 0")
        if self.weight_decay < 0 or self.aux_loss_weight < 0 or self.max_grad_norm <= 0:
            raise ValueError("Invalid weight decay, auxiliary loss weight or gradient clipping threshold")
        if self.dtype not in ("float32", "bfloat16"):
            raise ValueError("dtype must be float32 or bfloat16")


def learning_rate_at(step: int, config: TrainConfig) -> float:
    if step <= config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    decay_steps = config.max_steps - config.warmup_steps
    progress = 1.0 if decay_steps == 1 else (step - config.warmup_steps - 1) / (decay_steps - 1)
    return config.min_learning_rate + 0.5 * (config.learning_rate - config.min_learning_rate) * (
        1 + math.cos(math.pi * progress)
    )


def compute_losses(model, inputs, targets, config):
    with torch.autocast(device_type=inputs.device.type, dtype=torch.bfloat16, enabled=config.dtype == "bfloat16"):
        output = model(inputs)
        lm_loss = F.cross_entropy(
            rearrange(output.logits, "batch seq vocab -> (batch seq) vocab"),
            rearrange(targets, "batch seq -> (batch seq)"),
        )
        total_loss = lm_loss + config.aux_loss_weight * output.aux_loss
    return lm_loss, output.aux_loss, total_loss


def create_optimizer(model, config):
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": config.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=config.learning_rate, betas=(0.9, 0.95), fused=next(model.parameters()).is_cuda,
    )


def train_step(model, optimizer, get_batch, config, device):
    optimizer.zero_grad(set_to_none=True)
    step_losses = torch.zeros(3, device=device)
    for _ in range(config.grad_accum_steps):
        inputs, targets = get_batch()
        losses = compute_losses(model, inputs.to(device), targets.to(device), config)
        (losses[2] / config.grad_accum_steps).backward()
        step_losses += torch.stack([loss.detach() for loss in losses]) / config.grad_accum_steps
    if not torch.isfinite(step_losses).all():
        raise FloatingPointError("Non-finite training loss")
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    return step_losses, grad_norm.detach()


@torch.no_grad()
def evaluate(model, dataset, config, device):
    was_training = model.training
    model.eval()
    totals = torch.zeros(3, device=device)
    token_count = 0
    try:
        limit = min(len(dataset), config.eval_batches * config.batch_size)
        for start in range(0, limit, config.batch_size):
            inputs, targets = dataset.get_batch(range(start, min(start + config.batch_size, limit)))
            inputs, targets = inputs.to(device), targets.to(device)
            losses = torch.stack(compute_losses(model, inputs, targets, config))
            totals += losses * targets.numel()
            token_count += targets.numel()
        if not torch.isfinite(totals).all():
            raise FloatingPointError("Non-finite validation loss")
        return (totals / token_count).tolist()
    finally:
        model.train(was_training)


def save_checkpoint(path, model, optimizer, stream, step, config, metadata):
    device = next(model.parameters()).device
    state = {
        "model": model.state_dict(), "model_config": asdict(model.config),
        "optimizer": optimizer.state_dict(), "train_config": asdict(config),
        "step": step, "data_metadata": metadata, "data_state": stream.state_dict(),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def run_training(config: TrainConfig, model_config: ModelConfig | None = None):
    device = torch.device(config.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Training supports CPU and CUDA devices")
    if device.type == "cuda":
        device = torch.device("cuda", device.index if device.index is not None else torch.cuda.current_device())
        torch.cuda.set_device(device)
        if config.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("This GPU does not support BF16; use --dtype float32")
    torch.manual_seed(config.seed)
    data_dir = Path(config.data_dir)
    metadata = json.loads((data_dir / "metadata.json").read_text())
    checkpoint = torch.load(config.resume, map_location="cpu", weights_only=False) if config.resume else None
    if checkpoint:
        saved_model_config = ModelConfig(**checkpoint["model_config"])
        if model_config is not None and model_config != saved_model_config:
            raise ValueError("Model configuration differs from the checkpoint")
        model_config = saved_model_config
        mutable = {"output_dir", "resume", "device", "max_steps", "log_interval", "eval_interval", "eval_batches", "save_interval"}
        for name, value in asdict(config).items():
            if name not in mutable and checkpoint["train_config"][name] != value:
                raise ValueError(f"Cannot change {name} when resuming")
        if metadata != checkpoint["data_metadata"]:
            raise ValueError("Dataset metadata differs from the checkpoint")
    model_config = model_config or ModelConfig()
    if model_config.vocab_size != metadata["vocab_size"]:
        raise ValueError("Dataset and model vocabularies must match")
    if config.seq_len > model_config.max_seq_len:
        raise ValueError("seq_len exceeds the model context length")

    train_data = TokenDataset(data_dir / "train.bin", config.seq_len, metadata["dtype"])
    val_data = TokenDataset(data_dir / "val.bin", config.seq_len, metadata["dtype"])
    if len(train_data.tokens) != metadata["train_tokens"] or len(val_data.tokens) != metadata["val_tokens"]:
        raise ValueError("Token file sizes differ from dataset metadata")
    stream = BatchStream(train_data, config.batch_size, config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "last.pt"
    if checkpoint_path.exists() and not config.resume:
        raise FileExistsError(f"{checkpoint_path} already exists; use --resume or a new output directory")

    model = CimiModel(model_config)
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.gradient_checkpointing = config.gradient_checkpointing
    optimizer = create_optimizer(model, config)
    start_step = 0
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        stream.load_state_dict(checkpoint["data_state"])
        start_step = checkpoint["step"]
        torch.set_rng_state(checkpoint["rng_state"])
        if device.type == "cuda" and checkpoint["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"], device)
        if config.max_steps != checkpoint["train_config"]["max_steps"]:
            print("Resuming with a new max_steps; cosine decay uses the new training horizon.", flush=True)
        del checkpoint
    if start_step >= config.max_steps:
        raise ValueError("The checkpoint has already reached max_steps")

    tokens_per_step = config.batch_size * config.grad_accum_steps * config.seq_len
    print(
        f"Parameters={sum(p.numel() for p in model.parameters()):,}, tokens/step={tokens_per_step:,}, "
        f"target tokens={config.max_steps * tokens_per_step:,}, device={device}, dtype={config.dtype}",
        flush=True,
    )
    writer = SummaryWriter(str(output_dir / "tensorboard"), purge_step=start_step + 1 if config.resume else None)
    writer.add_text("config", json.dumps({"train": asdict(config), "model": asdict(model_config)}, indent=2), start_step)
    window = torch.zeros(4, device=device)
    window_steps = 0
    try:
        if start_step == 0:
            for name, value in zip(("lm_loss", "aux_loss", "total_loss"), evaluate(model, val_data, config, device)):
                writer.add_scalar(f"val/{name}", value, 0)
            writer.flush()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        model.train()
        for step in range(start_step + 1, config.max_steps + 1):
            lr = learning_rate_at(step, config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            step_losses, grad_norm = train_step(model, optimizer, stream.next_batch, config, device)
            window[:3] += step_losses
            window[3] += grad_norm
            window_steps += 1

            if step == 1 or step % config.log_interval == 0 or step == config.max_steps:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                lm_loss, aux_loss, total_loss, norm = (window / window_steps).tolist()
                metrics = {
                    "train/lm_loss": lm_loss, "train/aux_loss": aux_loss, "train/total_loss": total_loss,
                    "train/grad_norm": norm, "train/learning_rate": lr, "train/tokens_seen": step * tokens_per_step,
                    "train/epoch": stream.epoch + stream.position / len(train_data),
                    "perf/tokens_per_second": window_steps * tokens_per_step / elapsed,
                    "perf/seconds_per_step": elapsed / window_steps,
                }
                if device.type == "cuda":
                    metrics.update({
                        "memory/allocated_gib": torch.cuda.memory_allocated(device) / 1024**3,
                        "memory/reserved_gib": torch.cuda.memory_reserved(device) / 1024**3,
                        "memory/peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                    })
                for name, value in metrics.items():
                    writer.add_scalar(name, value, step)
                writer.flush()
                print(
                    f"step={step}/{config.max_steps} loss={lm_loss:.4f} aux={aux_loss:.4f} "
                    f"grad_norm={norm:.3f} lr={lr:.3e} tokens/s={metrics['perf/tokens_per_second']:.0f}",
                    flush=True,
                )
                window.zero_()
                window_steps = 0
                started = time.perf_counter()
            if step % config.eval_interval == 0 or step == config.max_steps:
                values = evaluate(model, val_data, config, device)
                for name, value in zip(("lm_loss", "aux_loss", "total_loss"), values):
                    writer.add_scalar(f"val/{name}", value, step)
                writer.flush()
                print(f"validation step={step} loss={values[0]:.4f} aux={values[1]:.4f}", flush=True)
            if step % config.save_interval == 0 or step == config.max_steps:
                save_checkpoint(checkpoint_path, model, optimizer, stream, step, config, metadata)
                print(f"Checkpoint saved: {checkpoint_path}", flush=True)
    finally:
        writer.close()
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser()
    for field in fields(TrainConfig):
        options = {"default": field.default}
        if isinstance(field.default, bool):
            options["action"] = argparse.BooleanOptionalAction
        else:
            options["type"] = str if field.default is None else type(field.default)
        parser.add_argument("--" + field.name.replace("_", "-"), **options)
    run_training(TrainConfig(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
