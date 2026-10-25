# Cimi Math

A project to train a small Mixture-of-Experts (MoE) language model from scratch for basic English conversation and mathematical reasoning. The goal is to learn the full training pipeline with modest compute resources.

## Model

- Approximately **463M total parameters**, with 24 layers, hidden size 1024, and 16 attention heads.
- Alternating dense and MoE feed-forward layers; 4 experts per MoE layer with top-2 routing. Dense FFN intermediate size: 2816; each expert: 1408.
- RoPE, RMSNorm, SwiGLU, shared input/output embeddings, and an initial context length of 1024 tokens.
- Reuse the **SmolLM2 fast tokenizer** with a 49,152-token vocabulary; initialize student model weights from scratch.
- Implement the model, sparse routing, generation, training loops, SFT, and GRPO in PyTorch.

## Training Pipeline

1. **General pretraining:** Train on an English FineWeb-Edu subset, initially targeting roughly 1B tokens..
2. **Teacher data distillation:** Use a Qwen teacher to generate concise mathematical solutions.
3. **Mixed supervised fine-tuning (SFT):** Combine general English instruction data with distilled math solutions, computing loss only on assistant responses.
4. **Math reinforcement learning (GRPO):** Sample student solutions, reward verifiably correct answers, and regularize against a fixed SFT reference policy with a KL penalty.

