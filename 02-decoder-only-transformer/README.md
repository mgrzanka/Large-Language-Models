# 02 — Decoder-Only Transformer from Scratch

A GPT-style autoregressive language model for Polish, built from primitives in PyTorch and pretrained on the Wolne Lektury corpus.

## What is implemented

| Component | Detail |
|---|---|
| Causal mask | Upper-triangular `-inf` above the diagonal — a token cannot attend to its own future |
| Attention | Multi-head attention, 12 heads |
| Feed-forward | 4× expansion, GELU |
| Transformer block | Pre-norm: LayerNorm → attention → residual → LayerNorm → FFN → residual |
| Language model | Token + positional embeddings, 12 blocks, projection head onto the vocabulary |

Configuration: 256-token context, 768-dim embeddings, 12 layers, 12 heads, dropout 0.1. Tokeniser: `sdadas/polish-gpt2-small`.

## Data and training

Documents are split into overlapping chunks (stride = half the context) and tokenised in batches. Training uses AdamW with gradient accumulation over 4 steps, `CosineAnnealingLR`, `torchmetrics` for loss and next-token accuracy, and checkpoints every tenth of the run.

## Generation

Autoregressive decoding with the sampling function passed in, so strategies are swappable: greedy (`argmax`) and temperature sampling are both implemented and compared on the same prompts.

## Quantised pretrained model

The final section loads `microsoft/phi-1_5` in 8-bit via `bitsandbytes`, reports the memory footprint, and generates with temperature sampling — the contrast between a small model trained from scratch and a pretrained one that only fits comfortably when quantised.

## Run

Colab with a GPU. The corpus dump is fetched with `gdown` in the notebook.
