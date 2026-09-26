# 03 — Parameter-Efficient Fine-Tuning

Five ways to make GPT-2 perform E2E NLG — turning a structured restaurant record into a natural-language description — compared on the same data, the same prompt and the same generation call, so that only the adaptation mechanism differs.

## The methods

| Method | What is trained | Where it acts |
|---|---|---|
| Few-shot prompting | nothing | the prompt (0, 3, 8 examples) |
| **LoRA** | low-rank adapters | attention projections |
| **Prompt tuning** | soft prompt vectors | model input only |
| **Prefix-tuning** | prefixes | every layer |
| **P-tuning** | an LSTM prompt encoder | generates the prompt rather than learning raw vectors |

## Why this comparison is the point

Each method makes a different bet about *where* task knowledge should live. Few-shot puts it in the context window and trains nothing. Prompt tuning learns a handful of input vectors. Prefix-tuning pushes them into every layer, buying depth of influence at the cost of more parameters. P-tuning replaces raw learnable vectors with an encoder that produces them, which stabilises training. LoRA leaves the prompt alone entirely and edits the weights — but only along a low-rank update.

Running all five against one baseline makes the trade-off legible in a way that any single method in isolation does not.

## Setup

Base model GPT-2, dataset `e2e-dataset` (restaurant records → descriptions), trained with the HF `Trainer` and evaluated by generating from the same held-out prompt after each method. `DataCollatorForLanguageModeling` handles causal-LM batching.

## Run

```bash
pip install transformers peft datasets "bitsandbytes>=0.46.1"
```

Then run the notebook top to bottom; each section trains one adaptation and generates from it.
