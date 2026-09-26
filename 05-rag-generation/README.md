# 05 — RAG Generation

The generator: the component that receives a numbered list of retrieved documents plus a question, and produces a grounded answer with citations. Built on `Llama-PLLuM-8B`.

## What is compared

**Instruct vs chat model.** The same RAG prompt through `Llama-PLLuM-8B-instruct` and `Llama-PLLuM-8B-chat`. The instruct model is more precise but reads mechanically; the chat variant is more fluent and more inclined to add framing the documents do not support — which for a grounded-answering task is a cost, not a feature.

**`device_map="auto"` vs `"cuda"`.** Observed directly rather than assumed: `auto` will offload layers when VRAM is tight, `cuda` forces everything onto the GPU and fails if it does not fit.

**`pipeline` vs native `transformers`.** The second implementation drops the pipeline abstraction for `AutoModelForCausalLM` + `AutoTokenizer`, applies the chat template manually, calls `generate`, and slices the output so only newly generated tokens are decoded — the step a pipeline hides and people forget when they write it themselves.

**FlashAttention-2.** The same generation with the optimised attention kernel: about **8 minutes against 11**. The notebook notes why the gain is moderate rather than dramatic — FlashAttention's advantage grows with sequence length, and these prompts are short, so the quadratic attention term is not yet the bottleneck.

## Run

Colab with a GPU large enough for an 8B model in fp16. FlashAttention-2 requires a compatible GPU architecture.
