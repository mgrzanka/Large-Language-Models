# Large Language Models

**Seven labs from the _Wielkie Modele Językowe_ (LLM) course** at Warsaw University of Technology, covering the stack end to end: from TF-IDF baselines, through building a decoder-only transformer from scratch, to parameter-efficient fine-tuning, retrieval-augmented generation, multimodal CLIP and agent systems.

Each lab is a single notebook with its outputs preserved - training logs, generated text, retrieved documents and plots are all still in the cells.

---

| #                                 | Lab                      | Topic                                | Built                                                                 | Key libraries                    |
| --------------------------------- | ------------------------ | ------------------------------------ | --------------------------------------------------------------------- | -------------------------------- |
| [01](01-text-classification)      | Text classification      | 5-class Yelp review rating           | TF-IDF + MLP, error analysis, two ablation sweeps                     | scikit-learn, PyTorch, W&B       |
| [02](02-decoder-only-transformer) | Decoder-only transformer | Polish GPT trained from scratch      | Causal mask, MHA, Transformer block, pretraining, sampling strategies | PyTorch, transformers            |
| [03](03-peft-adaptation)          | **PEFT adaptation**      | E2E NLG with GPT-2                   | **Few-shot vs LoRA vs prompt / prefix / P-tuning**                    | peft, transformers               |
| [04](04-hybrid-retrieval)         | Hybrid retrieval         | RAG over 50k Polish Wikipedia chunks | BM25 + bi-encoder + cross-encoder reranking                           | rank_bm25, sentence-transformers |
| [05](05-rag-generation)           | RAG generation           | Grounded answers with citations      | PLLuM-8B, native `generate`, FlashAttention-2                         | transformers                     |
| [06](06-clip-vision-language)     | **CLIP**                 | Image–text retrieval on Flickr30k    | **Contrastive multimodal training from scratch**                      | timm, transformers               |
| [07](07-llm-agents)               | LLM agents               | Clinic recommendation agent          | LangGraph state machine, tools, RAG over reviews                      | LangChain, LangGraph             |

## Detailed descriptions

**[03 - PEFT adaptation](03-peft-adaptation)** puts five adaptation strategies side by side on one task and one prompt: few-shot prompting (nothing trained), LoRA adapters in the attention layers, prompt tuning, prefix-tuning and P-tuning with an LSTM prompt encoder. The comparison is what makes it useful - the same dataset, the same base model, the same generation call, only the adaptation mechanism changes.

**[04 - Hybrid retrieval](04-hybrid-retrieval)** builds a retriever rather than calling one. Lexical search (BM25 with Polish stop words), dense search (bi-encoder), and a cross-encoder that reranks the union of both candidate pools. Polish Wikipedia is chunked into 50k overlapping passages so a sentence cut at a chunk boundary is not lost, and the result feeds a prompt that forces `[document_number]` citations.

**[06 - CLIP](06-clip-vision-language)** trains a multimodal model contrastively: DistilBERT text encoder, ResNet50 image encoder, a projection head per modality into a shared space, and a temperature-scaled contrastive loss whose negatives come from the batch itself. Then it uses the result for image-to-image and text-to-image retrieval.

## Layout

```
NN-lab/
├── *.ipynb     the lab, with outputs preserved
└── README.md   problem, approach, what the notebook shows
```

## Stack

PyTorch · Hugging Face Transformers + PEFT · sentence-transformers · rank_bm25 · timm · LangChain · LangGraph · scikit-learn · Weights & Biases

## A note on provenance

These are course laboratories: the task statements and, in some notebooks, the initial scaffolding came from the course. The implementations, experiments and conclusions in them are mine. Course-provided prose has been removed so each notebook reads as the work it documents rather than as a filled-in worksheet.

Datasets are pulled from Hugging Face at runtime, except lab 07, whose `clinics.csv` and `clinic_reviews.csv` were distributed by the course and are not redistributed here.
