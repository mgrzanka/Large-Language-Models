# 06 — CLIP: Vision–Language Representation Learning

A multimodal model trained contrastively to place images and their captions in one shared embedding space, then used for image-to-image and text-to-image retrieval. Dataset: Flickr30k.

## Architecture

```
caption ──► DistilBERT ──► [CLS] vector ──► projection head ──┐
                                                              ├──► shared space
image   ──► ResNet50 (timm) ──► pooled features ──► projection head ──┘
```

Both projection heads map into the same dimensionality: linear → GELU → linear → dropout → residual + LayerNorm. Without them the two encoders' outputs would live in incomparable spaces.

## The contrastive loss

Temperature-scaled. For a batch of *N* (image, caption) pairs, the matching pair must score higher than every mismatched combination in that batch — **negatives come from the batch itself**, which is why contrastive training benefits from large batches and why no explicit negative sampling is needed.

## Training

Separate learning rates per sub-network (a pretrained backbone should not be dragged at the same rate as a randomly initialised projection head), with image-side and text-side retrieval accuracy tracked separately so a collapse in one direction is visible.

## What it is used for

1. **Pair similarity** — embed two examples and compare all four image/caption combinations; matched pairs should dominate.
2. **Image-to-image retrieval** — embed the whole test set once, then *k*-nearest neighbours in the shared space.
3. **Text-to-image retrieval** — a natural-language query goes through the *text* encoder and is compared against *image* embeddings. This is the capability the shared space exists for: the query never touches an image model.

## Provenance

This lab's scaffolding came from the course; the implementations of the projection head, encoders, CLIP model, contrastive loss, embedding computation and nearest-neighbour search are mine.

## Run

Colab with a GPU. Flickr30k is pulled from Hugging Face; expect a long first download.
