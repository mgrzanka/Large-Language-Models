# 01 — Text Classification with Classical Methods

Predict the star rating (1–5) of a Yelp review from its text. TF-IDF features, MLP classifier, pure PyTorch, tracked in Weights & Biases.

## Result

**58% accuracy** on 10,000 held-out test reviews across 5 classes (random ≈ 20%). Precision, recall and F1 are all 0.58 macro-averaged — the model is not trading one class off against another.

## What the notebook does beyond training

**Error analysis.** A function that surfaces the *k* test examples where the model is most wrong — lowest logit on the true class — plus the label distribution of those failures. The finding: most errors are reviews whose wording contradicts the score (positive phrasing on a 2-star review), which is exactly the failure mode a bag-of-words representation cannot avoid, since it has no access to negation or irony.

**Feature ablation.** Vocabulary size, lowercasing and stop-word removal, each varied independently:

- vocabulary size matters most — about +3 points going from 1,000 to 13,000 terms,
- most of that gain arrives early (1,000 → 5,000); past ~9,000 the curve flattens, which is saturation rather than a reason to keep growing it.

**Hyperparameter ablation.** Hidden-layer count, epoch budget and initial learning rate. One to three hidden layers perform comparably; deeper networks degrade. Learning rate is the sharpest knob — 0.3 fails to train at all (accuracy pinned at 0.20, i.e. chance), while 0.01 overfits hard (91% train, 54% validation).

Runs are single-seed: the RAM ceiling on Colab and the cost of re-vectorising the dataset made repeats impractical, and the notebook says so rather than implying more.

## Run

Colab with a GPU. Set `WANDB_API_KEY` in the environment before the W&B login cell.
