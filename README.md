# Waste Sorting with Deep Learning: A CNN Classifier for Recyclables

A DenseNet121-based image classifier that sorts waste into **cardboard, glass, metal, and plastic**, built to explore not just baseline accuracy but how well a recycling classifier holds up when it meets images from the real world.

Incorrect sorting is one of the biggest practical barriers to effective recycling — contaminated batches often get sent to landfill even when most of the material was recoverable. This project treats that as the core problem: train a strong baseline, then deliberately stress-test it against harder, more realistic conditions rather than stopping at a single accuracy number.

## Project structure

The project is organised as three linked stages, using the **same CNN backbone throughout** so results are directly comparable:

1. **Baseline model** — DenseNet121 (ImageNet-pretrained, frozen backbone) fine-tuned on the [TrashNet](https://github.com/garythung/trashnet) dataset (1,796 images across 4 classes).
2. **Controlled augmentation study** — an augmentation pipeline (random-resized crop, flips, rotation, colour jitter) is added under an equal training-budget comparison, plus a pipeline-profiling pass to identify the training bottleneck.
3. **Domain-shift evaluation & targeted improvement** — the model is tested against a second, independently sourced dataset ([RealWaste](https://archive.ics.uci.edu/dataset/908/realwaste), collected at an actual waste recovery facility) to measure how much performance drops outside the clean, studio-lit training conditions, followed by a failure analysis and a revised augmentation strategy targeted at the observed error patterns.

## Key results

| Stage                                       | Test accuracy | Test macro-F1 |
| ------------------------------------------- | ------------: | ------------: |
| Baseline (DenseNet121, frozen backbone)     |         0.859 |         0.863 |
| + Data augmentation (equal training budget) |         0.859 |         0.863 |
| + Targeted augmentation (revised pipeline)  |     **0.870** |     **0.874** |

| Test set                                         | Accuracy | Macro-F1 |
| ------------------------------------------------ | -------: | -------: |
| Original TrashNet test split                     |    0.870 |    0.874 |
| New-domain set (RealWaste, real facility images) |    0.552 |    0.535 |

**Takeaways:**

- The baseline model separates cardboard cleanly (it has a visually distinct matte texture) but confuses glass, metal, and plastic more often — largely due to reflections, transparency, and ambiguous object boundaries.
- A naive augmentation pipeline didn't help on its own; some transforms (aggressive cropping, vertical flips) actually destroyed shape cues the model relied on.
- Revising the augmentation pipeline to preserve object structure (gentler cropping, no vertical flips, added lighting variation) improved same-domain accuracy and macro-F1.
- The sharp drop against the RealWaste facility images is the more important finding: it shows that strong benchmark performance on a clean dataset like TrashNet does **not** guarantee real-world robustness — a key consideration for anyone deploying a sorting model in an actual recycling facility.

## Method details

- **Backbone:** DenseNet121, ImageNet-1K pretrained, feature extractor frozen, classifier head retrained.
- **Training:** Cross-entropy loss, Adam optimiser (lr = 0.001), batch size 32, 10 epochs, stratified 70/15/15 train/validation/test split, fixed random seed for reproducibility.
- **Augmentation (final version):** random-resized crop, horizontal flip, colour jitter, random autocontrast — chosen to add realistic lighting/viewpoint variation without destroying shape information.
- **Evaluation:** accuracy, macro-F1, and confusion matrices on both the held-out TrashNet test set and the independently collected RealWaste subset.

## Tech stack

Python, PyTorch, torchvision, scikit-learn, pandas, matplotlib.

## Data sources

- TrashNet — Thung & Yang, [github.com/garythung/trashnet](https://github.com/garythung/trashnet)
- RealWaste — Single, S. et al. (2023), UCI Machine Learning Repository, [archive.ics.uci.edu/dataset/908/realwaste](https://archive.ics.uci.edu/dataset/908/realwaste)

## Possible extensions

- Fine-tune (rather than freeze) the backbone with a smaller learning rate to close the domain-shift gap.
- Add RealWaste-style images into the training augmentation mix (domain adaptation) rather than treating it purely as a held-out test set.
- Expand beyond 4 material classes to match a real sorting line's full category set.
