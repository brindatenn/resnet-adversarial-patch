# Adversarial Patch Attacks on ResNet-50: Classifier Loss vs Feature-Space Loss

A step-by-step PyTorch implementation of a **universal adversarial patch** against a pretrained ResNet-50, comparing two attack objectives:

1. **Classifier loss:** maximise the cross-entropy of the true label (untargeted).
2. **Feature-space loss:** maximise the L1 distance between clean and patched features at a hidden layer, read out with a forward hook. This is label-free, and similar to the encoder-level objective used in TAPDreamer.

This is a learning reproduction of established techniques, not a new method. The notebook is heavily commented and explains each step, so it can also be read as a tutorial.

## What the notebook does

| Step | Content |
|---|---|
| 1 | Colab setup and GPU check |
| 2 | Load ResNet-50 (ImageNet weights) and the Imagenette dataset |
| 3 | Clean-accuracy baseline, plus reusable evaluation and visualisation functions |
| 4 | Create the patch and a differentiable function that pastes it onto images; random-patch control |
| 5 | Train the patch with the classifier (cross-entropy) loss |
| 6 | Evaluate: accuracy, attack success rate, fixed vs random placement, prediction distribution |
| 7 | Train a second patch with the feature-space L1 loss via a forward hook on `layer3`, and compare |

## Setup

| | |
|---|---|
| Model | ResNet-50, torchvision `IMAGENET1K_V1` weights, frozen, eval mode |
| Data | [Imagenette](https://github.com/fastai/imagenette) (10 ImageNet classes); 2,000 seeded training images to train the patch, all 3,925 validation images to evaluate |
| Patch | 50 × 50 pixels (~5% of a 224 × 224 image), optimised directly in pixel space and clamped to [0, 1] |
| Placement | Random location on every image during training (a position-only form of Expectation over Transformation) |
| Optimiser | Adam, learning rate 0.01, 5 epochs, batch size 32 |
| Predictions | Over all 1,000 ImageNet classes, not just the 10 Imagenette classes |

ImageNet normalisation is applied inside a wrapper around the model, so images and the patch stay in pixel space throughout.

**Attack success rate (ASR)** = the fraction of validation images classified correctly when clean that are misclassified with the patch.

## Results

| Patch | Accuracy | ASR |
|---|---|---|
| None (clean) | 84.54% | – |
| Random noise (control) | 81.86% | – |
| Cross-entropy patch | **0.15%** | **99.88%** |
| Feature-L1 patch (`layer3`) | **51.97%** | **39.39%** |

### Key findings

- **The cross-entropy patch hijacks the model.** Although the attack is untargeted, 99.9% of validation images are classified as *thimble*, almost always at ~100% confidence. The patch converged on a metallic, dimpled texture, consistent with the known texture bias of ImageNet-trained CNNs (Geirhos et al., 2019). It also works when fixed in one corner (4.41% accuracy), confirming that random-placement training generalises across positions.
- **The feature-space patch breaks the model rather than hijacking it.** It reduces accuracy by ~33 points without ever using a label, far more than the ~3-point drop from random noise. Its errors are scattered and mostly near-misses (English springer → Sussex spaniel, cassette player → CD player, French horn → sax), with no dominant wrong class.
- **The two patches look very different.** The cross-entropy patch is object-like; the feature patch is a fine, high-contrast, high-frequency pattern.
- **The feature attack is weaker here, and not yet converged.** Its loss was still falling at the end of training. A likely factor is dilution: the patch covers ~5% of the image, but the L1 loss averages over all 14 × 14 spatial positions of `layer3`, most of which still describe clean content.

## How to run

1. Click the **Open in Colab** badge above.
2. Go to **Runtime → Change runtime type** and select **T4 GPU**.
3. Go to **Runtime → Run all**.

Everything needed (PyTorch, torchvision, matplotlib) is preinstalled in Colab. The dataset (~340 MB) and model weights download automatically. A full run takes roughly 10 minutes on a T4.

## Limitations

- **Single run.** Patch initialisation and placement are random, so exact numbers vary between runs. No averaging over seeds.
- **Digital only.** The patch is pasted into pixel space. It has not been printed or tested against a camera, and it is clearly visible to a human.
- **One model, one layer.** Only ResNet-50 and only `layer3` were tested for the feature loss.
- **Untargeted attacks only.**

## Possible extensions

- Compare feature losses at `layer2`, `layer3` and `layer4`.
- Sweep the patch size to see where attack success saturates.
- Train the feature-space patch for longer.
- Add a targeted variant of the classifier loss.
- Test whether patches transfer to other architectures (e.g. VGG, ViT).

## References

- Brown, T. B., Mané, D., Roy, A., Abadi, M., & Gilmer, J. (2017). *Adversarial Patch.* arXiv:1712.09665.
- Athalye, A., Engstrom, L., Ilyas, A., & Kwok, K. (2018). *Synthesizing Robust Adversarial Examples.* ICML.
- Geirhos, R., et al. (2019). *ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness.* ICLR.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
- Xuanyu Lu, Fengqing Jiang, Kaiyuan Zheng, Yichen Feng, Yaorui Ding, Yuetai Li,
Zhen Xiang, Bhaskar Ramasubramanian, Basel Alomair, Luyao Niu, Radha Poovendran (2026) *TAPDreamer* 
- Imagenette dataset: Jeremy Howard / fast.ai, https://github.com/fastai/imagenette

## Licence

MIT
