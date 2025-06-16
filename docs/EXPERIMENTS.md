# Experimentation Guide

This file provides a set of guided experiments to help learners and researchers extend the base U-ViT model. These experiments are meant to reinforce foundational knowledge, test hypotheses, and possibly lead to improved performance or insights. Contributions are welcome!

This file will adhere to the following structured approach for each experiment: 

#### Experiment Name
- **Objective**: What’s being tested or changed?
- **Why it matters**: The intuition behind it.
- **Expected impact**: Hypothesize what might happen.
- **How to implement**: A short summary or pointer in the codebase.
- **Optional papers**: Links to references if relevant.
- **Screenshot of actual in-code implementation**: Look at this after attempting the experiment yourself. 

## Architecture Experiments

### Replace ViT with Swin Transformer
- **Objective**: Swap the vanilla ViT encoder with a hierarchical Swin Transformer.
- **Why it matters**: Swin uses local attention and hierarchical design, potentially capturing finer structures in medical images.
- **Expected impact**: Better boundary delineation and generalization.
- **How to implement**: Replace `pass_through_vit()` with a Swin block. Consider using `timm` or `keras-cv`.
- **Reference**: [Swin Transformer (Liu et al., 2021)](https://arxiv.org/abs/2103.14030)

**Screenshot:**

### Add Skip Connections Across ViT Block
- **Objective**: Reconnect the ViT bottleneck with encoder features.
- **Why it matters**: Enables richer spatial context retention.
- **Expected impact**: Improved fine-detail segmentation.
- **How to implement**: Add skip connections before and after the ViT block and handle shape alignment.

**Screenshot:**

---

## Training & Optimization Experiments

### Use Dice + Focal Loss
- **Objective**: Improve segmentation on difficult boundary regions.
- **Why it matters**: Focal loss emphasizes hard examples.
- **Expected impact**: Better class balance, especially with small masks.
- **How to implement**: Swap current loss with a hybrid Dice + Focal formulation.

**Screenshot:**

### Use a Learning Rate Scheduler
- **Objective**: Improve convergence behavior.
- **How to implement**: Try cosine decay, exponential decay, or cyclical learning rates using `tf.keras.callbacks.LearningRateScheduler`.

**Screenshot:**

---

## Data Augmentation Experiments

### Add Elastic Deformations
- **Objective**: Simulate more realistic anatomical variability.
- **Why it matters**: Helps the model generalize to real-world shape variations.
- **How to implement**: Use `imgaug`, `albumentations`, or TensorFlow's custom functions.

**Screenshot:**

---

## Explainability Experiments

### Apply Grad-CAM to ViT Outputs
- **Objective**: Visualize what the ViT is focusing on.
- **How to implement**: Use Grad-CAM or attention rollout on patch embeddings.

**Screenshot:**

### Compare Predictions with and without Attention
- **Objective**: Quantify attention’s real value in segmentation.
- **How to implement**: Run ablation: remove the ViT block entirely and analyze differences.

**Screenshot:**

---

## Real-World Extension Experiments

### Transfer to CT Scan Lung Masks
- **Objective**: Use model on different modality.
- **Why it matters**: Real-world medical imaging varies widely.
- **Expected impact**: Helps assess model transferability.
- **How to implement**: Adjust preprocessing pipeline to handle Hounsfield units, input channels, and depth (3D).

**Screenshot:**

### Add Clinical Metadata as Side Input
- **Objective**: Fuse patient age, gender, or comorbidity data with image features.
- **How to implement**: Add an MLP for structured input and concatenate to encoder bottleneck.

**Screenshot:**

---

## Notes for Contributors

- Feel free to fork and submit a pull request with new experiments!
- Be sure to update the README or include results in `results/` if you run any experiments from this list.

Acknowledgement:
*Some content in this guide was developed with the help of [ChatGPT](https://chat.openai.com), based on custom project context and ideas.*