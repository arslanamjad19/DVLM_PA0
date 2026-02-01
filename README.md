# DVLM_PA0
To use this as a GitHub `README.md`, copy and paste the block below into a file named `README.md` in the root of your repository.

---

# Deep Learning Architectures & Interpretability Analysis

This repository contains a suite of technical evaluations focusing on four major pillars of modern Deep Learning: **Residual Networks (ResNet)**, **Vision Transformers (ViT)**, **Variational Autoencoders (VAE)**, and **Contrastive Language-Image Pre-training (CLIP)**.

The project covers performance benchmarking, internal mechanism visualization (attention maps), robustness testing, and latent space analysis.

## 📂 Project Structure

| File | Task | Description |
| --- | --- | --- |
| `Task1.ipynb` | **ResNet Analysis** | Fine-tuning ResNet-152 on CIFAR-10 and conducting ablation studies on residual connections. |
| `vit_comprehensive_analysis.py` | **ViT Core** | Backend classes for ViT inference, attention rollout, and patch masking. |
| `Test_ViT.py` | **ViT Execution** | Main script to run the Vision Transformer analysis on custom images. |
| `Task3.ipynb` | **VAE Generative** | Training a VAE on FashionMNIST to visualize latent space clusters and reconstruction. |
| `Task4.ipynb` | **CLIP Modality Gap** | Measuring the geometric gap between text and image embeddings in CLIP using t-SNE and Procrustes alignment. |

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ and a CUDA-enabled GPU (recommended for all tasks).

### Installation

Install the required dependencies via pip:

```bash
# General Requirements
pip install torch torchvision numpy matplotlib scikit-learn pillow

# ViT Requirements
pip install transformers

# CLIP & UMAP (Task 4)
pip install git+https://github.com/openai/CLIP.git
pip install tqdm scipy umap-learn

```

---

## 🛠 Task Breakdown & Execution

### 1. ResNet Hierarchy & Residual Ablation

Located in `Task1.ipynb`, this module explores why "shortcuts" matter.

* **Key Steps:** Fine-tune ResNet-152, visualize feature clustering via t-SNE at different depths, and observe the effect of removing residual blocks.

### 2. Vision Transformer (ViT) Interpretability

This task uses `Test_ViT.py` and the helper script `vit_comprehensive_analysis.py`.

* **Inference:** Classifies images using the `google/vit-base-patch16-224` model.
* **Attention Rollout:** Generates heatmaps to visualize which patches the model attends to most for a prediction.
* **Patch Masking:** Tests model robustness by progressively "blacking out" image patches.

**How to run:**

1. Update `MY_TEST_IMAGES` in `Test_ViT.py` with your image paths.
2. Run: `python Test_ViT.py`

### 3. VAE Latent Space Analysis

Located in `Task3.ipynb`.

* **Objective:** Learns a compressed representation of the FashionMNIST dataset.
* **Analysis:** Includes loss tracking (Reconstruction vs. KL-Divergence) and a scatter plot of the 2D latent means to show how the model clusters different clothing types.

### 4. CLIP Modality Gap & Alignment

Located in `Task4.ipynb`.

* **The Problem:** Even though CLIP aligns text and images, they often reside in separate "cones" in the feature space.
* **Analysis:**
* Performs Zero-Shot classification on the STL-10 dataset.
* Visualizes the "Modality Gap" using t-SNE.
* Applies **Procrustes Alignment** to bridge the gap and evaluates if alignment improves zero-shot accuracy.

---

## 📊 Results and Outputs

* **Visualizations:** All scripts/notebooks produce `.png` or inline plots showing attention maps, latent clusters, and embedding distributions.
* **Logs:** Training progress and accuracy metrics are printed to the console/notebook cells.
