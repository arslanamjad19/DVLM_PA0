# DVLM_PA0
Deep Learning Architectures Analysis Suite
This repository contains a collection of experiments analyzing the internal mechanisms, interpretability, and performance of four foundational deep learning architectures: ResNet, Vision Transformers (ViT), Variational Autoencoders (VAE), and CLIP.

Project Structure
The codebase is divided into four distinct tasks:

1. ResNet & Residual Connections (Task1.ipynb)
Focus: CNN Architectures and Feature Hierarchies.

Objective: Fine-tune a pre-trained ResNet-152 on CIFAR-10.

Experiments:

Standard fine-tuning (freezing backbone, training head).

Ablation Study: "Breaking" the residual connections to observe performance degradation.

Visualization: Using t-SNE to visualize feature separation at early, middle, and late layers.

2. Vision Transformer Analysis (Test_ViT.py, vit_comprehensive_analysis.py)
Focus: Attention Mechanisms and Robustness.

Objective: Perform a deep dive into how ViTs process images using the google/vit-base-patch16-224 model.

Experiments:

Inference: Basic classification on custom images.

Interpretability: Visualizing attention maps and overlays (Attention Rollout) to see where the model "looks."

Robustness: Analyzing performance degradation when random image patches are masked.

Linear Probe: (Optional in code) Comparing CLS token vs. Mean pooling features.

Files:

vit_comprehensive_analysis.py: Contains the core analysis classes (ViTBasicInference, ViTAttentionVisualizer, PatchMaskingAnalyzer).

Test_ViT.py: The execution script to run the analysis.

3. Variational Autoencoder (Task3.ipynb)
Focus: Generative Models and Latent Spaces.

Objective: Train a VAE on the FashionMNIST dataset.

Experiments:

Optimization of the Evidence Lower Bound (ELBO): Reconstruction Loss + KL Divergence.

Visualization: Plotting the latent space clusters colored by class labels.

Generation: (Implied) Latent space sampling.

4. CLIP Modality Gap (Task4.ipynb)
Focus: Multi-modal Alignment.

Objective: Analyze the geometric gap between Image and Text embeddings in the CLIP feature space.

Experiments:

Zero-Shot Classification: Testing different prompting strategies ("Plain", "Prompted", "Descriptive") on STL-10.

Modality Gap Visualization: Using t-SNE to visualize the separation between image and text embeddings.

Alignment: Using Procrustes analysis to align the two modalities and re-evaluating accuracy.

Installation & Prerequisites
This project relies on PyTorch and several ecosystem libraries. You can install the required dependencies using pip:

Bash
# Core DL libraries
pip install torch torchvision numpy matplotlib scikit-learn

# For ViT Analysis
pip install transformers pillow

# For CLIP Analysis (Task 4)
pip install git+https://github.com/openai/CLIP.git
pip install tqdm scipy umap-learn
Note: It is recommended to use a GPU (CUDA) for execution, especially for training the ResNet (Task 1) and VAE (Task 3).

Execution Guide
Task 1: ResNet Analysis
Open Task1.ipynb in Jupyter Notebook or Google Colab.

Ensure runtime is set to GPU for faster training.

Run all cells sequentially. The notebook handles data downloading (CIFAR-10) automatically.

Task 2: Vision Transformer (ViT)
Place your test images in a known directory (e.g., ./images/).

Open Test_ViT.py in a text editor.

Update the MY_TEST_IMAGES list with paths to your local images:

Python
MY_TEST_IMAGES = [
    './images/my_cat.jpg',
    './images/my_car.jpg'
]
Run the script:

Bash
python Test_ViT.py
Visualizations will be saved to the configured output directory (check script for default save paths, usually /content/ or current directory).

Task 3: VAE Training
Open Task3.ipynb in Jupyter Notebook or Google Colab.

Run cells to download FashionMNIST and train the VAE.

The notebook will output the loss curves and a 2D scatter plot of the latent space.

Task 4: CLIP Modality Gap
Open Task4.ipynb in Jupyter Notebook or Google Colab.

Run the installation cell to setup CLIP.

Run the analysis cells. The notebook will perform zero-shot inference on STL-10 and generate t-SNE plots comparing the modality gap before and after Procrustes alignment.
