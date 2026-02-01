"""
Comprehensive Vision Transformer (ViT) Analysis
================================================
This script implements:
1. Basic ViT inference with pretrained model
2. Attention visualization for interpretability
3. Attention map analysis and comparison with CNN approaches
4. Patch masking robustness analysis
5. Linear probe comparison (CLS token vs mean pooling)

Author: Claude
Date: 2026-01-28
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import ViTImageProcessor, ViTForImageClassification, ViTModel
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# TASK 1: Basic ViT Model Testing on Custom Images
# ============================================================================

class ViTBasicInference:
    """
    Class to handle basic ViT inference on custom images.
    Uses google/vit-base-patch16-224 pretrained on ImageNet-21k and fine-tuned on ImageNet-1k
    """
    
    def __init__(self, model_name='google/vit-base-patch16-224'):
        """
        Initialize the ViT model and processor.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        print(f"\nInitializing ViT model: {model_name}")
        
        # Load processor (feature extractor) - handles image preprocessing
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        
        # Load pretrained model for classification
        self.model = ViTForImageClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        # Get ImageNet labels
        self.labels = self.model.config.id2label
        
        print(f"Model loaded successfully!")
        print(f"Image size expected: {self.processor.size}")
        print(f"Number of classes: {len(self.labels)}")
    
    def predict(self, image_path):
        """
        Predict class for a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with prediction results
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Process image using the feature extractor
        # This handles resizing, normalization, and tensor conversion
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
        # Get predicted class
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class = self.labels[predicted_class_idx]
        confidence = torch.softmax(logits, dim=-1)[0, predicted_class_idx].item()
        
        # Get top-5 predictions
        probs = torch.softmax(logits, dim=-1)[0]
        top5_prob, top5_idx = torch.topk(probs, 5)
        top5_classes = [(self.labels[idx.item()], prob.item()) 
                        for idx, prob in zip(top5_idx, top5_prob)]
        
        return {
            'image_path': image_path,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'top5_predictions': top5_classes,
            'image': image
        }
    
    def test_on_images(self, image_paths):
        """
        Test model on multiple images and display results.
        
        Args:
            image_paths: List of image file paths
        """
        print("\n" + "="*80)
        print("TASK 1: Testing ViT on Custom Images")
        print("="*80)
        
        results = []
        for img_path in image_paths:
            result = self.predict(img_path)
            results.append(result)
            
            print(f"\nImage: {Path(img_path).name}")
            print(f"Predicted Class: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print("Top-5 Predictions:")
            for i, (cls, prob) in enumerate(result['top5_predictions'], 1):
                print(f"  {i}. {cls}: {prob:.4f}")
        
        # Visualize results
        self._visualize_predictions(results)
        return results
    
    def _visualize_predictions(self, results):
        """Visualize prediction results."""
        n_images = len(results)
        fig, axes = plt.subplots(1, n_images, figsize=(5*n_images, 5))
        if n_images == 1:
            axes = [axes]
        
        for idx, (ax, result) in enumerate(zip(axes, results)):
            ax.imshow(result['image'])
            ax.axis('off')
            title = f"{result['predicted_class']}\n({result['confidence']:.3f})"
            ax.set_title(title, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/task1_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\nVisualization saved to: task1_predictions.png")


# ============================================================================
# TASK 2 & 3: Attention Visualization and Analysis
# ============================================================================

class ViTAttentionVisualizer:
    """
    Visualize and analyze attention maps from ViT for interpretability.
    """
    
    def __init__(self, model_name='google/vit-base-patch16-224'):
        """Initialize model with attention output capability."""
        self.model_name = model_name
        print(f"\nInitializing ViT with attention visualization: {model_name}")
        
        # Load processor
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        
        # Load base model (not for classification) to get attention weights
        self.model = ViTModel.from_pretrained(
            model_name,
            output_attentions=True  # Critical: Enable attention output
        )
        self.model.to(device)
        self.model.eval()
        
        # Also load classification model for predictions
        self.classifier = ViTForImageClassification.from_pretrained(model_name)
        self.classifier.to(device)
        self.classifier.eval()
        
        # Get model configuration
        self.config = self.model.config
        self.num_patches = (self.config.image_size // self.config.patch_size) ** 2
        self.patch_size = self.config.patch_size
        self.image_size = self.config.image_size
        self.num_heads = self.config.num_attention_heads
        self.num_layers = self.config.num_hidden_layers
        
        print(f"Model configuration:")
        print(f"  - Image size: {self.image_size}x{self.image_size}")
        print(f"  - Patch size: {self.patch_size}x{self.patch_size}")
        print(f"  - Number of patches: {self.num_patches}")
        print(f"  - Number of attention heads: {self.num_heads}")
        print(f"  - Number of layers: {self.num_layers}")
    
    def get_attention_maps(self, image_path):
        """
        Extract attention maps from ViT model.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary containing attention weights and predictions
        """
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get model outputs with attention
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Get prediction for context
            classifier_outputs = self.classifier(**inputs)
            logits = classifier_outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            predicted_class = self.classifier.config.id2label[predicted_class_idx]
            confidence = torch.softmax(logits, dim=-1)[0, predicted_class_idx].item()
        
        # Extract attention weights
        # outputs.attentions is a tuple of length num_layers
        # Each element has shape: (batch_size, num_heads, seq_len, seq_len)
        # where seq_len = num_patches + 1 (the +1 is for the CLS token)
        attentions = outputs.attentions
        
        return {
            'image': image,
            'image_path': image_path,
            'attentions': attentions,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'last_layer_attention': attentions[-1]  # Focus on last layer
        }
    
    def extract_cls_attention(self, attention_weights, aggregate='mean'):
        """
        Extract CLS token attention to patches.
        
        Args:
            attention_weights: Attention tensor of shape (batch, heads, seq_len, seq_len)
            aggregate: How to aggregate across heads ('mean', 'max', or head index)
            
        Returns:
            Attention map of shape (num_patches,)
        """
        # attention_weights shape: (1, num_heads, seq_len, seq_len)
        # CLS token is at position 0
        # We want attention FROM CLS token TO all other tokens (patches)
        
        # Extract CLS token's attention to all tokens
        cls_attention = attention_weights[0, :, 0, 1:]  # (num_heads, num_patches)
        
        if aggregate == 'mean':
            # Average across all heads
            cls_attention_aggregated = cls_attention.mean(dim=0)  # (num_patches,)
        elif aggregate == 'max':
            # Take maximum across heads
            cls_attention_aggregated = cls_attention.max(dim=0)[0]
        elif isinstance(aggregate, int):
            # Use specific head
            cls_attention_aggregated = cls_attention[aggregate]
        else:
            raise ValueError(f"Unknown aggregation method: {aggregate}")
        
        return cls_attention_aggregated.cpu().numpy()
    
    def reshape_attention_to_grid(self, attention_vector):
        """
        Reshape 1D attention vector to 2D spatial grid.
        
        Args:
            attention_vector: Array of shape (num_patches,)
            
        Returns:
            2D array of shape (grid_size, grid_size)
        """
        grid_size = int(np.sqrt(self.num_patches))
        attention_grid = attention_vector.reshape(grid_size, grid_size)
        return attention_grid
    
    def visualize_attention_overlay(self, image_path, save_prefix='attention'):
        """
        Create attention map overlay visualization.
        
        Args:
            image_path: Path to image
            save_prefix: Prefix for saved files
        """
        print("\n" + "="*80)
        print("TASK 2 & 3: Attention Visualization and Analysis")
        print("="*80)
        
        # Get attention maps
        result = self.get_attention_maps(image_path)
        image = result['image']
        attentions = result['attentions']
        
        print(f"\nAnalyzing image: {Path(image_path).name}")
        print(f"Predicted class: {result['predicted_class']} ({result['confidence']:.4f})")
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Original image
        ax_orig = fig.add_subplot(gs[0, 0])
        ax_orig.imshow(image)
        ax_orig.set_title(f"Original Image\n{result['predicted_class']}", fontsize=12, fontweight='bold')
        ax_orig.axis('off')
        
        # Last layer attention - mean aggregation
        last_layer_attn = result['last_layer_attention']
        cls_attn_mean = self.extract_cls_attention(last_layer_attn, aggregate='mean')
        attn_grid_mean = self.reshape_attention_to_grid(cls_attn_mean)
        
        # Visualize attention grid
        ax_grid = fig.add_subplot(gs[0, 1])
        im = ax_grid.imshow(attn_grid_mean, cmap='hot', interpolation='nearest')
        ax_grid.set_title('Attention Grid (14x14)\nMean across heads', fontsize=12)
        plt.colorbar(im, ax=ax_grid, fraction=0.046)
        ax_grid.axis('off')
        
        # Create overlay
        ax_overlay = fig.add_subplot(gs[0, 2])
        overlay_img = self._create_attention_overlay(image, attn_grid_mean)
        ax_overlay.imshow(overlay_img)
        ax_overlay.set_title('Attention Overlay\n(Mean aggregation)', fontsize=12)
        ax_overlay.axis('off')
        
        # Upsampled attention
        ax_upsample = fig.add_subplot(gs[0, 3])
        attn_upsampled = self._upsample_attention(attn_grid_mean, (self.image_size, self.image_size))
        ax_upsample.imshow(attn_upsampled, cmap='hot')
        ax_upsample.set_title('Upsampled Attention Map', fontsize=12)
        ax_upsample.axis('off')
        
        # Visualize individual attention heads from last layer
        print(f"\nAnalyzing individual attention heads in last layer...")
        for head_idx in range(min(8, self.num_heads)):
            row = 1 + head_idx // 4
            col = head_idx % 4
            
            cls_attn_head = self.extract_cls_attention(last_layer_attn, aggregate=head_idx)
            attn_grid_head = self.reshape_attention_to_grid(cls_attn_head)
            
            ax_head = fig.add_subplot(gs[row, col])
            overlay_head = self._create_attention_overlay(image, attn_grid_head)
            ax_head.imshow(overlay_head)
            ax_head.set_title(f'Head {head_idx}', fontsize=10)
            ax_head.axis('off')
        
        plt.savefig(f'/mnt/user-data/outputs/{save_prefix}_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Comprehensive visualization saved to: {save_prefix}_comprehensive.png")
        
        # Analyze attention across layers
        self._analyze_attention_across_layers(attentions, image, save_prefix)
        
        # Analyze attention head specialization
        self._analyze_head_specialization(last_layer_attn, image, save_prefix)
        
        # Generate analysis report
        self._generate_attention_analysis_report(result, cls_attn_mean, attn_grid_mean, save_prefix)
    
    def _create_attention_overlay(self, image, attention_grid, alpha=0.6):
        """
        Create semi-transparent attention overlay on image.
        
        Args:
            image: PIL Image
            attention_grid: 2D attention array
            alpha: Transparency factor
            
        Returns:
            Overlaid image as numpy array
        """
        # Resize image to standard size
        img_resized = image.resize((self.image_size, self.image_size))
        img_array = np.array(img_resized).astype(float) / 255.0
        
        # Upsample attention to image size
        attn_upsampled = self._upsample_attention(attention_grid, (self.image_size, self.image_size))
        
        # Normalize attention to [0, 1]
        attn_norm = (attn_upsampled - attn_upsampled.min()) / (attn_upsampled.max() - attn_upsampled.min() + 1e-8)
        
        # Create heatmap (red channel overlay)
        heatmap = np.zeros_like(img_array)
        heatmap[:, :, 0] = attn_norm  # Red channel
        
        # Blend image with heatmap
        overlay = (1 - alpha) * img_array + alpha * heatmap
        overlay = np.clip(overlay, 0, 1)
        
        return overlay
    
    def _upsample_attention(self, attention_grid, target_size):
        """Upsample attention grid to target size using bilinear interpolation."""
        from scipy.ndimage import zoom
        
        zoom_factors = (target_size[0] / attention_grid.shape[0], 
                        target_size[1] / attention_grid.shape[1])
        upsampled = zoom(attention_grid, zoom_factors, order=1)
        return upsampled
    
    def _analyze_attention_across_layers(self, attentions, image, save_prefix):
        """Analyze how attention evolves across layers."""
        print("\nAnalyzing attention evolution across layers...")
        
        num_layers_to_show = min(6, self.num_layers)
        layer_indices = np.linspace(0, self.num_layers-1, num_layers_to_show, dtype=int)
        
        fig, axes = plt.subplots(2, num_layers_to_show, figsize=(4*num_layers_to_show, 8))
        
        for idx, layer_idx in enumerate(layer_indices):
            # Get attention for this layer
            layer_attn = attentions[layer_idx]
            cls_attn = self.extract_cls_attention(layer_attn, aggregate='mean')
            attn_grid = self.reshape_attention_to_grid(cls_attn)
            
            # Show attention grid
            axes[0, idx].imshow(attn_grid, cmap='hot')
            axes[0, idx].set_title(f'Layer {layer_idx}', fontsize=10)
            axes[0, idx].axis('off')
            
            # Show overlay
            overlay = self._create_attention_overlay(image, attn_grid)
            axes[1, idx].imshow(overlay)
            axes[1, idx].axis('off')
        
        plt.suptitle('Attention Evolution Across Layers', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'/mnt/user-data/outputs/{save_prefix}_layer_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Layer evolution visualization saved to: {save_prefix}_layer_evolution.png")
    
    def _analyze_head_specialization(self, last_layer_attn, image, save_prefix):
        """Analyze whether different attention heads are specialized."""
        print("\nAnalyzing attention head specialization...")
        
        # Extract attention for all heads
        head_attentions = []
        for head_idx in range(self.num_heads):
            cls_attn = self.extract_cls_attention(last_layer_attn, aggregate=head_idx)
            head_attentions.append(cls_attn)
        
        head_attentions = np.array(head_attentions)  # (num_heads, num_patches)
        
        # Compute pairwise correlation between heads
        from scipy.stats import pearsonr
        correlations = np.zeros((self.num_heads, self.num_heads))
        
        for i in range(self.num_heads):
            for j in range(self.num_heads):
                corr, _ = pearsonr(head_attentions[i], head_attentions[j])
                correlations[i, j] = corr
        
        # Visualize correlation matrix
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        im1 = ax1.imshow(correlations, cmap='coolwarm', vmin=-1, vmax=1)
        ax1.set_title('Attention Head Correlation Matrix', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Head Index')
        ax1.set_ylabel('Head Index')
        plt.colorbar(im1, ax=ax1)
        
        # Compute diversity metric (1 - mean correlation)
        mean_corr = (correlations.sum() - self.num_heads) / (self.num_heads * (self.num_heads - 1))
        diversity = 1 - mean_corr
        
        # Show distribution of attention concentration
        attention_entropy = []
        for head_idx in range(self.num_heads):
            attn = head_attentions[head_idx]
            attn_prob = attn / attn.sum()
            entropy = -(attn_prob * np.log(attn_prob + 1e-10)).sum()
            attention_entropy.append(entropy)
        
        ax2.bar(range(self.num_heads), attention_entropy, color='steelblue')
        ax2.set_title('Attention Entropy per Head\n(Higher = more distributed)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Head Index')
        ax2.set_ylabel('Entropy')
        ax2.axhline(np.mean(attention_entropy), color='red', linestyle='--', label='Mean')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'/mnt/user-data/outputs/{save_prefix}_head_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Head specialization analysis saved to: {save_prefix}_head_analysis.png")
        print(f"Mean inter-head correlation: {mean_corr:.4f}")
        print(f"Head diversity score: {diversity:.4f}")
        print(f"Attention entropy range: [{min(attention_entropy):.4f}, {max(attention_entropy):.4f}]")
    
    def _generate_attention_analysis_report(self, result, cls_attn, attn_grid, save_prefix):
        """Generate detailed analysis report."""
        print("\n" + "="*80)
        print("ATTENTION ANALYSIS REPORT")
        print("="*80)
        
        # Find top attended patches
        top_patches_idx = np.argsort(cls_attn)[-10:][::-1]
        top_patches_values = cls_attn[top_patches_idx]
        
        grid_size = int(np.sqrt(self.num_patches))
        
        print(f"\nPrediction: {result['predicted_class']} ({result['confidence']:.4f})")
        print(f"\nTop 10 Most Attended Patches:")
        for i, (idx, val) in enumerate(zip(top_patches_idx, top_patches_values), 1):
            row = idx // grid_size
            col = idx % grid_size
            print(f"  {i}. Patch ({row}, {col}): attention = {val:.6f}")
        
        # Compute attention statistics
        print(f"\nAttention Statistics:")
        print(f"  Mean: {cls_attn.mean():.6f}")
        print(f"  Std: {cls_attn.std():.6f}")
        print(f"  Min: {cls_attn.min():.6f}")
        print(f"  Max: {cls_attn.max():.6f}")
        print(f"  Attention concentration (top 10%): {top_patches_values.sum() / cls_attn.sum():.4f}")
        
        # Compute spatial statistics
        center_patches = []
        border_patches = []
        
        for i in range(self.num_patches):
            row = i // grid_size
            col = i % grid_size
            
            # Center region (middle 50%)
            if grid_size // 4 <= row < 3 * grid_size // 4 and \
               grid_size // 4 <= col < 3 * grid_size // 4:
                center_patches.append(cls_attn[i])
            # Border region
            elif row == 0 or row == grid_size - 1 or col == 0 or col == grid_size - 1:
                border_patches.append(cls_attn[i])
        
        print(f"\nSpatial Attention Distribution:")
        print(f"  Center patches mean attention: {np.mean(center_patches):.6f}")
        print(f"  Border patches mean attention: {np.mean(border_patches):.6f}")
        print(f"  Center/Border ratio: {np.mean(center_patches) / (np.mean(border_patches) + 1e-10):.4f}")
        
        # Analysis insights
        print(f"\n" + "="*80)
        print("INTERPRETABILITY INSIGHTS")
        print("="*80)
        
        print("\n1. Attention Focus:")
        if np.mean(center_patches) > np.mean(border_patches) * 1.5:
            print("   ✓ Model focuses more on CENTER regions (likely object-centric)")
        elif np.mean(border_patches) > np.mean(center_patches) * 1.5:
            print("   ⚠ Model focuses more on BORDER regions (may use context cues)")
        else:
            print("   → Model has DISTRIBUTED attention across image")
        
        concentration_ratio = top_patches_values[:5].sum() / cls_attn.sum()
        print(f"\n2. Attention Concentration:")
        print(f"   Top 5 patches account for {concentration_ratio:.1%} of total attention")
        if concentration_ratio > 0.4:
            print("   ✓ Highly FOCUSED attention (strong localization)")
        elif concentration_ratio > 0.25:
            print("   → MODERATE attention spread")
        else:
            print("   ⚠ DIFFUSE attention (may consider many regions)")
        
        print("\n3. Comparison with CNN Approaches (CAM/Grad-CAM):")
        print("   Advantages of ViT Attention:")
        print("   ✓ Direct access to attention weights (no gradient computation needed)")
        print("   ✓ Multi-head attention provides multiple interpretations")
        print("   ✓ Attention maps available at every layer (not just final layer)")
        print("   ✓ Patch-level granularity with explicit spatial relationships")
        print("   ✓ CLS token attention directly shows classification-relevant regions")
        print("\n   Limitations compared to CNNs:")
        print("   - Coarser spatial resolution (14x14 vs full resolution)")
        print("   - Attention may not perfectly correlate with importance")
        print("   - Multiple heads can show conflicting patterns")


# ============================================================================
# TASK 4: Patch Masking Robustness Analysis
# ============================================================================

class PatchMaskingAnalyzer:
    """
    Analyze ViT robustness to missing patches through various masking strategies.
    """
    
    def __init__(self, model_name='google/vit-base-patch16-224'):
        """Initialize model and processor."""
        self.model_name = model_name
        print(f"\nInitializing Patch Masking Analyzer: {model_name}")
        
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTForImageClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        self.config = self.model.config
        self.patch_size = self.config.patch_size
        self.image_size = self.config.image_size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.grid_size = self.image_size // self.patch_size
        
        print(f"Patch configuration: {self.grid_size}x{self.grid_size} = {self.num_patches} patches")
    
    def mask_patches(self, pixel_values, mask_indices, mask_value=0.0):
        """
        Mask specific patches in the input.
        
        Args:
            pixel_values: Tensor of shape (batch, channels, height, width)
            mask_indices: List of patch indices to mask
            mask_value: Value to use for masking
            
        Returns:
            Masked pixel values
        """
        masked_pixels = pixel_values.clone()
        
        for patch_idx in mask_indices:
            row = patch_idx // self.grid_size
            col = patch_idx % self.grid_size
            
            # Calculate pixel coordinates
            y_start = row * self.patch_size
            y_end = y_start + self.patch_size
            x_start = col * self.patch_size
            x_end = x_start + self.patch_size
            
            # Mask the patch
            masked_pixels[:, :, y_start:y_end, x_start:x_end] = mask_value
        
        return masked_pixels
    
    def predict_with_masking(self, image_path, mask_indices):
        """
        Get prediction with specific patches masked.
        
        Args:
            image_path: Path to image
            mask_indices: List of patch indices to mask
            
        Returns:
            Prediction results
        """
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Mask patches
        masked_pixels = self.mask_patches(inputs['pixel_values'], mask_indices)
        inputs['pixel_values'] = masked_pixels
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predicted_class_idx = logits.argmax(-1).item()
            confidence = probs[0, predicted_class_idx].item()
        
        return {
            'predicted_class_idx': predicted_class_idx,
            'predicted_class': self.model.config.id2label[predicted_class_idx],
            'confidence': confidence,
            'top5_probs': probs[0].topk(5)
        }
    
    def analyze_robustness(self, image_path, mask_fractions=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]):
        """
        Analyze robustness to different amounts of random masking.
        
        Args:
            image_path: Path to image
            mask_fractions: List of masking fractions to test
        """
        print("\n" + "="*80)
        print("TASK 4: Patch Masking Robustness Analysis")
        print("="*80)
        
        print(f"\nAnalyzing image: {Path(image_path).name}")
        
        # Get baseline prediction
        baseline = self.predict_with_masking(image_path, [])
        print(f"\nBaseline (no masking):")
        print(f"  Predicted: {baseline['predicted_class']} ({baseline['confidence']:.4f})")
        
        # Test random masking
        print("\n" + "-"*80)
        print("Random Masking Analysis")
        print("-"*80)
        
        random_results = []
        num_trials = 10  # Multiple trials for random masking
        
        for frac in mask_fractions:
            num_masked = int(frac * self.num_patches)
            
            if num_masked == 0:
                random_results.append({
                    'fraction': frac,
                    'num_masked': 0,
                    'accuracies': [1.0],
                    'confidences': [baseline['confidence']],
                    'mean_accuracy': 1.0,
                    'std_accuracy': 0.0,
                    'mean_confidence': baseline['confidence']
                })
                continue
            
            trial_accuracies = []
            trial_confidences = []
            
            for trial in range(num_trials):
                # Random masking
                mask_indices = np.random.choice(self.num_patches, num_masked, replace=False)
                result = self.predict_with_masking(image_path, mask_indices)
                
                # Check if prediction matches baseline
                is_correct = result['predicted_class_idx'] == baseline['predicted_class_idx']
                trial_accuracies.append(float(is_correct))
                trial_confidences.append(result['confidence'])
            
            random_results.append({
                'fraction': frac,
                'num_masked': num_masked,
                'accuracies': trial_accuracies,
                'confidences': trial_confidences,
                'mean_accuracy': np.mean(trial_accuracies),
                'std_accuracy': np.std(trial_accuracies),
                'mean_confidence': np.mean(trial_confidences)
            })
            
            print(f"\nMasking {frac:.1%} ({num_masked} patches):")
            print(f"  Accuracy: {np.mean(trial_accuracies):.3f} ± {np.std(trial_accuracies):.3f}")
            print(f"  Confidence: {np.mean(trial_confidences):.4f}")
        
        # Test structured masking
        print("\n" + "-"*80)
        print("Structured Masking Analysis")
        print("-"*80)
        
        structured_results = self._test_structured_masking(image_path, baseline)
        
        # Visualize results
        self._visualize_masking_results(image_path, random_results, structured_results, baseline)
        
        # Generate insights
        self._generate_masking_insights(random_results, structured_results)
    
    def _test_structured_masking(self, image_path, baseline):
        """Test various structured masking patterns."""
        structured_results = []
        
        # 1. Center masking (mask central 25% of patches)
        center_size = self.grid_size // 2
        center_start = self.grid_size // 4
        center_indices = []
        for i in range(center_start, center_start + center_size):
            for j in range(center_start, center_start + center_size):
                center_indices.append(i * self.grid_size + j)
        
        result_center = self.predict_with_masking(image_path, center_indices)
        structured_results.append({
            'name': 'Center Masking',
            'num_masked': len(center_indices),
            'fraction': len(center_indices) / self.num_patches,
            'correct': result_center['predicted_class_idx'] == baseline['predicted_class_idx'],
            'confidence': result_center['confidence'],
            'predicted_class': result_center['predicted_class']
        })
        
        # 2. Border masking (mask outer ring)
        border_indices = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if i == 0 or i == self.grid_size - 1 or j == 0 or j == self.grid_size - 1:
                    border_indices.append(i * self.grid_size + j)
        
        result_border = self.predict_with_masking(image_path, border_indices)
        structured_results.append({
            'name': 'Border Masking',
            'num_masked': len(border_indices),
            'fraction': len(border_indices) / self.num_patches,
            'correct': result_border['predicted_class_idx'] == baseline['predicted_class_idx'],
            'confidence': result_border['confidence'],
            'predicted_class': result_border['predicted_class']
        })
        
        # 3. Vertical stripe masking
        vertical_indices = []
        col_start = self.grid_size // 3
        col_end = 2 * self.grid_size // 3
        for i in range(self.grid_size):
            for j in range(col_start, col_end):
                vertical_indices.append(i * self.grid_size + j)
        
        result_vertical = self.predict_with_masking(image_path, vertical_indices)
        structured_results.append({
            'name': 'Vertical Stripe',
            'num_masked': len(vertical_indices),
            'fraction': len(vertical_indices) / self.num_patches,
            'correct': result_vertical['predicted_class_idx'] == baseline['predicted_class_idx'],
            'confidence': result_vertical['confidence'],
            'predicted_class': result_vertical['predicted_class']
        })
        
        # 4. Horizontal stripe masking
        horizontal_indices = []
        row_start = self.grid_size // 3
        row_end = 2 * self.grid_size // 3
        for i in range(row_start, row_end):
            for j in range(self.grid_size):
                horizontal_indices.append(i * self.grid_size + j)
        
        result_horizontal = self.predict_with_masking(image_path, horizontal_indices)
        structured_results.append({
            'name': 'Horizontal Stripe',
            'num_masked': len(horizontal_indices),
            'fraction': len(horizontal_indices) / self.num_patches,
            'correct': result_horizontal['predicted_class_idx'] == baseline['predicted_class_idx'],
            'confidence': result_horizontal['confidence'],
            'predicted_class': result_horizontal['predicted_class']
        })
        
        # 5. Checkerboard masking
        checkerboard_indices = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i + j) % 2 == 0:
                    checkerboard_indices.append(i * self.grid_size + j)
        
        result_checkerboard = self.predict_with_masking(image_path, checkerboard_indices)
        structured_results.append({
            'name': 'Checkerboard',
            'num_masked': len(checkerboard_indices),
            'fraction': len(checkerboard_indices) / self.num_patches,
            'correct': result_checkerboard['predicted_class_idx'] == baseline['predicted_class_idx'],
            'confidence': result_checkerboard['confidence'],
            'predicted_class': result_checkerboard['predicted_class']
        })
        
        # Print results
        print("\nStructured Masking Results:")
        for result in structured_results:
            status = "✓ Correct" if result['correct'] else "✗ Incorrect"
            print(f"\n{result['name']} ({result['fraction']:.1%}, {result['num_masked']} patches):")
            print(f"  {status}: {result['predicted_class']} ({result['confidence']:.4f})")
        
        return structured_results
    
    def _visualize_masking_results(self, image_path, random_results, structured_results, baseline):
        """Visualize masking analysis results."""
        # Load original image
        image = Image.open(image_path).convert('RGB')
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 6, hspace=0.35, wspace=0.3)
        
        # Original image
        ax_orig = fig.add_subplot(gs[0, :2])
        ax_orig.imshow(image)
        ax_orig.set_title(f"Original Image\n{baseline['predicted_class']} ({baseline['confidence']:.3f})", 
                         fontsize=12, fontweight='bold')
        ax_orig.axis('off')
        
        # Random masking accuracy curve
        ax_random = fig.add_subplot(gs[0, 2:4])
        fractions = [r['fraction'] for r in random_results]
        accuracies = [r['mean_accuracy'] for r in random_results]
        stds = [r['std_accuracy'] for r in random_results]
        
        ax_random.plot(fractions, accuracies, 'o-', linewidth=2, markersize=8, label='Accuracy')
        ax_random.fill_between(fractions, 
                              np.array(accuracies) - np.array(stds),
                              np.array(accuracies) + np.array(stds),
                              alpha=0.3)
        ax_random.set_xlabel('Fraction of Patches Masked', fontsize=11)
        ax_random.set_ylabel('Accuracy', fontsize=11)
        ax_random.set_title('Random Masking Robustness', fontsize=12, fontweight='bold')
        ax_random.grid(True, alpha=0.3)
        ax_random.set_ylim([-0.05, 1.05])
        ax_random.legend()
        
        # Random masking confidence curve
        ax_conf = fig.add_subplot(gs[0, 4:])
        confidences = [r['mean_confidence'] for r in random_results]
        ax_conf.plot(fractions, confidences, 's-', linewidth=2, markersize=8, color='orange', label='Confidence')
        ax_conf.set_xlabel('Fraction of Patches Masked', fontsize=11)
        ax_conf.set_ylabel('Confidence', fontsize=11)
        ax_conf.set_title('Prediction Confidence', fontsize=12, fontweight='bold')
        ax_conf.grid(True, alpha=0.3)
        ax_conf.legend()
        
        # Visualize structured masking patterns
        image_np = np.array(image.resize((self.image_size, self.image_size)))
        
        structured_patterns = [
            ('Center', self._get_center_mask()),
            ('Border', self._get_border_mask()),
            ('Vertical', self._get_vertical_mask()),
            ('Horizontal', self._get_horizontal_mask()),
            ('Checkerboard', self._get_checkerboard_mask())
        ]
        
        for idx, ((name, mask_indices), result) in enumerate(zip(structured_patterns, structured_results)):
            ax = fig.add_subplot(gs[1, idx])
            
            # Create masked image
            masked_img = self._create_masked_visualization(image_np, mask_indices)
            ax.imshow(masked_img)
            
            status_symbol = "✓" if result['correct'] else "✗"
            color = 'green' if result['correct'] else 'red'
            ax.set_title(f"{status_symbol} {name}\n{result['confidence']:.3f}", 
                        fontsize=10, color=color, fontweight='bold')
            ax.axis('off')
        
        # Comparison bar chart
        ax_comparison = fig.add_subplot(gs[2, :3])
        names = [r['name'] for r in structured_results]
        accuracies = [1.0 if r['correct'] else 0.0 for r in structured_results]
        colors = ['green' if r['correct'] else 'red' for r in structured_results]
        
        bars = ax_comparison.bar(range(len(names)), accuracies, color=colors, alpha=0.7)
        ax_comparison.set_xticks(range(len(names)))
        ax_comparison.set_xticklabels(names, rotation=45, ha='right')
        ax_comparison.set_ylabel('Accuracy', fontsize=11)
        ax_comparison.set_title('Structured Masking Accuracy', fontsize=12, fontweight='bold')
        ax_comparison.set_ylim([0, 1.2])
        ax_comparison.axhline(1.0, color='black', linestyle='--', alpha=0.5)
        ax_comparison.grid(True, alpha=0.3, axis='y')
        
        # Add confidence values on bars
        for bar, result in zip(bars, structured_results):
            height = bar.get_height()
            ax_comparison.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                             f'{result["confidence"]:.2f}',
                             ha='center', va='bottom', fontsize=9)
        
        # Masking statistics
        ax_stats = fig.add_subplot(gs[2, 3:])
        ax_stats.axis('off')
        
        stats_text = "MASKING ANALYSIS SUMMARY\n" + "="*40 + "\n\n"
        stats_text += "Random Masking:\n"
        
        for r in random_results:
            if r['fraction'] > 0:
                stats_text += f"  {r['fraction']:.0%}: Acc={r['mean_accuracy']:.3f}, Conf={r['mean_confidence']:.3f}\n"
        
        stats_text += "\nStructured Masking:\n"
        for r in structured_results:
            status = "✓" if r['correct'] else "✗"
            stats_text += f"  {status} {r['name']}: {r['confidence']:.3f}\n"
        
        ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                     fontsize=10, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.savefig('/mnt/user-data/outputs/task4_masking_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\nMasking analysis visualization saved to: task4_masking_analysis.png")
    
    def _get_center_mask(self):
        """Get center masking indices."""
        center_size = self.grid_size // 2
        center_start = self.grid_size // 4
        indices = []
        for i in range(center_start, center_start + center_size):
            for j in range(center_start, center_start + center_size):
                indices.append(i * self.grid_size + j)
        return indices
    
    def _get_border_mask(self):
        """Get border masking indices."""
        indices = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if i == 0 or i == self.grid_size - 1 or j == 0 or j == self.grid_size - 1:
                    indices.append(i * self.grid_size + j)
        return indices
    
    def _get_vertical_mask(self):
        """Get vertical stripe masking indices."""
        col_start = self.grid_size // 3
        col_end = 2 * self.grid_size // 3
        indices = []
        for i in range(self.grid_size):
            for j in range(col_start, col_end):
                indices.append(i * self.grid_size + j)
        return indices
    
    def _get_horizontal_mask(self):
        """Get horizontal stripe masking indices."""
        row_start = self.grid_size // 3
        row_end = 2 * self.grid_size // 3
        indices = []
        for i in range(row_start, row_end):
            for j in range(self.grid_size):
                indices.append(i * self.grid_size + j)
        return indices
    
    def _get_checkerboard_mask(self):
        """Get checkerboard masking indices."""
        indices = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i + j) % 2 == 0:
                    indices.append(i * self.grid_size + j)
        return indices
    
    def _create_masked_visualization(self, image_np, mask_indices):
        """Create visualization of masked image."""
        masked = image_np.copy()
        
        for idx in mask_indices:
            row = idx // self.grid_size
            col = idx % self.grid_size
            
            y_start = row * self.patch_size
            y_end = y_start + self.patch_size
            x_start = col * self.patch_size
            x_end = x_start + self.patch_size
            
            # Mask with gray color
            masked[y_start:y_end, x_start:x_end] = 128
        
        return masked
    
    def _generate_masking_insights(self, random_results, structured_results):
        """Generate insights from masking analysis."""
        print("\n" + "="*80)
        print("MASKING ROBUSTNESS INSIGHTS")
        print("="*80)
        
        # Analyze random masking degradation
        print("\n1. Random Masking Degradation:")
        
        # Find critical threshold
        critical_threshold = None
        for r in random_results:
            if r['mean_accuracy'] < 0.5:
                critical_threshold = r['fraction']
                break
        
        if critical_threshold:
            print(f"   Critical threshold: {critical_threshold:.1%} of patches")
            print(f"   → Model maintains accuracy up to ~{critical_threshold:.0%} random masking")
        else:
            print(f"   → Model robust even with 50%+ random masking!")
        
        # Analyze confidence degradation
        conf_drop = random_results[-1]['mean_confidence'] / random_results[0]['mean_confidence']
        print(f"\n2. Confidence Degradation:")
        print(f"   Confidence at 50% masking: {conf_drop:.1%} of baseline")
        
        if conf_drop > 0.7:
            print("   → Model remains CONFIDENT despite masking")
        elif conf_drop > 0.4:
            print("   → Model shows MODERATE confidence drop")
        else:
            print("   → Model shows SIGNIFICANT confidence drop")
        
        # Analyze structured masking
        print(f"\n3. Structured Masking Comparison:")
        
        correct_patterns = [r for r in structured_results if r['correct']]
        incorrect_patterns = [r for r in structured_results if not r['correct']]
        
        print(f"   Robust patterns: {len(correct_patterns)}/{len(structured_results)}")
        if correct_patterns:
            print("   Successful patterns:")
            for r in correct_patterns:
                print(f"     ✓ {r['name']}: {r['confidence']:.3f}")
        
        if incorrect_patterns:
            print("   Failed patterns:")
            for r in incorrect_patterns:
                print(f"     ✗ {r['name']}: {r['confidence']:.3f}")
        
        # Key insights
        print(f"\n4. Key Findings:")
        print("   WHY is ViT robust to missing patches?")
        print("   ✓ Self-attention mechanism can attend to available patches")
        print("   ✓ CLS token aggregates information from non-masked patches")
        print("   ✓ Global receptive field from layer 1 (unlike CNNs)")
        print("   ✓ Position embeddings help maintain spatial relationships")
        
        # Compare masking strategies
        center_result = next(r for r in structured_results if r['name'] == 'Center Masking')
        border_result = next(r for r in structured_results if r['name'] == 'Border Masking')
        
        print(f"\n5. Center vs Border Masking:")
        if center_result['correct'] and not border_result['correct']:
            print("   → Object information likely in CENTER (object-centric)")
        elif border_result['correct'] and not center_result['correct']:
            print("   → Object information likely in BORDERS (uses context)")
        elif center_result['correct'] and border_result['correct']:
            print("   → Model uses DISTRIBUTED information (robust)")
        else:
            print("   → Both critical (high masking fraction)")
        
        print(f"\n6. Checkerboard Pattern:")
        checker_result = next(r for r in structured_results if r['name'] == 'Checkerboard')
        if checker_result['correct']:
            print("   ✓ Model can interpolate from scattered patches")
            print("   → Strong evidence of global context understanding")
        else:
            print("   ✗ Disrupted spatial continuity affects performance")


# ============================================================================
# TASK 5: Linear Probe Comparison (CLS vs Mean Pooling)
# ============================================================================

class LinearProbeComparison:
    """
    Compare linear probes trained on CLS token vs mean of patch tokens.
    """
    
    def __init__(self, model_name='google/vit-base-patch16-224'):
        """Initialize model for feature extraction."""
        self.model_name = model_name
        print(f"\nInitializing Linear Probe Comparison: {model_name}")
        
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        self.hidden_size = self.model.config.hidden_size
        print(f"Feature dimension: {self.hidden_size}")
    
    def extract_features(self, image_path, pooling='cls'):
        """
        Extract features from image using specified pooling.
        
        Args:
            image_path: Path to image
            pooling: 'cls' or 'mean'
            
        Returns:
            Feature vector
        """
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        
        if pooling == 'cls':
            # CLS token is at position 0
            features = last_hidden_state[:, 0, :]  # (batch, hidden_size)
        elif pooling == 'mean':
            # Mean over all patch tokens (excluding CLS)
            features = last_hidden_state[:, 1:, :].mean(dim=1)  # (batch, hidden_size)
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
        
        return features.cpu().numpy()
    
    def compare_linear_probes(self, train_image_dir, test_image_dir):
        """
        Compare linear probes with different pooling strategies.
        
        Args:
            train_image_dir: Directory with training images (organized in class subfolders)
            test_image_dir: Directory with test images (organized in class subfolders)
        """
        print("\n" + "="*80)
        print("TASK 5: Linear Probe Comparison (CLS vs Mean Pooling)")
        print("="*80)
        
        # Load dataset
        print("\nLoading dataset...")
        X_train_cls, y_train, X_test_cls, y_test, class_names = self._load_dataset(
            train_image_dir, test_image_dir, pooling='cls')
        
        X_train_mean, _, X_test_mean, _, _ = self._load_dataset(
            train_image_dir, test_image_dir, pooling='mean')
        
        print(f"Training samples: {len(X_train_cls)}")
        print(f"Test samples: {len(X_test_cls)}")
        print(f"Number of classes: {len(class_names)}")
        print(f"Classes: {class_names}")
        
        # Train linear probes
        print("\n" + "-"*80)
        print("Training Linear Probes")
        print("-"*80)
        
        # CLS token probe
        print("\nTraining CLS token probe...")
        clf_cls = LogisticRegression(max_iter=1000, multi_class='ovr', random_state=42)
        clf_cls.fit(X_train_cls, y_train)
        
        y_pred_cls_train = clf_cls.predict(X_train_cls)
        y_pred_cls_test = clf_cls.predict(X_test_cls)
        
        acc_cls_train = accuracy_score(y_train, y_pred_cls_train)
        acc_cls_test = accuracy_score(y_test, y_pred_cls_test)
        
        print(f"CLS Token Probe:")
        print(f"  Train Accuracy: {acc_cls_train:.4f}")
        print(f"  Test Accuracy: {acc_cls_test:.4f}")
        
        # Mean pooling probe
        print("\nTraining Mean Pooling probe...")
        clf_mean = LogisticRegression(max_iter=1000, multi_class='ovr', random_state=42)
        clf_mean.fit(X_train_mean, y_train)
        
        y_pred_mean_train = clf_mean.predict(X_train_mean)
        y_pred_mean_test = clf_mean.predict(X_test_mean)
        
        acc_mean_train = accuracy_score(y_train, y_pred_mean_train)
        acc_mean_test = accuracy_score(y_test, y_pred_mean_test)
        
        print(f"Mean Pooling Probe:")
        print(f"  Train Accuracy: {acc_mean_train:.4f}")
        print(f"  Test Accuracy: {acc_mean_test:.4f}")
        
        # Detailed comparison
        print("\n" + "="*80)
        print("COMPARISON RESULTS")
        print("="*80)
        
        print(f"\nAccuracy Comparison:")
        print(f"  CLS Token:    Train={acc_cls_train:.4f}, Test={acc_cls_test:.4f}")
        print(f"  Mean Pooling: Train={acc_mean_train:.4f}, Test={acc_mean_test:.4f}")
        print(f"  Difference:   Train={abs(acc_cls_train - acc_mean_train):.4f}, Test={abs(acc_cls_test - acc_mean_test):.4f}")
        
        # Determine winner
        if acc_cls_test > acc_mean_test:
            winner = "CLS Token"
            diff = acc_cls_test - acc_mean_test
        else:
            winner = "Mean Pooling"
            diff = acc_mean_test - acc_cls_test
        
        print(f"\n✓ Winner: {winner} (by {diff:.4f} or {diff*100:.2f}%)")
        
        # Per-class analysis
        print("\n" + "-"*80)
        print("Per-Class Performance")
        print("-"*80)
        
        from sklearn.metrics import classification_report
        
        print("\nCLS Token Probe:")
        print(classification_report(y_test, y_pred_cls_test, target_names=class_names, digits=4))
        
        print("\nMean Pooling Probe:")
        print(classification_report(y_test, y_pred_mean_test, target_names=class_names, digits=4))
        
        # Visualize results
        self._visualize_probe_comparison(
            acc_cls_train, acc_cls_test, acc_mean_train, acc_mean_test,
            y_test, y_pred_cls_test, y_pred_mean_test, class_names,
            X_train_cls, X_train_mean, y_train
        )
        
        # Generate insights
        self._generate_probe_insights(
            acc_cls_train, acc_cls_test, acc_mean_train, acc_mean_test,
            winner, diff
        )
    
    def _load_dataset(self, train_dir, test_dir, pooling='cls'):
        """Load and extract features from dataset."""
        from pathlib import Path
        
        def load_from_dir(data_dir):
            X = []
            y = []
            class_names = sorted([d.name for d in Path(data_dir).iterdir() if d.is_dir()])
            class_to_idx = {name: idx for idx, name in enumerate(class_names)}
            
            for class_name in class_names:
                class_dir = Path(data_dir) / class_name
                for img_path in class_dir.glob('*.jpg'):
                    try:
                        features = self.extract_features(str(img_path), pooling=pooling)
                        X.append(features[0])
                        y.append(class_to_idx[class_name])
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
            
            return np.array(X), np.array(y), class_names
        
        X_train, y_train, class_names = load_from_dir(train_dir)
        X_test, y_test, _ = load_from_dir(test_dir)
        
        return X_train, y_train, X_test, y_test, class_names
    
    def _visualize_probe_comparison(self, acc_cls_train, acc_cls_test, acc_mean_train, 
                                    acc_mean_test, y_test, y_pred_cls, y_pred_mean, 
                                    class_names, X_train_cls, X_train_mean, y_train):
        """Visualize comparison results."""
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Accuracy comparison bar chart
        ax1 = fig.add_subplot(gs[0, 0])
        x = np.arange(2)
        width = 0.35
        
        train_accs = [acc_cls_train, acc_mean_train]
        test_accs = [acc_cls_test, acc_mean_test]
        
        bars1 = ax1.bar(x - width/2, train_accs, width, label='Train', alpha=0.8)
        bars2 = ax1.bar(x + width/2, test_accs, width, label='Test', alpha=0.8)
        
        ax1.set_ylabel('Accuracy', fontsize=11)
        ax1.set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['CLS Token', 'Mean Pooling'])
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim([0, 1.1])
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Confusion matrix for CLS
        ax2 = fig.add_subplot(gs[0, 1])
        from sklearn.metrics import confusion_matrix
        cm_cls = confusion_matrix(y_test, y_pred_cls)
        im2 = ax2.imshow(cm_cls, cmap='Blues')
        ax2.set_title('CLS Token Confusion Matrix', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        if len(class_names) <= 10:
            ax2.set_xticks(range(len(class_names)))
            ax2.set_yticks(range(len(class_names)))
            ax2.set_xticklabels(class_names, rotation=45, ha='right')
            ax2.set_yticklabels(class_names)
        plt.colorbar(im2, ax=ax2)
        
        # 3. Confusion matrix for Mean
        ax3 = fig.add_subplot(gs[0, 2])
        cm_mean = confusion_matrix(y_test, y_pred_mean)
        im3 = ax3.imshow(cm_mean, cmap='Oranges')
        ax3.set_title('Mean Pooling Confusion Matrix', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        if len(class_names) <= 10:
            ax3.set_xticks(range(len(class_names)))
            ax3.set_yticks(range(len(class_names)))
            ax3.set_xticklabels(class_names, rotation=45, ha='right')
            ax3.set_yticklabels(class_names)
        plt.colorbar(im3, ax=ax3)
        
        # 4. Feature visualization using t-SNE
        try:
            from sklearn.manifold import TSNE
            
            # Sample for faster computation
            n_samples = min(500, len(X_train_cls))
            indices = np.random.choice(len(X_train_cls), n_samples, replace=False)
            
            # CLS features t-SNE
            ax4 = fig.add_subplot(gs[1, 0])
            X_sample_cls = X_train_cls[indices]
            y_sample = y_train[indices]
            
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            X_tsne_cls = tsne.fit_transform(X_sample_cls)
            
            scatter = ax4.scatter(X_tsne_cls[:, 0], X_tsne_cls[:, 1], 
                                 c=y_sample, cmap='tab10', alpha=0.6, s=30)
            ax4.set_title('CLS Token Features (t-SNE)', fontsize=12, fontweight='bold')
            ax4.set_xlabel('t-SNE 1')
            ax4.set_ylabel('t-SNE 2')
            if len(class_names) <= 10:
                legend = ax4.legend(*scatter.legend_elements(), title="Classes",
                                   labels=class_names, loc='upper right', fontsize=8)
            
            # Mean features t-SNE
            ax5 = fig.add_subplot(gs[1, 1])
            X_sample_mean = X_train_mean[indices]
            
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            X_tsne_mean = tsne.fit_transform(X_sample_mean)
            
            scatter = ax5.scatter(X_tsne_mean[:, 0], X_tsne_mean[:, 1], 
                                 c=y_sample, cmap='tab10', alpha=0.6, s=30)
            ax5.set_title('Mean Pooling Features (t-SNE)', fontsize=12, fontweight='bold')
            ax5.set_xlabel('t-SNE 1')
            ax5.set_ylabel('t-SNE 2')
            if len(class_names) <= 10:
                legend = ax5.legend(*scatter.legend_elements(), title="Classes",
                                   labels=class_names, loc='upper right', fontsize=8)
        except Exception as e:
            print(f"Could not generate t-SNE plots: {e}")
        
        # 5. Statistics summary
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        summary_text = "LINEAR PROBE SUMMARY\n" + "="*35 + "\n\n"
        summary_text += f"CLS Token Probe:\n"
        summary_text += f"  Train Acc: {acc_cls_train:.4f}\n"
        summary_text += f"  Test Acc:  {acc_cls_test:.4f}\n"
        summary_text += f"  Overfitting: {(acc_cls_train - acc_cls_test):.4f}\n\n"
        
        summary_text += f"Mean Pooling Probe:\n"
        summary_text += f"  Train Acc: {acc_mean_train:.4f}\n"
        summary_text += f"  Test Acc:  {acc_mean_test:.4f}\n"
        summary_text += f"  Overfitting: {(acc_mean_train - acc_mean_test):.4f}\n\n"
        
        diff = abs(acc_cls_test - acc_mean_test)
        winner = "CLS" if acc_cls_test > acc_mean_test else "Mean"
        summary_text += f"Winner: {winner}\n"
        summary_text += f"Difference: {diff:.4f} ({diff*100:.2f}%)"
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.savefig('/mnt/user-data/outputs/task5_linear_probe_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\nLinear probe comparison saved to: task5_linear_probe_comparison.png")
    
    def _generate_probe_insights(self, acc_cls_train, acc_cls_test, acc_mean_train, 
                                acc_mean_test, winner, diff):
        """Generate insights about linear probe comparison."""
        print("\n" + "="*80)
        print("LINEAR PROBE INSIGHTS")
        print("="*80)
        
        print(f"\n1. Performance Winner: {winner}")
        if diff < 0.02:
            print("   → Performance is VERY SIMILAR (< 2% difference)")
            print("   → Both pooling methods capture comparable information")
        elif diff < 0.05:
            print("   → MODERATE performance difference (2-5%)")
            print(f"   → {winner} pooling is slightly better for this task")
        else:
            print("   → SIGNIFICANT performance difference (> 5%)")
            print(f"   → {winner} pooling is clearly superior for this task")
        
        print(f"\n2. Generalization Analysis:")
        cls_gap = acc_cls_train - acc_cls_test
        mean_gap = acc_mean_train - acc_mean_test
        
        print(f"   CLS Token overfitting:  {cls_gap:.4f}")
        print(f"   Mean Pool overfitting:  {mean_gap:.4f}")
        
        if cls_gap < mean_gap:
            print("   → CLS token generalizes BETTER (less overfitting)")
        elif mean_gap < cls_gap:
            print("   → Mean pooling generalizes BETTER (less overfitting)")
        else:
            print("   → Both methods have similar generalization")
        
        print(f"\n3. Why Does {winner} Perform Better?")
        
        if winner == "CLS Token":
            print("   Possible reasons:")
            print("   ✓ CLS token is EXPLICITLY trained for classification")
            print("   ✓ Self-attention allows CLS to aggregate discriminative features")
            print("   ✓ CLS token learns task-specific representations during pretraining")
            print("   ✓ Supervised pretraining objective (classification) aligns with probe task")
        else:
            print("   Possible reasons:")
            print("   ✓ Mean pooling captures MORE COMPLETE spatial information")
            print("   ✓ Averaging reduces noise and outliers in individual patches")
            print("   ✓ Less dependent on specific pretraining objective")
            print("   ✓ May work better when object spans multiple patches")
        
        print(f"\n4. Interaction with Pretraining Objectives:")
        print("   Current model pretraining: Supervised ImageNet-21k + ImageNet-1k")
        print("   ")
        print("   Effect on pooling methods:")
        print("   • Supervised pretraining → CLS token advantage")
        print("     - CLS explicitly trained as classifier")
        print("     - Direct optimization for discrimination")
        print("   ")
        print("   Alternative pretraining scenarios:")
        print("   • MAE (Masked Autoencoder) pretraining:")
        print("     - CLS token less emphasized during pretraining")
        print("     - Mean pooling might perform BETTER")
        print("     - Patch tokens learn reconstruction features")
        print("   ")
        print("   • Contrastive learning (e.g., DINO, MoCo-v3):")
        print("     - CLS token used for contrastive loss")
        print("     - CLS should have strong advantage")
        print("     - Learns discriminative global features")
        print("   ")
        print("   • Self-supervised masking (e.g., BEiT):")
        print("     - Patch tokens are primary focus")
        print("     - Mean pooling likely competitive or better")
        
        print(f"\n5. Practical Recommendations:")
        if acc_cls_test >= acc_mean_test:
            print("   ✓ Use CLS token for this task")
            print("   ✓ Especially good for: classification, retrieval")
        else:
            print("   ✓ Use Mean pooling for this task")
            print("   ✓ Especially good for: dense prediction, localization")
        
        print("\n   General guidelines:")
        print("   • CLS token: Best for supervised/contrastive pretraining")
        print("   • Mean pooling: More robust, less pretraining-dependent")
        print("   • Consider ensemble of both for critical applications")
        print("   • Task complexity matters: simple tasks may not show big differences")


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def create_sample_images():
    """
    Create sample images for testing if user doesn't provide any.
    This is just a helper function - users should use their own images.
    """
    print("\nCreating sample images for demonstration...")
    
    from PIL import ImageDraw, ImageFont
    
    # Create simple colored images with shapes
    sample_dir = Path('/mnt/user-data/outputs/sample_images')
    sample_dir.mkdir(exist_ok=True, parents=True)
    
    # Image 1: Red circle
    img1 = Image.new('RGB', (224, 224), color='white')
    draw = ImageDraw.Draw(img1)
    draw.ellipse([50, 50, 174, 174], fill='red', outline='black', width=3)
    img1.save(sample_dir / 'red_circle.jpg')
    
    # Image 2: Blue square
    img2 = Image.new('RGB', (224, 224), color='white')
    draw = ImageDraw.Draw(img2)
    draw.rectangle([50, 50, 174, 174], fill='blue', outline='black', width=3)
    img2.save(sample_dir / 'blue_square.jpg')
    
    # Image 3: Green triangle
    img3 = Image.new('RGB', (224, 224), color='white')
    draw = ImageDraw.Draw(img3)
    draw.polygon([(112, 50), (50, 174), (174, 174)], fill='green', outline='black', width=3)
    img3.save(sample_dir / 'green_triangle.jpg')
    
    print(f"Sample images created in: {sample_dir}")
    return [str(sample_dir / 'red_circle.jpg'),
            str(sample_dir / 'blue_square.jpg'),
            str(sample_dir / 'green_triangle.jpg')]


def main():
    """
    Main execution function to run all 5 tasks.
    """
    print("="*80)
    print("COMPREHENSIVE VISION TRANSFORMER (ViT) ANALYSIS")
    print("="*80)
    print("\nThis script implements:")
    print("1. Basic ViT inference on custom images")
    print("2. Attention visualization for interpretability")
    print("3. Attention map analysis")
    print("4. Patch masking robustness analysis")
    print("5. Linear probe comparison (CLS vs Mean pooling)")
    print("="*80)
    
    # Create output directory
    output_dir = Path('/mnt/user-data/outputs')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # ========================================================================
    # TASK 1: Basic ViT Testing
    # ========================================================================
    
    print("\n\nSTARTING TASK 1: Basic ViT Model Testing")
    print("="*80)
    
    # For demonstration, create sample images
    # Users should replace these with their own images
    image_paths = create_sample_images()
    
    # Alternatively, users can specify their own images:
    # image_paths = [
    #     '/path/to/your/image1.jpg',
    #     '/path/to/your/image2.jpg',
    #     '/path/to/your/image3.jpg'
    # ]
    
    vit_basic = ViTBasicInference()
    results = vit_basic.test_on_images(image_paths)
    
    # ========================================================================
    # TASK 2 & 3: Attention Visualization and Analysis
    # ========================================================================
    
    print("\n\nSTARTING TASKS 2 & 3: Attention Visualization and Analysis")
    print("="*80)
    
    vit_attention = ViTAttentionVisualizer()
    
    # Visualize attention for the first image
    vit_attention.visualize_attention_overlay(image_paths[0], save_prefix='image1_attention')
    
    # If you want to analyze more images:
    # for idx, img_path in enumerate(image_paths, 1):
    #     vit_attention.visualize_attention_overlay(img_path, save_prefix=f'image{idx}_attention')
    
    # ========================================================================
    # TASK 4: Patch Masking Robustness
    # ========================================================================
    
    print("\n\nSTARTING TASK 4: Patch Masking Robustness Analysis")
    print("="*80)
    
    masking_analyzer = PatchMaskingAnalyzer()
    masking_analyzer.analyze_robustness(
        image_paths[0],
        mask_fractions=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    )
    
    # ========================================================================
    # TASK 5: Linear Probe Comparison
    # ========================================================================
    
    print("\n\nSTARTING TASK 5: Linear Probe Comparison")
    print("="*80)
    
    # Note: This requires a proper dataset with train/test splits
    # Users need to organize their data as:
    # train_dir/
    #   class1/
    #     img1.jpg
    #     img2.jpg
    #   class2/
    #     img1.jpg
    #     img2.jpg
    # test_dir/
    #   class1/
    #     img1.jpg
    #   class2/
    #     img1.jpg
    
    print("\nNote: Task 5 requires a proper dataset with train/test splits.")
    print("Please organize your data in the following structure:")
    print("  train_dir/class1/, train_dir/class2/, ...")
    print("  test_dir/class1/, test_dir/class2/, ...")
    print("\nThen uncomment and modify the following lines:")
    
    # probe_comparison = LinearProbeComparison()
    # probe_comparison.compare_linear_probes(
    #     train_image_dir='/path/to/your/train/directory',
    #     test_image_dir='/path/to/your/test/directory'
    # )
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nAll visualizations and results have been saved to:")
    print(f"  {output_dir}")
    print("\nGenerated files:")
    print("  - task1_predictions.png")
    print("  - image1_attention_comprehensive.png")
    print("  - image1_attention_layer_evolution.png")
    print("  - image1_attention_head_analysis.png")
    print("  - task4_masking_analysis.png")
    print("  - task5_linear_probe_comparison.png (if Task 5 is run)")


if __name__ == "__main__":
    main()
