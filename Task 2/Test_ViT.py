from vit_comprehensive_analysis import (
    ViTBasicInference,
    ViTAttentionVisualizer,
    PatchMaskingAnalyzer,
    LinearProbeComparison
)

# ============================================================================
# CONFIGURATION - EDIT THESE PATHS
# ============================================================================

MY_TEST_IMAGES = [
    '/content/grey-heron.jpg',
    '/content/Maltese_dog.JPEG',
    '/content/parakeet.jpg'
]

# For Task 5 (Linear Probes) - optional
MY_TRAIN_DIR = 'path/to/train/directory'  # Contains class subdirectories
MY_TEST_DIR = 'path/to/test/directory'    # Contains class subdirectories

# ============================================================================
# TASK 1: Basic Inference
# ============================================================================

print("="*80)
print("TASK 1: Testing ViT on Our Images")
print("="*80)

vit = ViTBasicInference()
results = vit.test_on_images(MY_TEST_IMAGES)

# Access individual results
for result in results:
    print(f"\nImage: {result['image_path']}")
    print(f"Prediction: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.3f}")

# ============================================================================
# TASK 2 & 3: Attention Visualization
# ============================================================================

print("\n" + "="*80)
print("TASKS 2 & 3: Attention Visualization")
print("="*80)

visualizer = ViTAttentionVisualizer()

# Analyze first image
visualizer.visualize_attention_overlay(
    MY_TEST_IMAGES[0],
    save_prefix='my_image1'
)

# for idx, img_path in enumerate(MY_TEST_IMAGES, 1):
#     visualizer.visualize_attention_overlay(img_path, save_prefix=f'image{idx}')

# ============================================================================
# TASK 4: Patch Masking Robustness
# ============================================================================

print("\n" + "="*80)
print("TASK 4: Patch Masking Robustness")
print("="*80)

analyzer = PatchMaskingAnalyzer()

# Test robustness on first image
analyzer.analyze_robustness(
    MY_TEST_IMAGES[0],
    mask_fractions=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
)

# Advanced: Test specific masking pattern
# mask_indices = [0, 1, 2, 14, 15, 16]  # Specific patches to mask
# result = analyzer.predict_with_masking(MY_TEST_IMAGES[0], mask_indices)
# print(f"Prediction with masking: {result['predicted_class']}")

# ============================================================================
# TASK 5: Linear Probe Comparison (Optional)
# ============================================================================

print("\n" + "="*80)
print("TASK 5: Linear Probe Comparison")
print("="*80)

# Uncomment if have a proper dataset with train/test splits:
# probe = LinearProbeComparison()
# probe.compare_linear_probes(
#     train_image_dir=MY_TRAIN_DIR,
#     test_image_dir=MY_TEST_DIR
# )

# Alternative: Extract features for custom use
# probe = LinearProbeComparison()
# features_cls = probe.extract_features(MY_TEST_IMAGES[0], pooling='cls')
# features_mean = probe.extract_features(MY_TEST_IMAGES[0], pooling='mean')
# print(f"CLS features shape: {features_cls.shape}")
# print(f"Mean features shape: {features_mean.shape}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nCheck the 'outputs' directory for all visualizations!")
