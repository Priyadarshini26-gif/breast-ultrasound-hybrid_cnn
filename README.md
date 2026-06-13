# Breast Ultrasound CNN: Interpretable Deep Learning with Attention

Hybrid CNN-Transformer model with attention visualization for interpretable breast ultrasound image classification.

## Overview

This project implements a **HybridCNNTransformer** model combining ResNet18 with attention mechanisms for automated breast ultrasound image classification. The model classifies ultrasound images into three categories: **Benign, Malignant, and Normal**, with visual attention maps (Score-CAM) for interpretability.

**Architecture:** ResNet18 backbone + Spatial/Channel Attention + Classification Head  
**Task:** 3-class classification with attention-based interpretability  
**Framework:** PyTorch with Torchvision

## Model Architecture

```
Input Image (Ultrasound)
        ↓
ResNet18 Encoder (pretrained)
        ↓
Attention Mechanism (Conv2d layers)
        ↓
Spatial Pooling
        ↓
Classification Head (Fully Connected)
        ↓
Output (Benign/Malignant/Normal)
```

**Key Components:**
- **Backbone:** ResNet18 (pretrained on ImageNet)
- **Attention:** Spatial & Channel attention layers
- **Interpretability:** Score-CAM visualization of model decisions

## Performance Highlights

**State-of-the-art Results on BUSI Dataset**

- **Overall Accuracy: 96%** - Excellent classification across all three classes
- **Malignant Precision: 100%** - Perfect: no false positives for cancer diagnosis
- **Normal Recall: 100%** - Perfectly identifies all normal cases
- **Macro Average: 96%** - Balanced, reliable performance across classes
- **Validation Samples: 263** - Robust evaluation on diverse ultrasound images

### Clinical Significance

This performance is clinically relevant because:

1. **Zero False Positives for Malignant (100% Precision)**: Critical feature - avoids unnecessary procedures
2. **94% Malignant Recall**: Catches majority of cancerous lesions
3. **100% Normal Recall**: All normal cases identified, avoiding missed diagnoses
4. **High Precision Across Classes**: Reduces clinical uncertainty
5. **Consistent Performance**: Not overfitting to any single class

### Validation Dataset Composition

- Benign samples: 87 (33%)
- Malignant samples: 88 (33%)
- Normal samples: 88 (33%)
- Total: 263 samples with balanced class distribution


## Performance Metrics

### Validation Results (263 samples)

```
              Precision  Recall  F1-Score  Support
Benign           0.95     0.93      0.94       87
Malignant        1.00     0.94      0.97       88
Normal           0.93     1.00      0.96       88

Macro Avg        0.96     0.96      0.96      263
Weighted Avg     0.96     0.96      0.96      263

Overall Accuracy: 96%
```

### Key Metrics Interpretation

| Class | Metric | Value | Analysis |
|-------|--------|-------|----------|
| **Benign** | Precision | 95% | Only 5% false positives |
| | Recall | 93% | Correctly identifies 93% of benign cases |
| | F1-Score | 0.94 | Excellent balanced performance |
| **Malignant** | Precision | 100% | Zero false positives (critical!) |
| | Recall | 94% | Correctly identifies 94% of malignant cases |
| | F1-Score | 0.97 | Outstanding performance |
| **Normal** | Precision | 93% | Few false positives |
| | Recall | 100% | Identifies all normal cases |
| | F1-Score | 0.96 | Excellent balanced performance |

### Model Strengths

**96% Overall Accuracy** - Excellent classification performance  
**100% Precision for Malignant** - No false positives (critical for clinical use)  
**100% Recall for Normal** - All normal cases correctly identified  
**93%+ Performance Across All Classes** - Consistent, reliable model  
**Balanced Performance** - Good precision-recall trade-off  

### Evaluation Metrics Generated

- **Confusion Matrix** - Visualization of classification results (`results/confusion_matrix.png`)
- **ROC-AUC Curves** - Multi-class ROC analysis for benign, malignant, and normal (`results/ROC_curve.png`)
- **Training Curves** - Loss and accuracy over epochs (`results/acc_&_loss_vs_epoch.png`)
- **Sample Predictions** - Attention visualizations for each class (`results/sample_predictions/`)

*See `results/` folder for all generated plots and visualizations*

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/breast-ultrasound-cnn.git
cd breast-ultrasound-cnn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

## Usage

### 1. Open the Notebook

```bash
jupyter notebook breast_ultrasound_analysis.ipynb
```

### 2. Workflow

The notebook includes the following sections:

1. **Data Loading & Preprocessing**
   - Load BUSI dataset
   - Create balanced dataset
   - Apply image preprocessing and augmentation

2. **Model Definition**
   - Define HybridCNNTransformer architecture
   - Load pretrained ResNet18 backbone
   - Initialize attention layers

3. **Training**
   - Train model on balanced dataset
   - Monitor training/validation loss and accuracy
   - Early stopping and model checkpointing

4. **Evaluation**
   - Compute classification metrics (accuracy, precision, recall, F1)
   - Generate confusion matrix
   - Plot ROC curves for each class

5. **Interpretability Analysis**
   - Generate Score-CAM attention maps
   - Visualize model focus regions for each sample
   - Save attention visualizations for benign, malignant, and normal samples

### 3. Make Predictions

```python
# Load trained model
model = HybridCNNTransformer(num_classes=3)
model.load_state_dict(torch.load('path_to_model.pth'))
model.eval()

# Prepare image
image = Image.open('ultrasound.jpg')
image_tensor = val_transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(image_tensor)
    probabilities = torch.softmax(output, dim=1)
    prediction = torch.argmax(probabilities, dim=1)

# Visualize attention map
attention_map = score_cam(image_tensor, model, target_class=prediction.item())
```

## Key Features

### 1. Three-Class Classification
- Distinguishes between Benign, Malignant, and Normal findings
- Critical for clinical decision support

### 2. Attention Mechanism
- Spatial attention identifies important image regions
- Channel attention learns feature importance
- Improves model transparency

### 3. Interpretability with Score-CAM
- Visualizes which regions influence model decisions
- Generates attention maps showing model focus areas
- Helps clinicians understand predictions
- Sample outputs saved for benign, malignant, and normal classes

### 4. Comprehensive Evaluation
- Per-class metrics with classification report
- Confusion matrix for error analysis
- ROC-AUC curves for multi-class evaluation
- Training curves showing convergence

## Training Configuration

```
Device: GPU (CUDA recommended)
Optimizer: Adam
Loss Function: CrossEntropyLoss
Batch Size: 32 (or adjust based on GPU memory)
Epochs: 50+ (with early stopping)
Learning Rate: 0.001 (with scheduler)
Validation Split: 20%
Random Seed: 42 (for reproducibility)
```

## Output Files Generated

### Plots (`results/` folder)
- **acc_&_loss_vs_epoch.png** - Training and validation curves
- **confusion_matrix.png** - Classification confusion matrix
- **ROC_curve.png** - Multi-class ROC analysis

### Sample Predictions (`results/sample_predictions/`)
- **benign.png** - Example benign case with attention map
- **malignant.png** - Example malignant case with attention map
- **normal.png** - Example normal case with attention map

Each image shows:
- Original ultrasound image
- Model prediction with confidence
- Score-CAM attention map (highlighted regions)
- True label

## Model Interpretability

### Score-CAM Visualization
The model includes Score-CAM (Score-weighted Class Activation Mapping) to visualize which regions the model uses for classification:

```python
def score_cam(input_tensor, feature_maps, target_class):
    """
    Generates attention maps showing model focus areas
    """
    # Compute importance weights for each activation map
    # Apply weights to feature maps
    # Upsample to original image size
    return attention_map
```

This helps clinicians:
- Understand model decisions
- Identify if model focuses on relevant anatomical regions
- Build trust in AI-assisted diagnosis
- Detect potential biases or artifacts

## Classes Information

| Class | Label | Typical Findings |
|-------|-------|-----------------|
| Benign | 0 | Non-cancerous growths, cysts, fibroadenomas |
| Malignant | 1 | Cancerous tumors, suspicious lesions |
| Normal | 2 | Normal breast tissue, no abnormalities |

## Data Augmentation

Applied during training:
- Random rotations (±15°)
- Horizontal/vertical flips
- Random brightness adjustment
- Random contrast adjustment
- Small random crops

Helps model generalize to variations in ultrasound acquisition.


## References

- ResNet: He, K., et al. (2016). "Deep Residual Learning for Image Recognition"
- Attention Mechanisms: Woo, S., et al. (2018). "CBAM: Convolutional Block Attention Module"
- Score-CAM: Wang, H., et al. (2020). "Score-CAM: Score-weighted visual explanations for convolutional neural networks"
- BUSI Dataset: Al-Dhabyani, W., et al. (2020). "Dataset of Breast Ultrasound Images"
