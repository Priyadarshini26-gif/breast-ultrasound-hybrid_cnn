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


## Output Files Generated

### Plots (`results/` folder)
- **acc_&_loss_vs_epoch.png** - Training and validation curves
- **confusion_matrix.png** - Classification confusion matrix
- **ROC_curve.png** - Multi-class ROC analysis

Each image shows:
- Original ultrasound image
- Model prediction with confidence
- Score-CAM attention map (highlighted regions)
- True label

## Model Interpretability

### Score-CAM Visualization
The model includes Score-CAM (Score-weighted Class Activation Mapping) to visualize which regions the model uses for classification.

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
