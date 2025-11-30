# Solar Panel Defect Detection: SOTA Model Comparison

Comparative benchmark of six state-of-the-art (**SOTA**) computer vision architectures for the **binary classification** of solar panel defects (**crack vs. hotspot**) using ultraviolet (UV) and thermal imaging data.

This project focuses on evaluating **classification performance**, **computational efficiency**, and the suitability of diverse model families (CNNs, Vision Transformers, Object Detectors, Vision-Language Models) for industrial quality control applications.

---

## 🔬 Models Evaluated

The following architectures were fine-tuned and benchmarked for this specific task:

* **YOLOv12-XL:** An **Object Detection** approach used for its dual capability (classification + localization).
* **Qwen 3 VLM-8B:** A **Vision-Language Model (VLM)** fine-tuned with **LoRA** for feature extraction.
* **Swin Transformer V2:** A highly performant **Vision Transformer** architecture.
* **ConvNeXt V2:** A modern, highly effective **Convolutional Architecture** that blends ViT design with CNN structure.
* **ModernCNN (Custom):** A novel, efficient **Custom Architecture** incorporating dual attention mechanisms.
* **EfficientNet-B0:** An efficient, scalable **CNN Baseline**.
* **ResNet50:** A classical, robust **CNN Baseline**.

---

## 📊 Dataset Overview

The dataset consists of ultraviolet and thermal images of solar panels, categorized into two primary defect types.

| Metric | Details |
| :--- | :--- |
| **Training Set** | 1,000+ images |
| **Validation Set** | 329 images |
| **Test Set** | 164 images |
| **Classes** | Crack (0), Hotspot (1) |
| **Format** | Image folder structure with a metadata CSV file |

---

## 🛠️ Methodology & Custom Architecture

### 1. Fine-tuning Approach

| Model Family | Framework/Method | Notes |
| :--- | :--- | :--- |
| **Vision Transformers (Swin V2, ConvNeXt V2)** | HuggingFace Trainer API | Standard transfer learning. |
| **Vision-Language (Qwen 3 VLM)** | Unsloth + LoRA | Targeted fine-tuning of vision layers, attention, and MLPs for efficiency. |
| **Object Detector (YOLOv12-XL)** | Ultralytics Framework | End-to-end detection training. |
| **PyTorch Models (ModernCNN, ResNet, EfficientNet)** | Custom training loops | Used AdamW optimizer and cosine scheduling. |

### 2. Custom Architecture: ModernCNN

The **ModernCNN** was designed for parameter efficiency and performance by combining key SOTA design principles:

* **Depthwise Separable Convolutions:** Reduces computational cost and parameter count.
* **Inverted Bottleneck Blocks:** (Expand-Depthwise-Compress) for efficiency, inspired by MobileNet V2.
* **Dual Attention:**
    * **Squeeze-Excitation (SE):** Channel-wise feature re-calibration.
    * **Coordinate Attention (CA):** Spatial attention mechanism that captures positional information.
* **Residual Connections:** Used throughout for stable training and progressive downsampling.

#### ModernCNN Block Structure
$$\text{Input} \to \text{LayerNorm} \to 1 \times 1 \text{ Conv (expand)} \to \text{GELU} \to$$
$$\to 7 \times 7 \text{ Depthwise Conv} \to \text{GELU} \to \text{SE Attention} \to$$
$$\to \text{Coordinate Attention} \to 1 \times 1 \text{ Conv (compress)} \to$$
$$\to \text{Dropout} \to \text{Add Residual} \to \text{Output}$$


---

## 🏆 Results on Test Set

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **ModernCNN** | **1.000** | **1.000** | **1.000** | **1.000** |
| **Swin V2** | **1.000** | **1.000** | **1.000** | **1.000** |
| **ResNet50** | **1.000** | **1.000** | **1.000** | **1.000** |
| YOLOv12-XL | 0.963 | 0.963 | 0.963 | 0.963 |
| Qwen 3 VLM | 0.762 | 0.573 | 0.762 | 0.653 |
| EfficientNet-B0 | 0.640 | 0.732 | 0.640 | 0.673 |

---

## 💡 Key Findings

1.  **High Separability:** The high-performing CNN (ModernCNN, ResNet50) and Transformer (Swin V2) architectures achieved **perfect accuracy** (1.000) on this specific dataset, suggesting a high degree of visual separability between crack and hotspot defects in the UV/thermal images.
2.  **Efficiency Wins:** The **ModernCNN** matched the SOTA performance of the Vision Transformer (Swin V2) while maintaining **significantly lower computational cost** and faster inference speed due to its efficient design choices (depthwise separable convolutions, dual attention).
3.  **VLM Underperformance:** The Vision-Language Model (**Qwen 3 VLM**) and the EfficientNet baseline significantly underperformed. This confirms that complex VLMs are better suited for **multimodal reasoning** and tasks requiring natural language grounding, rather than simple, high-throughput pattern recognition.
4.  **Localization Tradeoff:** **YOLOv12-XL** provides valuable **spatial localization** of the defect, which is critical for real-world application, at a minor cost to overall classification accuracy compared to the pure classification models.

---

## 💻 Technical Stack

### Frameworks & Libraries
* **PyTorch**
* **HuggingFace Transformers**
* **Ultralytics** (for YOLOv12)
* **Unsloth** (for LoRA fine-tuning of Qwen 3 VLM)
* **Scikit-learn** (for metric evaluation)

### Training Configuration
* **Optimizer:** AdamW (weight decay: 0.01)
* **Learning Rate:** $1\text{e-3}$ to $5\text{e-5}$ (model-dependent)
* **Epochs:** 10
* **Batch Size:** 16-32
* **Optimization:** Mixed precision (FP16), gradient accumulation, cosine annealing schedule.
* **Augmentation:** Resize, RandomCrop, HorizontalFlip, Normalize.
* **Hardware:** NVIDIA L4 GPU (Google Colab Pro).

### Jupyter Notebooks
* `Fine_Tuning_YOLOv12,_Qwen_3_VLM,_Building_ConvXNet2,Training_SwinNet(Microsoft),_Building_Custom_CNN,_train_ResNet_and_EfficientNet.ipynb`

---

## 🚀 Future Work

This project lays the foundation for a robust E2E industrial quality control system. Next steps include:

* **Multi-class Classification:** Expand the model's capability to detect and classify additional defect types (e.g., corrosion, delamination, snail trails).
* **Instance Segmentation:** Utilize models like **SAM 3** to achieve pixel-level segmentation of defects and conditions, which is crucial for repair and quality assessment.
* **Deployment Optimization:** Convert models to production-ready formats like **ONNX** and optimize inference using **TensorRT**. Possibly utilize GCP Vertex AI to reduce training time by 30-60% and test our top models by performance (YOLOv12, Custom CNN, SwinTransformerV2) with ONNX and TensorRT

---

## 👤 Author

Kevlar

---
