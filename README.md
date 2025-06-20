# Deepfake Detection & Interpretability System

## 🔍 Project Overview

A web-based application that detects **deepfake images** using two powerful models:

* A **custom CNN** model built from scratch
* A **pre-trained EfficientNet B0** fine-tuned for binary classification

The project not only predicts whether an image is real or fake but also explains *why* using **Grad-CAM visualizations**, giving insight into which parts of the image influenced the model.

## 🌟 Key Features

* 🔍 **Image-based Deepfake Detection**
* 📊 **Dual-model Inference** (CNN and EfficientNet)
* 🔥 **Dynamic Grad-CAM** overlay from the more confident model
* 🧠 **Explainable AI** integration (Grad-CAM)
* 🖼️ **User Interface** for drag-and-drop prediction
* 🌙 **Dark/Light mode** toggle

## 🧠 Technology Stack

### 🔙 Backend

* Python 3.11
* Flask (REST API)
* PyTorch
* torchvision (EfficientNet, transforms)
* grad-cam
* Pillow, OpenCV
* Flask-CORS

### 💻 Frontend

* HTML5 + CSS3 (Tailwind)
* Vanilla JavaScript

## 🚀 Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Shubhraa03/Image-Deepfake-Dectection-.git
cd deepfake-detection
```

### 2. Create & Activate Virtual Environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate.bat
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Trained Models

Put these files in `saved_models/`:

* `final_EfficientNet_model.pth`
* `cnn_model.pth`

Folder structure:

```
saved_models/
├── final_EfficientNet_model.pth
└── cnn_model.pth
```

## 📦 Dataset

This project uses a deepfake image dataset available on Kaggle.

🔗 [Kaggle - Deepfake and Real Images Dataset](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)

### Dataset Structure (after manual download):

```
data/deepfake/
├── train/
│   ├── Fake/
│   └── Real/
├── validation/
│   ├── Fake/
│   └── Real/
└── test/
    ├── Fake/
    └── Real/
```

## 💻 Usage

### 1. Start the Backend

```bash
# Activate venv if not already:
venv\Scripts\activate.bat

# Run the backend
python Backend/withUpdatedEff.py
```

### 2. Open the Frontend

Open `Frontend/maychoose.html` in your browser manually or use VS Code Live Server.

### 3. Upload Image for Analysis

* Drag and drop or upload an image
* View model predictions
* See Grad-CAM heatmap from the model with higher confidence

## 📁 Project Structure

```
deepfake-detection/
├── Backend/
│   └── withUpdatedEff.py      # Flask backend
├── Frontend/
│   └── maychoose.html         # Web UI
├── models/
│   ├── cnn.py                 # CNN model architecture
│   └── efficientnet_model.py # EfficientNet loading
├── saved_models/
│   ├── cnn_model.pth
│   └── final_EfficientNet_model.pth
├── train.py                  # CNN training script
├── test.py                   # CNN testing script
├── finalEfficientNet_train.py   # EfficientNet training
├── testEfficientNet.py          # EfficientNet testing
├── requirements.txt
├── config/
│   └── config.yaml           # (optional)
├── utils/
│   └── dataset_loader.py     # CNN dataset loader
└── README.md
```

## 📊 Model Evaluation & Explainability

* Both CNN and EfficientNet are trained for **binary classification**:

  * `Real` → class `1`
  * `Fake` → class `0`
* CNN uses raw image tensors with no normalization.
* EfficientNet uses standard image **normalization**:

  * Mean: `[0.485, 0.456, 0.406]`
  * Std: `[0.229, 0.224, 0.225]`
* Models output confidence scores, and the **most confident** model is used for **Grad-CAM** heatmap generation.

## 🌱 Future Improvements

* 🔁 Support for video deepfake detection
* 🔍 Include Vision Transformers (ViTs)
* 📈 Dashboard with performance metrics
* 📦 Docker containerization
* 👥 Feedback loop for user-reported accuracy

## 🙏 Acknowledgements

* [PyTorch](https://pytorch.org)
* [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
* [Kaggle Deepfake Dataset](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)
* Tailwind CSS

---

*This project was developed as part of an academic initiative on Deepfake Detection and Explainable AI.*
