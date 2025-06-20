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

### 5. (Optional) Dataset Structure for Training

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
python withUpdatedEff.py
```

### 2. Open the Frontend

Open `Frontend/index.html` in your browser manually or use VS Code Live Server.

### 3. Upload Image for Analysis

* Drag and drop or upload an image
* View model predictions
* See Grad-CAM heatmap from the model with higher confidence

## 📁 Project Structure

```
deepfake-detection/
├── app.py                  # Flask backend(currently named as withUpdatedEff.py under the backend folder)
├── Frontend/
│   └── index.html          # Web UI (currently named as maychoose.html)
├── models/
│   └── cnn.py              # CNN model class
├── saved_models/
│   ├── cnn_model.pth
│   └── efficientnet_deepfake_best.pth
├── train.py                # CNN training script
├── test.py                 # CNN testing script
├── finalEfficientNet_train.py    # EfficientNet training script
├── testEfficientNet.py    # EfficientNet evaluation
├── requirements.txt
└── README.md
```

## 📊 Model Evaluation & Explainability

* CNN and EfficientNet both output confidence scores
* The model with higher confidence is used for Grad-CAM
* Grad-CAM highlights important regions influencing the decision

## 🌱 Future Improvements

* 🔁 Support for video deepfake detection
* 🔍 Include Vision Transformers (ViTs)
* 📈 Dashboard with performance metrics
* 📦 Docker containerization
* 👥 Feedback loop for user-reported accuracy


##  Acknowledgements

* [PyTorch](https://pytorch.org)
* [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
* [Kaggle Deepfake Dataset](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)
* Tailwind CSS

---

*This project was developed as part of an academic initiative on Deepfake Detection and Explainable AI.*
