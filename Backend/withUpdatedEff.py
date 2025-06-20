import os
import io
import base64
import numpy as np
from PIL import Image
import cv2 # Used by pytorch-grad-cam for image manipulation

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

from flask import Flask, request, jsonify
from flask_cors import CORS

# Grad-CAM specific imports
# Assuming 'grad_cam' package is installed (pip install grad-cam)
from pytorch_grad_cam import GradCAM 
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from pytorch_grad_cam.utils.image import show_cam_on_image

app = Flask(__name__)
CORS(app) # Enable CORS for cross-origin requests from your frontend

# --- Configuration ---
# Set the path to your trained models relative to app.py
MODEL_DIR = 'saved_models' # Ensure this folder exists and contains your .pth files
EFFICIENTNET_MODEL_NAME = 'final_efficientNet_model.pth' 
CNN_MODEL_NAME = 'cnn_model.pth' 

# Define device (CPU will be used if CUDA is not available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- CNN Model Definition (Copied directly from your models/cnn.py) ---
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # Input: 3xHxW
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output size (H/2, W/2)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output size (H/4, W/4)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # Last Conv2d layer
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # Output size (H/8, W/8) -> (1,1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), # Flattens the 128x1x1 output to 128
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Binary classification: Real vs Fake
        )

    def forward(self, x):
        x = self.network(x)
        x = self.classifier(x)
        return x

# --- Global Model Loading ---
# Load models once when the app starts to avoid reloading on every request
efficientnet_model = None
cnn_model = None

# EfficientNet Model Setup
try:
    # Instantiate EfficientNet B0, pretrained=False as you load state_dict
    efficientnet_model_base = models.efficientnet_b0(weights=None)
    num_classes_efficientnet = 2 
    # Modify the classifier head to match your training (if you replaced it)
    efficientnet_model_base.classifier[1] = nn.Linear(efficientnet_model_base.classifier[1].in_features, num_classes_efficientnet)

    efficientnet_model_path = os.path.join(MODEL_DIR, EFFICIENTNET_MODEL_NAME)
    efficientnet_model_base.load_state_dict(torch.load(efficientnet_model_path, map_location=device))
    efficientnet_model_base.to(device)
    efficientnet_model_base.eval() # Set to evaluation mode
    efficientnet_model = efficientnet_model_base # Assign to global variable
    print(f"EfficientNet model loaded successfully from {efficientnet_model_path}")
except FileNotFoundError:
    print(f"ERROR: EfficientNet model not found at {efficientnet_model_path}. Please check path.")
    efficientnet_model = None
except Exception as e:
    print(f"ERROR loading EfficientNet model: {e}")
    import traceback
    traceback.print_exc()
    efficientnet_model = None


# CNN Model Setup
try:
    cnn_model_base = CNNModel() # Instantiate your CNNModel
    cnn_model_path = os.path.join(MODEL_DIR, CNN_MODEL_NAME)
    cnn_model_base.load_state_dict(torch.load(cnn_model_path, map_location=device))
    cnn_model_base.to(device)
    cnn_model_base.eval() # Set to evaluation mode
    cnn_model = cnn_model_base # Assign to global variable
    print(f"CNN model loaded successfully from {cnn_model_path}")
except FileNotFoundError:
    print(f"ERROR: CNN model not found at {cnn_model_path}. Please check path.")
    cnn_model = None
except Exception as e:
    print(f"ERROR loading CNN model: {e}")
    import traceback
    traceback.print_exc()
    cnn_model = None


# --- Image Preprocessing for Models ---
# EfficientNet preprocessing (MATCHES YOUR TRAINING SCRIPT EXACTLY: 128x128 with normalization)
efficientnet_transform = transforms.Compose([
    transforms.Resize((128, 128)), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# CNN preprocessing (matching your test_efficientNet.py for CNN, assuming 128x128 and NO normalization)
cnn_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    # Assuming no normalization for CNN. If your CNN was trained with normalization, ADD it here.
])


# Class labels (based on your "Fake": 0, "Real": 1 logic)
LABELS = ['Fake', 'Real']


# --- Grad-CAM Helper Function ---
def generate_gradcam_heatmap(model, input_tensor, original_image_pil, model_name):
    """
    Generates a Grad-CAM heatmap and overlays it on the original image.

    Args:
        model (torch.nn.Module): The PyTorch model.
        input_tensor (torch.Tensor): Preprocessed image tensor for the model.
        original_image_pil (PIL.Image.Image): The original PIL Image (RGB, 0-255).
        model_name (str): The name of the model ('EfficientNet' or 'CNN') to determine target layer.

    Returns:
        str: Base64 encoded string of the overlaid image (data:image/png;base64,...), or None.
    """
    if model is None:
        print(f"Model ({model_name}) is None, cannot generate Grad-CAM.")
        return None

    target_layers = []
    # Determine target layer based on model name
    if model_name == 'EfficientNet':
        # For torchvision.models.efficientnet_b0, 'features.8' is the common last conv block.
        # This is where the highest-level features are extracted before the classifier.
        target_layer_name = 'features.8' 
    elif model_name == 'CNN':
        # For your CNNModel, the last Conv2d layer in the 'network' Sequential module is at index 6.
        target_layer_name = 'network.6' 
    else:
        print(f"Unknown model_name for Grad-CAM: {model_name}")
        return None

    # Find the specific target layer by name
    found_target_layer = False
    for name, module in model.named_modules():
        if name == target_layer_name:
            target_layers.append(module)
            found_target_layer = True
            break # Found the layer, no need to continue searching
    
    if not found_target_layer:
        print(f"Error: Named target layer '{target_layer_name}' not found for {model_name} Grad-CAM.")
        # Fallback to last Conv2d if named target not found (less reliable, but safer than crashing)
        temp_target_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                temp_target_layers.append(module)
        if temp_target_layers:
            target_layers = [temp_target_layers[-1]] # Take the very last Conv2d layer
            print(f"Warning: Fallback to last Conv2d '{temp_target_layers[-1]}' for {model_name}.")
        else:
            print(f"Critical: No suitable convolutional layer found for {model_name} Grad-CAM.")
            return None


    try:
        # Create GradCAM object (use_cuda argument removed as per fix)
        cam = GradCAM(model=model, target_layers=target_layers)
        
        # Generate the grayscale heatmap
        # targets=None uses the highest scoring class for CAM generation
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :] # Get the single image heatmap from the batch

    except Exception as e:
        print(f"Error during Grad-CAM generation for {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

    # The heatmap (grayscale_cam) is already resized by grad-cam to the model's input_tensor size (128x128).
    # So, resize the original PIL image to match this for correct overlay dimensions.
    target_size_for_cam_overlay = (128, 128) # Matches your model input and heatmap size
    resized_original_pil = original_image_pil.resize(target_size_for_cam_overlay)
    
    # Convert the resized PIL Image to NumPy array (RGB, 0-255)
    rgb_img_float = np.float32(resized_original_pil) / 255
    
    # Ensure grayscale_cam is not None before passing to show_cam_on_image
    if grayscale_cam is None:
        print(f"grayscale_cam is None for {model_name}, cannot overlay.")
        return None

    try:
        # Overlay heatmap on original image. show_cam_on_image expects float32 [0,1] RGB image.
        cam_image = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
    except Exception as e:
        print(f"Error during overlaying heatmap on image for {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Convert the resulting NumPy array image to PIL Image, then to base64
    cam_image_pil = Image.fromarray(cam_image)
    buffered = io.BytesIO()
    cam_image_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


# --- Flask Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected image file'}), 400

    try:
        # Read the image bytes and open with PIL
        img_bytes = file.read()
        pil_image = Image.open(io.BytesIO(img_bytes)).convert('RGB') 
        
        predictions_for_frontend = [] 
        
        # Store full prediction details (including models and tensors) for internal use (e.g., Grad-CAM)
        efficientnet_full_pred = None
        cnn_full_pred = None

        # --- EfficientNet Prediction ---
        if efficientnet_model:
            efficientnet_input_tensor = efficientnet_transform(pil_image).unsqueeze(0).to(device)
            #with torch.no_grad():
             #   efficientnet_outputs = efficientnet_model(efficientnet_input_tensor)
              #  efficientnet_probabilities = F.softmax(efficientnet_outputs, dim=1)[0] # Get probabilities for the single image
               # efficientnet_confidence, efficientnet_predicted_idx = torch.max(efficientnet_probabilities, 0)
            with torch.no_grad():
             efficientnet_outputs = efficientnet_model(efficientnet_input_tensor)
            efficientnet_probabilities = F.softmax(efficientnet_outputs, dim=1)[0]
            efficientnet_confidence, efficientnet_predicted_idx = torch.max(efficientnet_probabilities, 0)

           # Debug: Print raw softmax probabilities
            print(f"[DEBUG] EfficientNet Softmax Probabilities: {efficientnet_probabilities.tolist()}")
            print(f"[DEBUG] EfficientNet Prediction: {LABELS[efficientnet_predicted_idx.item()]} (Confidence: {efficientnet_confidence.item():.4f})")


            efficientnet_label = LABELS[efficientnet_predicted_idx.item()]
            efficientnet_confidence_score = round(efficientnet_confidence.item(), 4)

            # Store full prediction for internal use (Grad-CAM)
            efficientnet_full_pred = {
                'model': efficientnet_model, 
                'input_tensor': efficientnet_input_tensor, 
                'confidence': efficientnet_confidence_score,
                'label': efficientnet_label
            }

            # Add simplified data for frontend
            predictions_for_frontend.append({
                'model_name': 'EfficientNet',
                'confidence': efficientnet_confidence_score,
                'label': efficientnet_label
            })
            
        # --- CNN Prediction ---
        if cnn_model:
            cnn_input_tensor = cnn_transform(pil_image).unsqueeze(0).to(device)
            with torch.no_grad():
                cnn_outputs = cnn_model(cnn_input_tensor)
                cnn_probabilities = F.softmax(cnn_outputs, dim=1)[0] # Get probabilities for the single image
                cnn_confidence, cnn_predicted_idx = torch.max(cnn_probabilities, 0)

            cnn_label = LABELS[cnn_predicted_idx.item()]
            cnn_confidence_score = round(cnn_confidence.item(), 4)

            # Store full prediction for internal use (Grad-CAM)
            cnn_full_pred = {
                'model': cnn_model, 
                'input_tensor': cnn_input_tensor, 
                'confidence': cnn_confidence_score,
                'label': cnn_label
            }

            # Add simplified data for frontend
            predictions_for_frontend.append({
                'model_name': 'CNN',
                'confidence': cnn_confidence_score,
                'label': cnn_label
            })

        # --- Error Handling if no models loaded ---
        if not predictions_for_frontend: 
            return jsonify({'error': 'No models were loaded or able to make predictions.'}), 500
        
        # --- Dynamic Grad-CAM Selection ---
        selected_grad_cam_b64 = None
        selected_grad_cam_model_name = ""

        # Determine which model's Grad-CAM to generate based on higher confidence
        if efficientnet_full_pred and cnn_full_pred:
            effnet_chosen_conf = efficientnet_full_pred['confidence']
            cnn_chosen_conf = cnn_full_pred['confidence']
            
            if effnet_chosen_conf >= cnn_chosen_conf:
                selected_grad_cam_b64 = generate_gradcam_heatmap(
                    efficientnet_full_pred['model'], 
                    efficientnet_full_pred['input_tensor'], 
                    pil_image,
                    'EfficientNet'
                )
                selected_grad_cam_model_name = 'EfficientNet'
            else:
                selected_grad_cam_b64 = generate_gradcam_heatmap(
                    cnn_full_pred['model'], 
                    cnn_full_pred['input_tensor'], 
                    pil_image,
                    'CNN'
                )
                selected_grad_cam_model_name = 'CNN'
        elif efficientnet_full_pred: # Only EfficientNet available
            selected_grad_cam_b64 = generate_gradcam_heatmap(
                efficientnet_full_pred['model'],
                efficientnet_full_pred['input_tensor'],
                pil_image,
                'EfficientNet'
            )
            selected_grad_cam_model_name = 'EfficientNet'
        elif cnn_full_pred: # Only CNN available
            selected_grad_cam_b64 = generate_gradcam_heatmap(
                cnn_full_pred['model'],
                cnn_full_pred['input_tensor'],
                pil_image,
                'CNN'
            )
            selected_grad_cam_model_name = 'CNN'

        # --- Construct and Return Response ---
        response_data = {
            'predictions': predictions_for_frontend, 
            'dynamic_grad_cam': selected_grad_cam_b64,
            'grad_cam_model_name': selected_grad_cam_model_name
        }
        return jsonify(response_data)

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# --- Run the Flask app ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')