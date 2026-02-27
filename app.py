# app.py
import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
import numpy as np
from torchvision import datasets

# Load dataset structure to get class_to_idx
train_dataset = datasets.ImageFolder(root="../dataset/train")  # ضع مسار مجلد التدريب الصحيح

# Build class_names list according to class_to_idx
class_names = [None] * len(train_dataset.class_to_idx)
for key, value in train_dataset.class_to_idx.items():
    class_names[value] = key.capitalize()  # ['Fake', 'Real']

# --- Setup device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model ---
model = models.efficientnet_b0(weights=None)  # weights=None since we use our trained model
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)  # 2 classes: Fake, Real
model.load_state_dict(torch.load("final_model.pth", map_location=device))
model = model.to(device)
model.eval()

# --- Define image transforms ---
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# --- Define class names according to class_to_idx ---
# This is critical to avoid "reverse class" issue
class_names = ['Real', 'Fake']  # 0 = Fake, 1 = Real

# --- Streamlit interface ---
st.title("Deepfake Detection App")
st.write("Upload an image to classify it as Real or Fake with confidence percentage.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Model prediction
    with torch.no_grad():
      outputs = model(input_tensor)
      probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
      pred_class = np.argmax(probs)
      class_name = class_names[pred_class]  # دائمًا الصنف الصحيح
      confidence = probs[pred_class] * 100

    st.write(f"Prediction: **{class_name}**")
    st.write(f"Confidence: **{confidence:.2f}%**")