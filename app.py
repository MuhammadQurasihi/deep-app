# ==============================
# app.py - Deepfake Detector
# ==============================
import streamlit as st
import torch
import torch.nn as nn
import clip
from PIL import Image
import torchvision.transforms as transforms

# ---------- Device setup ----------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Load CLIP ----------
clip_model, preprocess = clip.load("ViT-L/14", device=device)
clip_model.eval()
for param in clip_model.parameters():
    param.requires_grad = False

# ---------- Build Classifier ----------
class CLIPClassifier(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.fc = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.clip_model.encode_image(x).float()
        logits = self.fc(features)
        return logits

# ---------- Load weights ----------
model = CLIPClassifier(clip_model).to(device)
state_dict = torch.load("clip_classifier.pth", map_location=device)
model.fc.load_state_dict(state_dict)
model.eval()

# ---------- Streamlit ----------
st.title("🖼️ Deep Fake Detector")
st.write("Upload an image to check if it is Real or Fake")

uploaded_file = st.file_uploader("Choose an image", type=["jpg","jpeg","png","bmp","webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ---------- Preprocess ----------
    preprocess_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                            (0.26862954, 0.26130258, 0.27577711))
    ])
    img_tensor = preprocess_transform(image).unsqueeze(0).to(device)

    # ---------- Prediction ----------
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)
        label = "Fake" if pred_class.item() == 1 else "Real"
        confidence = confidence.item() * 100

    st.markdown(f"### Result: **{label}**")
    st.markdown(f"### Confidence: **{confidence:.2f}%**")