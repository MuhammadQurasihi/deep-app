import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

# ---------- إعداد الصفحة ----------
st.title("Deepfake Detector")

# ---------- تحميل الموديل ----------
@st.cache_resource
def load_model():
    model = torch.load("final_model.pth", map_location=torch.device("cpu"))
    model.eval()
    return model

model = load_model()

# ---------- أسماء الكلاسات ----------
class_names = ["fake", "real"]  # مهم ترتيب نفس التدريب

# ---------- preprocessing ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------- رفع صورة ----------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    label = class_names[predicted.item()]

    st.subheader(f"Prediction: {label}")