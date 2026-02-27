import streamlit as st
from PIL import Image

st.set_page_config(page_title="deep.app", page_icon="🖼️", layout="centered")
st.title("deep.app - Pixel Resolution Checker")

# رفع الصورة
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # حساب معلومات البكسلات
    width, height = img.size
    total_pixels = width * height

    st.write(f"Width: {width} px")
    st.write(f"Height: {height} px")
    st.write(f"Total pixels: {total_pixels}")