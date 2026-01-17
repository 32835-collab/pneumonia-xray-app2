import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# โหลดโมเดล EfficientNet
model = tf.keras.models.load_model(
    "efficientnet_pneumonia_infer.keras",
    compile=False
)


st.title("Pneumonia X-ray Diagnosis App")
st.write("Upload X-ray image to diagnose pneumonia")

uploaded_file = st.file_uploader(
    "Choose an X-ray image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.error(f"Pneumonia detected ({prediction*100:.2f}%)")
    else:
        st.success(f"Normal ({(1-prediction)*100:.2f}%)")
