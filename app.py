import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image

# App title
st.header("Image Classification Model")

# Load model
model = load_model("final_Image_classify.h5")

# Class labels
data_cat = ['Blocked_Drainage',
 'Garbage_Dumping',
 'Hanging_Electrical_Cables',
 'Overflowing_Dustbin',
 'Potholes',
 'Streetlight_Not_Working',
 'Tree_Fallen',
 'Water_Leakage']

# Image dimensions
img_height = 180
img_width = 180

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    st.image(uploaded_file, width=200, caption="Uploaded Image")

    # Preprocess image
    image_load = Image.open(uploaded_file).convert("RGB")
    image_resized = image_load.resize((img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image_resized)
    img_bat = tf.expand_dims(img_arr, 0)  # create batch

    # Prediction
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict[0])

    # Show results
    st.write(f"**File name:** {uploaded_file.name}")
    st.write(f"**Predicted class:** {data_cat[np.argmax(score)]}")
    st.write(f"**Accuracy:** {np.max(score) * 100:.2f}%")
