import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained model
import os
model_path = os.path.join(os.path.dirname(__file__), 'tree_species_classifier.keras')
model = tf.keras.models.load_model(model_path)

# Get class labels from the training generator if you saved them
# For now, use dummy class names
class_names = [
    'amla', 'asopalav', 'babul', 'bamboo', 'banyan', 'bili', 'cactus', 'champa',
    'coconut', 'garmalo', 'gulmohor', 'gunda', 'jamun', 'kanchan', 'kesudo',
    'khajur', 'mango', 'motichanoti', 'neem', 'nilgiri', 'other', 'pilikaren',
    'pipal', 'saptaparni', 'shirish', 'simlo', 'sitafal', 'sonmahor',
    'sugarcane', 'vad'
]

# Set up the Streamlit app UI
st.set_page_config(page_title="Tree Species Classifier", layout="centered")

st.title("🌳 Tree Species Classifier")
st.write("Upload an image of a tree leaf or bark, and I'll try to predict the species.")

# Upload image
uploaded_file = st.file_uploader("Upload a tree image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = 100 * np.max(prediction)

    st.success(f"🌿 Predicted Species: **{predicted_class}**")
    st.write(f"🧠 Confidence: **{confidence:.2f}%**")
