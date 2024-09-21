import numpy as np
import cv2
from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image

# Load the trained model
model = load_model('final_model.h5')

# Function to prepare the image
def prepare_image(img, img_size=(64, 64)):
    # Convert PIL Image to OpenCV format
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # Resize the image to the required size
    img = cv2.resize(img, img_size)
    # Expand dimensions to fit model input shape (1, 64, 64, 3)
    img = np.expand_dims(img, axis=0)
    return img

# Streamlit App
st.title("Cat or Dog Classifier")

# Sidebar for image upload
st.sidebar.title("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Button to predict
if st.sidebar.button("Predict"):
    if uploaded_file is not None:
        # Prepare the image for prediction
        image = Image.open(uploaded_file)
        processed_image = prepare_image(image)

        # Make predictions
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]  # Get predicted class index
        class_labels = { 1: 'Dog',0: 'Cat'}  # Class labels
        predicted_label = class_labels[predicted_class]

        # Display the uploaded image and prediction
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write(f"Predicted Category: {predicted_label}")
    else:
        st.write("Please upload an image to make a prediction.")
