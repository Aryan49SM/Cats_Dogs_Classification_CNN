import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
import requests

# Load the model
model = tf.keras.models.load_model('cats_dog_classification_cnn_model.h5')

st.set_page_config(page_title="Dog vs Cat Classification", layout="wide")

st.title("Dog vs Cat Classification")
st.markdown("<h6> ⚠️ Please upload an image of a dog or cat. Using other images may lead to inaccurate results..</h6>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Upload or Enter an Image")
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])
image_url = st.sidebar.text_input("Or Enter an Image URL")

# Main section
if 'original_image' not in st.session_state:
    st.session_state['original_image'] = None
    st.session_state['prediction_text'] = None

try:
    # Process image
    if uploaded_file:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    elif image_url:
        try:
            response = requests.get(image_url)
            image = np.asarray(bytearray(response.content), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        except:
            st.sidebar.error("Error: Unable to fetch the image from the provided URL.")
            st.stop()
    else:
        st.sidebar.warning("Please upload an image or enter a URL!")
        st.stop()

    # Image validation
    if image is None:
        st.sidebar.error("Error: Invalid image format or URL. Please provide a valid image.")
        st.stop()

    # Convert image to RGB for correct display
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Image (resize and normalize)
    resized_image = cv2.resize(image, (256, 256))
    normalized_image = resized_image / 255.0
    image_input = np.expand_dims(normalized_image, axis=0)

    # Prediction
    prediction = model.predict(image_input)
    prediction_text = "This is an image of a dog." if prediction[0][0] > 0.5 else "This is an image of a cat."

    st.session_state['original_image'] = original_image
    st.session_state['prediction_text'] = prediction_text

except cv2.error as e:
    st.sidebar.error("Error: Invalid image. Please provide a valid image URL or file.")

# Display the image and prediction
if st.session_state['original_image'] is not None:
    col1, col2 = st.columns([2, 1], gap="medium")
    with col1:
        st.image(st.session_state['original_image'], caption="Uploaded Image", use_container_width=True)

    with col2:
        st.subheader("Prediction:")
        st.info(f"**{st.session_state['prediction_text']}**")
