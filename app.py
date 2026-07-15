import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="Solar Panel Defect Classification",
    page_icon="☀️",
    layout="centered"
)

# =====================================================
# Load Model
# =====================================================

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "best_model_efficientnetb0_finetuned.keras"
    )
    return model


# =====================================================
# Load Class Names
# =====================================================

@st.cache_data
def load_class_names():

    with open("class_names.json", "r") as f:
        class_names = json.load(f)

    return class_names


model = load_model()
class_names = load_class_names()

# =====================================================
# Sidebar
# =====================================================

with st.sidebar:

    st.title("Model Information")

    st.write("### Model")
    st.write("EfficientNetB0 (Fine-Tuned)")

    st.write("### Input Size")
    st.write("224 × 224")

    st.write("### Total Classes")
    st.write(len(class_names))

    st.write("### Classes")

    for cls in class_names:
        st.write(f"• {cls}")

    st.markdown("---")

    st.info(
        "Upload a clear solar panel image for the best prediction."
    )

# =====================================================
# Title
# =====================================================

st.title("☀️ Solar Panel Defect Classification")

st.write(
    "Upload a solar panel image to identify the defect type using an EfficientNetB0 deep learning model."
)

# =====================================================
# Upload Image
# =====================================================

uploaded_file = st.file_uploader(

    "Choose an Image",

    type=["jpg", "jpeg", "png"]

)

# =====================================================
# Image Preprocessing
# =====================================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize((224,224))

    image = np.array(image)

    image = np.expand_dims(image, axis=0)

    return image

# =====================================================
# Prediction
# =====================================================

def predict_image(image):

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image, verbose=0)

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    confidence = float(prediction[0][predicted_index])

    return predicted_class, confidence, prediction[0]

# =====================================================
# Main UI
# =====================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    st.write("")

    if st.button("Predict Defect", use_container_width=True):

        with st.spinner("Analyzing Image..."):

            predicted_class, confidence, probabilities = predict_image(image)

        st.success("Prediction Completed Successfully")

        st.markdown("---")

        st.subheader("Prediction Result")

        st.metric(

            "Predicted Class",

            predicted_class

        )

        st.metric(

            "Confidence",

            f"{confidence*100:.2f}%"

        )

        if confidence >= 0.90:

            st.success("Very High Confidence Prediction")

        elif confidence >= 0.70:

            st.info("High Confidence Prediction")

        elif confidence >= 0.50:

            st.warning("Moderate Confidence Prediction")

        else:

            st.error("Low Confidence Prediction")

        st.markdown("---")

        st.subheader("Class Probabilities")

        results = list(zip(class_names, probabilities))

        results = sorted(

            results,

            key=lambda x: x[1],

            reverse=True

        )

        for class_name, probability in results:

            st.write(

                f"**{class_name}**"

            )

            st.progress(float(probability))

            st.write(

                f"{probability*100:.2f}%"

            )

# =====================================================
# Footer
# =====================================================

st.markdown("---")

st.caption(
    "Developed using TensorFlow, EfficientNetB0 and Streamlit."
)