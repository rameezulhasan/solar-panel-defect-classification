import json
import streamlit as st
import requests
from PIL import Image
import os
# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="Solar Panel Defect Classification",
    page_icon="☀️",
    layout="centered"
)

# =====================================================
# FastAPI backend ka URL
# =====================================================
# Local testing ke liye ye rakho, Docker mein deploy karte waqt
# ye adjust karna hoga (jaise service name ya deployed URL)
API_URL = os.getenv("API_URL","http://localhost:8000/predict")

# =====================================================
# Sidebar
# =====================================================

with st.sidebar:

    st.title("Model Information")

    st.write("### Model")
    st.write("EfficientNetB0 (Fine-Tuned)")

    st.write("### Input Size")
    st.write("224 × 224")

    st.write("### Backend")
    st.write("FastAPI")

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
# Main UI
# =====================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        width='stretch'
    )

    st.write("")

    if st.button("Predict Defect", width='stretch'):

        with st.spinner("Analyzing Image..."):

            # Uploaded file ko FastAPI ko bhejna hai
            # File pointer ko wapas start pe le jao (Streamlit ka uploaded_file ek stream hai)
            uploaded_file.seek(0)

            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}

            try:
                response = requests.post(API_URL, files=files)

                if response.status_code == 200:
                    result = response.json()

                    predicted_class = result["predicted_class"]
                    confidence = result["confidence"]
                    probabilities = result["all_probabilities"]

                    st.success("Prediction Completed Successfully")

                    st.markdown("---")

                    st.subheader("Prediction Result")

                    st.metric("Predicted Class", predicted_class)
                    st.metric("Confidence", f"{confidence*100:.2f}%")

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

                    # Probabilities ko sort karo highest se lowest
                    sorted_probs = sorted(
                        probabilities.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )

                    for class_name, probability in sorted_probs:
                        st.write(f"**{class_name}**")
                        st.progress(float(probability))
                        st.write(f"{probability*100:.2f}%")

                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the backend API. Make sure FastAPI server is running.")

# =====================================================
# Footer
# =====================================================

st.markdown("---")

st.caption(
    "Developed using TensorFlow, EfficientNetB0, FastAPI, and Streamlit."
)