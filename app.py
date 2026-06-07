import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("models/gender_cnn.keras")

st.title("Gender Detection using CNN")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    st.image(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        caption="Uploaded Image"
    )

    face = cv2.resize(img, (64,64))

    face = face / 255.0

    face = np.expand_dims(face, axis=0)

    prediction = model.predict(
        face,
        verbose=0
    )

    prob = float(prediction[0][0])

    st.write("Raw Probability:", prob)

    gender = "Female" if prob > 0.5 else "Male"
    confidence = max(prob, 1-prob)

    st.success(
        f"Prediction: {gender}"
    )

    st.write(
        f"Confidence: {confidence:.2%}"
    )