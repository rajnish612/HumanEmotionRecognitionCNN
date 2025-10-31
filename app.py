import io
from typing import Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image


@st.cache_resource
def load_model_and_detector(model_path: str = "emotion_recognition_CNN.h5"):
    from tensorflow.keras.models import load_model

    model = load_model(model_path)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    return model, face_cascade


def preprocess_image(
    img: Image.Image, face_cascade: cv2.CascadeClassifier
) -> Tuple[np.ndarray, Image.Image]:

    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) > 0:
        # take the largest face
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h = faces[0]
        roi_gray = gray[y : y + h, x : x + w]
        annotated = img_cv.copy()
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        h_img, w_img = gray.shape
        side = min(h_img, w_img)
        x = (w_img - side) // 2
        y = (h_img - side) // 2
        roi_gray = gray[y : y + side, x : x + side]
        annotated = img_cv.copy()

    roi_resized = cv2.resize(roi_gray, (48, 48))
    roi_norm = roi_resized.astype("float32") / 255.0
    roi_norm = np.expand_dims(roi_norm, axis=(0, -1))

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    annotated_pil = Image.fromarray(annotated_rgb)
    return roi_norm, annotated_pil


def predict_emotion(model, roi: np.ndarray, labels):
    preds = model.predict(roi, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    top_label = labels[top_idx]
    top_conf = float(preds[top_idx])
    return top_label, top_conf, preds


def main():
    st.set_page_config(page_title="Emotion Webcam (Streamlit)", layout="wide")
    st.title("Live Emotion Detection (Camera)")

    # Project / author info
    st.title("**Author:** Rajnish Nath — BCA Undergraduate")
    st.title("**Project:** CNN Computer Vision Project — Human Emotion Recognition")
    st.markdown(
        "This demo uses your webcam (browser camera) to capture an image, detects the face, and predicts the emotion using a trained Keras model."
    )

    model, face_cascade = load_model_and_detector()

    labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Camera")
        img_file = st.camera_input("Take a photo")

        if img_file is not None:
            img = Image.open(img_file).convert("RGB")
            roi, annotated = preprocess_image(img, face_cascade)

            st.image(annotated, caption="Captured (annotated)", use_column_width=True)

            label, conf, preds = predict_emotion(model, roi, labels)

            st.success(f"Prediction: {label} ({conf*100:.1f}%)")

    with col2:
        st.header("Upload Image")
        uploaded = st.file_uploader("Or upload an image", type=["png", "jpg", "jpeg"])
        if uploaded is not None:
            img2 = Image.open(uploaded).convert("RGB")
            roi2, annotated2 = preprocess_image(img2, face_cascade)
            st.image(annotated2, caption="Uploaded (annotated)", use_column_width=True)
            label2, conf2, preds2 = predict_emotion(model, roi2, labels)
            st.success(f"Prediction: {label2} ({conf2*100:.1f}%)")


if __name__ == "__main__":
    main()
