import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# --- Constants ---
MODEL_PATH = "C:/Users/Jeba Jini/Documents/Project 6 Emotion/Best_model_final.keras"  # Or use .keras
IMG_SIZE = (96, 96)  # Match training image size
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# --- Load Model ---
@st.cache_resource
def load_emotion_model():
    return load_model(MODEL_PATH)

model = load_emotion_model()

# --- Face Detection ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_and_crop_face(image_pil):
    img = np.array(image_pil)
    img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    if len(faces) == 0:
        return image_pil  # fallback if no face found

    # Use the largest face detected (in case of multiple)
    largest_face = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)[0]
    x, y, w, h = largest_face
    cropped_face = img[y:y+h, x:x+w]
    return Image.fromarray(cropped_face)

# --- Prediction ---
def predict_emotion(image_pil):
    image_resized = image_pil.resize(IMG_SIZE)
    image_array = img_to_array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # shape: (1, h, w, 3)
    prediction = model.predict(image_array)
    return prediction

# --- Streamlit UI ---
st.title("😊 Emotion Detection App")
st.write("Upload a face image and detect the emotion!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Detect and crop face
    face_image = detect_and_crop_face(image)
    st.image(face_image, caption="Detected Face", use_column_width=False)

    # Predict
    st.write("⏳ Predicting...")
    predictions = predict_emotion(face_image)

    # Top result
    top_3_indices = predictions[0].argsort()[-3:][::-1]
    top_prediction = top_3_indices[0]
    st.subheader(f"🎯 Prediction: {CLASS_NAMES[top_prediction].capitalize()}")

    # Confidence scores
    st.markdown("**Top 3 Confidence Scores:**")
    for idx in top_3_indices:
        st.write(f"{CLASS_NAMES[idx].capitalize()}: {predictions[0][idx]:.4f}")

    # All scores (expandable)
    with st.expander("📊 Show all class scores"):
        for i, score in enumerate(predictions[0]):
            st.write(f"{CLASS_NAMES[i]}: {score:.4f}")
