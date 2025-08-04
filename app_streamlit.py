import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import time
import os
import base64
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------
# Load models and labels
# ------------------------
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_model = load_model("complete_model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ------------------------
# Audio files for emotions
# ------------------------
AUDIO_PATHS = {
    "Sad": "audio/sad.mp3",
    "Angry": "audio/angry.mp3"
}

# ------------------------
# Setup session state
# ------------------------
if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = []
if "checkbox_state" not in st.session_state:
    st.session_state.checkbox_state = False

# ------------------------
# Play MP3 from file
# ------------------------
def play_audio(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
            b64 = base64.b64encode(audio_bytes).decode()
            audio_html = f"""
            <audio autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)

# ------------------------
# Detect emotion in a frame
# ------------------------
def detect_emotion_in_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    emotions = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi = cv2.resize(roi_gray, (48, 48))
        roi = roi.astype('float') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        preds = emotion_model.predict(roi)[0]
        label = emotion_labels[np.argmax(preds)]
        emotions.append(label)

        # Update UI
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Save history
        st.session_state.emotion_history.append(label)

        # Play music if emotion is sad or angry
        if label in AUDIO_PATHS:
            play_audio(AUDIO_PATHS[label])

    return frame, emotions


# ------------------------
# UI - Streamlit Layout
# ------------------------

st.set_page_config(page_title="Emotion Detection App", layout="wide")
st.title("🎭 Mental Health Sentimental Analysis")

# 🧭 Vertical sidebar navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["📷 Image Upload", "💬 Text Detection", "🎥 Webcam Live", "📊 Emotion History"])

# 📷 Image Upload
if page == "📷 Image Upload":
    st.header("📷 Upload an Image")
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        result_img, emotions = detect_emotion_in_frame(image)
        st.image(result_img, channels="BGR", caption="Detected Emotions")

        if emotions:
            st.success(f"Detected Emotions: {', '.join(emotions)}")
        else:
            st.warning("No face detected.")

# 💬 Text Emotion Detection
elif page == "💬 Text Detection":
    st.header("💬 Text Emotion Detection")
    text_input = st.text_area("Enter your thoughts...")

    if st.button("Detect Text Emotion"):
        if text_input.strip():
            text = text_input.lower()
            emotion = "Neutral 😐"
            if any(w in text for w in ["sad", "depressed", "unhappy", "cry"]):
                emotion = "Sad 😢"
                play_audio(AUDIO_PATHS.get("Sad", ""))
                st.session_state.emotion_history.append("Sad")
            elif any(w in text for w in ["angry", "mad", "furious"]):
                emotion = "Angry 😡"
                play_audio(AUDIO_PATHS.get("Angry", ""))
                st.session_state.emotion_history.append("Angry")
            elif any(w in text for w in ["happy", "joy", "excited", "glad"]):
                emotion = "Happy 😊"
                st.session_state.emotion_history.append("Happy")
            elif any(w in text for w in ["fear", "scared", "afraid"]):
                emotion = "Fear 😱"
                st.session_state.emotion_history.append("Fear")
            else:
                st.session_state.emotion_history.append("Neutral")
            st.info(f"Predicted Emotion: **{emotion}**")
        else:
            st.warning("Please enter some text.")

# 🎥 Webcam Live Detection
elif page == "🎥 Webcam Live":
    st.header("🎥 Live Webcam Detection")
    run_webcam = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])

    if run_webcam:
        st.session_state.checkbox_state = True
        cap = cv2.VideoCapture(0)
        while run_webcam:
            success, frame = cap.read()
            if not success:
                st.error("Failed to access webcam.")
                break
            result_frame, emotions = detect_emotion_in_frame(frame)
            result_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(result_frame)

            if not st.session_state.get("checkbox_state", True):
                break
        cap.release()
    else:
        st.session_state.checkbox_state = False

# 📊 Emotion History & Charts
elif page == "📊 Emotion History":
    st.header("📊 Emotion History Chart")

    history = st.session_state.emotion_history

    if history:
        df = pd.DataFrame(history, columns=["Emotion"])
        st.subheader("Bar Chart")
        st.bar_chart(df["Emotion"].value_counts())

        st.subheader("Pie Chart")
        fig, ax = plt.subplots()
        df["Emotion"].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, shadow=True)
        st.pyplot(fig)

        st.subheader("Clear History")
        if st.button("Clear"):
            st.session_state.emotion_history = []
            st.success("History cleared.")
    else:
        st.info("No emotion history recorded yet.")
