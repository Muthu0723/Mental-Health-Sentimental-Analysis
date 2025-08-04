# Mental-Health-Sentimental-Analysis
A real-time emotion recognition system built using **Streamlit**, **OpenCV**, and a **CNN model**, capable of detecting human emotions from **facial expressions (webcam/image)** and **text input**. It also visualizes emotion history using charts and plays calming audio when negative emotions are detected.

---

## ✨ Features

- 🎥 **Live Webcam Emotion Detection**
- 📷 **Image Upload-Based Face Emotion Recognition**
- 💬 **Text-Based Sentiment Analysis**
- 📊 **Emotion History Tracking and Visualization (Bar + Pie Chart)**
- 🔊 **Automatic Calming Music Playback for Sad/Angry Emotions**

---

## 🧠 Technologies Used

- Python
- Streamlit
- OpenCV
- TensorFlow / Keras (CNN model)
- Pandas, Matplotlib
- HTML5 (for audio embedding)

---
## 📁 Project Structure

emotion_app/
├── app_streamlit.py # Main Streamlit App
├── complete_model.h5 # Pretrained CNN model for face emotion detection
├── audio/
│ ├── sad.mp3 # Calming music for "sad" emotion
│ └── angry.mp3 # Calming music for "angry" emotion

---
## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/emotion-detection-app.git
cd emotion-detection-app
2. Install Dependencies
Make sure Python 3.8+ is installed, then run:

bash
Copy
Edit
pip install -r requirements.txt
If you don't have a requirements.txt, create one with:

txt
Copy
Edit
streamlit
opencv-python-headless
tensorflow
keras
pandas
matplotlib
Pillow
3. Run the App
bash
Copy
Edit
streamlit run app_streamlit.py
📷 Demo Preview

Webcam emotion detection with real-time labels and music response.

🔮 Future Improvements
Integrate BERT/RoBERTa for advanced text sentiment

Export emotion logs as CSV

Save webcam snapshots on negative emotions

Email alerts for frequent sadness or anger

🤝 Contribution
Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

📜 License
MIT

🙋‍♂️ Author
Muthu Nivesh
Feel free to connect on LinkedIn or explore my other projects!
