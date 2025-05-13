# Real-Time Interview Feedback System Based on Non-Verbal Cues

import cv2
import numpy as np
import librosa
import sounddevice as sd
import queue
import threading
import time
from keras.models import load_model
from fer import FER

# Load pretrained emotion detector for facial expressions (from AffectNet or FER+)
face_emotion_detector = FER(mtcnn=True)

# Load pretrained audio emotion model (trained on RAVDESS)
audio_model = load_model('audio_emotion_model.h5')  # You must train this separately or get from a source

# Labels for emotions (example for 7-class model)
emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust']

audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy())

def extract_audio_features(audio_chunk, sr=22050):
    mfccs = librosa.feature.mfcc(y=audio_chunk, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0).reshape(1, -1)

def analyze_audio():
    duration = 2  # seconds
    fs = 22050
    while True:
        audio_data = []
        for _ in range(int(duration * fs / 1024)):
            chunk = audio_queue.get()
            audio_data.append(chunk[:, 0])
        audio_signal = np.concatenate(audio_data)
        features = extract_audio_features(audio_signal, sr=fs)
        emotion_pred = audio_model.predict(features)
        audio_emotion = emotion_labels[np.argmax(emotion_pred)]
        print("[AUDIO] Detected emotion:", audio_emotion)
        provide_feedback(audio_emotion, source="audio")

def provide_feedback(emotion, source="video"):
    if source == "video":
        if emotion in ['sad', 'angry', 'fearful']:
            print("[FEEDBACK] Your facial expression seems tense or negative. Try smiling more.")
        else:
            print("[FEEDBACK] Good facial expression. Keep it up!")
    else:
        if emotion in ['sad', 'angry', 'fearful']:
            print("[FEEDBACK] Your voice tone seems tense or anxious. Try speaking slower and with a calm tone.")
        else:
            print("[FEEDBACK] Good voice tone. You sound confident.")

def analyze_video():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        emotion_data = face_emotion_detector.detect_emotions(frame)
        if emotion_data:
            top_emotion = face_emotion_detector.top_emotion(frame)
            if top_emotion:
                print("[VIDEO] Detected emotion:", top_emotion[0])
                provide_feedback(top_emotion[0], source="video")
        cv2.imshow('Webcam Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("Starting Real-Time Interview Feedback System...")

    # Start audio stream
    audio_thread = threading.Thread(target=analyze_audio)
    audio_thread.daemon = True
    audio_thread.start()
    
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=22050, blocksize=1024):
        analyze_video()
