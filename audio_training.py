import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Example emotions from RAVDESS: calm, happy, sad, angry, fearful, disgust, surprised, neutral
EMOTIONS = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad', '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

features = []
labels = []

for file in os.listdir('RAVDESS_DIR'):
    if file.endswith('.wav'):
        path = os.path.join('RAVDESS_DIR', file)
        emotion = EMOTIONS[file.split('-')[2]]
        features.append(extract_features(path))
        labels.append(list(EMOTIONS.values()).index(emotion))

X = np.array(features)
y = to_categorical(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential()
model.add(Dense(256, input_shape=(13,), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(len(EMOTIONS), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

model.save('audio_emotion_model.h5')
