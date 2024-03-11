import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump

def extract_features(file_path, augment=False):
    y, sr = librosa.load(file_path, duration=30)
    if augment:
        y = np.roll(y, int(np.random.rand() * sr))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    chroma_scaled = np.mean(chroma.T, axis=0)
    contrast_scaled = np.mean(contrast.T, axis=0)
    return np.hstack([mfcc_scaled, chroma_scaled, contrast_scaled])

def get_features_and_labels(folder_path):
    features = []
    labels = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if not file.endswith('.wav'):
            continue
        class_label = file.split('_')[0]
        # Extract original features
        data = extract_features(file_path)
        features.append(data)
        labels.append(class_label)
        # Data augmentation by time shifting
        augmented_data = extract_features(file_path, augment=True)
        features.append(augmented_data)
        labels.append(class_label)
    return np.array(features), np.array(labels)

def prepare_data(folder_path):
    features, labels = get_features_and_labels(folder_path)
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    return train_test_split(features, encoded_labels, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model

def save_model(model, filename):
    dump(model, filename)

# 调整为你的音乐文件夹路径
folder_path = 'c:\\Users\\Jase\\Downloads\\music'
X_train, X_test, y_train, y_test = prepare_data(folder_path)
model = train_model(X_train, y_train)

# 在测试集上评估模型性能
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

# 保存模型
model_filename = 'music_genre_classifier.joblib'
save_model(model, model_filename)

print(f'Model saved to {model_filename}')
