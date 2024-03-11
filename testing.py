import numpy as np
import librosa
from joblib import load

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    # Extract Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    # Extract Spectral Contrast features
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    # Calculate the mean of each feature set to form the final feature vector
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    chroma_scaled = np.mean(chroma.T, axis=0)
    contrast_scaled = np.mean(contrast.T, axis=0)
    # Concatenate all features to create a single feature vector
    features = np.hstack([mfcc_scaled, chroma_scaled, contrast_scaled])
    return features.reshape(1, -1)  # Reshape to the format of 1 sample

def predict_genre_proba(file_path, model_path):
    # Load the model
    model = load(model_path)
    
    # Extract features from the music file
    features = extract_features(file_path)
    
    # Use the model to predict
    probabilities = model.predict_proba(features)
    
    return probabilities  # Return prediction probabilities

# Path to the model file
model_path = 'music_genre_classifier.joblib'

# Path to the music file to be classified
music_file_path = 'd:/Techin513/Output/hiphopcombine.wav'

# Predict the probabilities for music genres
predicted_genre_probabilities = predict_genre_proba(music_file_path, model_path)

# Assuming we know the order of labels, e.g.: [Boombap, Drill, Jazzhiphop, Trap]
genre_labels = ['Boombap', 'Drill', 'Jazzhiphop', 'Trap']  # These should be obtained from the LabelEncoder used during training

# Print the predicted probabilities for each genre
print("Predicted probabilities:")
for label, probability in zip(genre_labels, predicted_genre_probabilities[0]):
    print(f"{label}: {probability:.2f}")