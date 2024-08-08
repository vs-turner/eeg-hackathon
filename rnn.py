import os
import requests
import pyedflib
import numpy as np
import sklearn
from scipy import signal
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.utils import to_categorical
import pandas as pd

# Define your directory and the subject files you want to process
directory = '/Users/arwaadib/Desktop/eeg-hackathon/data'

subjects = [f'S{i:03}' for i in range(1, 110)]  # Creates S001, S002, ..., S109

# Initialize lists to hold features and labels
features = []
labels = []

def download_file(url, local_path):
    response = requests.get(url)
    with open(local_path, 'wb') as f:
        f.write(response.content)

def extract_band_power(signal_data, sample_rate, band):
    f, Pxx = signal.welch(signal_data, fs=sample_rate, nperseg=sample_rate*2)
    band_mask = (f >= band[0]) & (f <= band[1])
    return np.mean(Pxx[band_mask])

# Define frequency bands
bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 12),
    'Beta': (13, 30),
    'Gamma': (30, 50)
}

for subject_id in subjects:
    print(f'\nProcessing subject: {subject_id}')
    
    # Define file names for the subject
    file_R01 = f'{subject_id}R01.edf'  # Eyes open
    file_R02 = f'{subject_id}R02.edf'  # Eyes closed
    
    file_paths = [os.path.join(directory, file_R01), os.path.join(directory, file_R02)]
    
    try:
        # Open both EDF files
        with pyedflib.EdfReader(file_paths[0]) as f1, pyedflib.EdfReader(file_paths[1]) as f2:
            signal_labels = f1.getSignalLabels()
            sample_rate = int(f1.getSampleFrequency(0))

            # Loop through all channels and extract features
            for channel in signal_labels:
                index_R01 = signal_labels.index(channel)
                index_R02 = signal_labels.index(channel)

                # Read signals
                signal_data_R01 = f1.readSignal(index_R01)
                signal_data_R02 = f2.readSignal(index_R02)

                # Extract features for each frequency band
                feature_R01 = [extract_band_power(signal_data_R01, sample_rate, band_range) for band_name, band_range in bands.items()]
                feature_R02 = [extract_band_power(signal_data_R02, sample_rate, band_range) for band_name, band_range in bands.items()]

                # Append features and labels
                features.append(feature_R01)
                labels.append(0)  # Label for eyes open

                features.append(feature_R02)
                labels.append(1)  # Label for eyes closed

    except Exception as e:
        print(f'Error reading files for {subject_id}: {e}')
        continue  # Skip to the next subject if there's an error

# Convert to numpy arrays
features = np.array(features)  # Convert features to numpy array
labels = np.array(labels)     # Convert labels to numpy array

# Reshape features for RNN input (samples, timesteps, features)
features = features.reshape((features.shape[0], 1, features.shape[1]))

# One-hot encode labels
labels = to_categorical(labels)

# Initialize K-Fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Lists to hold performance metrics
accuracies = []
all_y_test = []
all_y_pred = []

# Function to create a DataFrame from the classification report
def classification_report_to_df(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split()
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = int(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    return dataframe

for train_index, test_index in kf.split(features):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    
    # Build the RNN model
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(2, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Train the model
    print("Training the model...")
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    accuracies.append(accuracy)
    all_y_test.extend(y_test_classes)
    all_y_pred.extend(y_pred_classes)
    print(f"Fold Accuracy: {accuracy}")

# Print average accuracy
print(f"Average Accuracy: {np.mean(accuracies)}")

# Plot the overall confusion matrix
conf_matrix = confusion_matrix(all_y_test, all_y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Eyes Open', 'Eyes Closed'],
            yticklabels=['Eyes Open', 'Eyes Closed'])
plt.title('Confusion Matrix for RNN Classifier\n(Overall Performance Across K-Folds)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()