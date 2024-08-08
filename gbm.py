import os
import pyedflib
import numpy as np
from scipy import signal
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Define your directory and the subject files you want to process
directory = '/Users/similovesyou/eeg-hackathon/data'

subjects = [f'S{i:03}' for i in range(1, 110)]  # Includes S109

# Initialize lists to hold features and labels
features = []
labels = []

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

# Convert to numpy arrays
features = np.array(features)  # Convert features to numpy array
labels = np.array(labels)     # Convert labels to numpy array

# Initialize K-Fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Lists to hold predictions and true labels
all_predictions = []
all_labels = []

for train_index, test_index in kf.split(features):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    
    # Train a Gradient Boosting classifier
    clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = clf.predict(X_test)
    
    # Collect all predictions and true labels
    all_predictions.extend(y_pred)
    all_labels.extend(y_test)

# Convert lists to numpy arrays
all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)

# Compute the confusion matrix for the entire dataset
cm = confusion_matrix(all_labels, all_predictions)

# Plot Confusion Matrix as Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Eyes Open', 'Eyes Closed'], yticklabels=['Eyes Open', 'Eyes Closed'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Gradient Boosting Classifier\n(Overall Performance Across K-Folds)')
plt.show()

# Print average accuracy
average_accuracy = accuracy_score(all_labels, all_predictions)
print(f"Average Accuracy: {average_accuracy}")
print("Classification Report:\n", classification_report(all_labels, all_predictions))