import mne
import numpy as np
import matplotlib.pyplot as plt

# Load the .edf file
file_path = 'data/S001R01.edf'
raw = mne.io.read_raw_edf(file_path, preload=True)

# Print information about the data
print(raw.info)

# Extract raw EEG data and the sampling frequency
data, sfreq = raw[:]

# Select a channel to create a spectrogram for (e.g., the first channel)
channel_data = data[0, :]  # Assuming you want to analyze the first channel

# Adjust the sensitivity
NFFT = 128  # Number of data points used in each block for FFT
noverlap = NFFT // 2  # Overlap between blocks

# Create the spectrogram
plt.figure(figsize=(10, 6))
plt.specgram(channel_data, NFFT=NFFT, Fs=sfreq, noverlap=noverlap, cmap='plasma')
plt.title('Spectrogram of EEG Data (Channel 1)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Intensity (dB)')
plt.show()
