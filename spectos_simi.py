import os
import pyedflib
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# Define your directory and the subject files you want to process
directory = '/Users/similovesyou/eeg-hackathon/data'

subjects = [f'S{i:03}' for i in range(1, 11)]  # Creates S001, S002, ..., S010

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
            print(f'Signal labels: {signal_labels}')
            print(f'Number of signals: {f1.signals_in_file}')
            sample_rate = f1.getSampleFrequency(0)
            print(f'Sampling frequency: {sample_rate} Hz')
            sample_rate = int(sample_rate)

            # Loop through all channels and generate spectrograms for both R01 and R02
            for channel in signal_labels:
                index_R01 = signal_labels.index(channel)
                index_R02 = signal_labels.index(channel)

                # Read signals
                signal_data_R01 = f1.readSignal(index_R01)
                signal_data_R02 = f2.readSignal(index_R02)

                # Generate spectrograms
                f_R01, t_R01, Sxx_R01 = signal.spectrogram(signal_data_R01, fs=sample_rate)
                f_R02, t_R02, Sxx_R02 = signal.spectrogram(signal_data_R02, fs=sample_rate)

                # Plotting
                plt.figure(figsize=(18, 8))

                # R01 (Eyes Open)
                plt.subplot(1, 2, 1)
                plt.pcolormesh(t_R01, f_R01, 10 * np.log10(Sxx_R01), shading='gouraud')
                plt.title(f'{subject_id} - {channel} - Eyes Open (R01)')
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [s]')
                plt.colorbar(label='Power/Frequency (dB/Hz)')

                # R02 (Eyes Closed)
                plt.subplot(1, 2, 2)
                plt.pcolormesh(t_R02, f_R02, 10 * np.log10(Sxx_R02), shading='gouraud')
                plt.title(f'{subject_id} - {channel} - Eyes Closed (R02)')
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [s]')
                plt.colorbar(label='Power/Frequency (dB/Hz)')

                plt.tight_layout()
                plt.show()
                
    except Exception as e:
        print(f'Error reading files for {subject_id}: {e}')