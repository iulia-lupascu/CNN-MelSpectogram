import pandas as pd
import os
import pickle
import librosa
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

def load_data(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            with open(os.path.join(directory, filename), 'rb') as file:
                data.append(pickle.load(file))
    return data

directory = '/home/u387021/DL/unzipped_data/train'
dataset = load_data(directory)

# Print basic information about the dataset
print("Number of entries:", len(dataset))
if len(dataset) > 0:
    # Accessing dictionary keys for audio data and labels
    print("Example of data entry:", dataset[0])
    print("Type of audio data:", type(dataset[0]['audio_data']))
    print("Type of label:", type(dataset[0]['valence']))
    print("Shape of audio data (if numpy array):", dataset[0]['audio_data'].shape)


# Assuming 'dataset' is a list of dictionaries with 'audio_data' as one of the keys
# Calculate durations in seconds
sampling_rate = 8000  # Hz
durations = [len(entry['audio_data']) / sampling_rate for entry in dataset]

p75 = np.percentile(durations, 75)

# Plotting the durations with the 75th percentile
plt.figure(figsize=(10, 6))
plt.hist(durations, bins=20, color='blue', alpha=0.7)
plt.axvline(p75, color='red', linestyle='dashed', linewidth=1)
plt.text(p75, plt.ylim()[1]*0.9, '75th percentile: {:.2f}s'.format(p75), color = 'red')
plt.title('Distribution of Audio File Durations')
plt.xlabel('Duration of Audio Files (seconds)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

def normalize_audio(audio):
    max_value = np.max(np.abs(audio))
    if max_value > 0:
        audio = audio / max_value
    return audio

# Applying normalization to each audio sample in the dataset
for entry in dataset:
    entry['audio_data'] = normalize_audio(entry['audio_data'])


def trim_audio_files(audio, target_length_sec=6, sr=8000):
    target_samples = target_length_sec * sr  # Calculate the number of samples for 5 seconds
    if len(audio) > target_samples:
        return audio[:target_samples]
    return audio  # Return the original audio if it's shorter than the target length


def pad_audio_files(audio, target_length_sec=6, sr=8000):
    target_samples = target_length_sec * sr  # Calculate the number of samples for 5 seconds
 
    current_length = len(audio)
    if current_length < target_samples:
        # Calculate how many zeros need to be added
        pad_length = target_samples - current_length
        # Return the padded audio
        return np.pad(audio, (0, pad_length), 'constant')
    return audio  # Return the original audio if it's longer than the target length


for entry in dataset:
    # Apply trimming and then padding
    trimmed_audio = trim_audio_files(entry['audio_data'])
    entry['audio_data'] = pad_audio_files(trimmed_audio)


sampling_rate = 8000  # Hz
durations = [len(entry['audio_data']) / sampling_rate for entry in dataset]

# Plotting the durations with the 75th percentile
plt.figure(figsize=(10, 6))
plt.hist(durations, bins=20, color='blue', alpha=0.7)
plt.title('Distribution of Audio File Durations')
plt.xlabel('Duration of Audio Files (seconds)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

def audio_to_melspectrogram(audio, sr=8000, n_fft=2048, hop_length=512, n_mels=128):
    """
    Convert audio to a Mel Spectrogram with a sampling rate of 8000 Hz.
    :param audio: numpy array, the audio signal
    :param sr: int, sampling rate (8000 Hz for this example)
    :param n_fft: int, length of the FFT window ()
    :param hop_length: int, number of samples between successive frames
    :param n_mels: int, number of Mel bands to generate (it is the same in some paper)
    :return: 2D numpy array, Mel spectrogram in dB
    """
    # Compute the Mel spectrogram
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

    # Convert to log scale (dB)
    S_DB = librosa.power_to_db(S, ref=np.max)
    
    return S_DB

normalized_dataset = []
for entry in dataset:
    audio_data = entry['audio_data']
    mel_spectrogram = audio_to_melspectrogram(audio=audio_data, sr=8000)
    normalized_dataset.append(mel_spectrogram)  # Append only the Mel spectrogram


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Assuming 'dataset' and 'normalized_dataset' are defined and loaded
labels = np.array([entry['valence'] for entry in dataset])

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(normalized_dataset, labels, test_size=0.2, random_state=42)

# Assuming that 'normalized_dataset' is a list and each entry contains a 'mel_spectrogram' key
sample_spectrogram_shape = normalized_dataset[0].shape

# Now we can extract the number of frequency bins and time steps
frequency_bins, time_steps = sample_spectrogram_shape

print(f"Frequency bins: {frequency_bins}, Time steps: {time_steps}")

class CNNModel2(nn.Module):
    def __init__(self, n_mels, time_steps, dropout_rate=0.5):
        super(CNNModel2, self).__init__()

        # Parallel Convolutional Layers
        self.conv1 = nn.Conv2d(1, 200, kernel_size=(12, 16))
        self.conv2 = nn.Conv2d(1, 200, kernel_size=(18, 24))
        self.conv3 = nn.Conv2d(1, 200, kernel_size=(24, 32))
        self.conv4 = nn.Conv2d(1, 200, kernel_size=(30, 40))
        
        # Max Pooling Layers
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))

        # Calculate the flattened size
        size1 = ((n_mels - 12 + 1) // 2) * ((time_steps - 16 + 1) // 2)
        size2 = ((n_mels - 18 + 1) // 2) * ((time_steps - 24 + 1) // 2)
        size3 = ((n_mels - 24 + 1) // 2) * ((time_steps - 32 + 1) // 2)
        size4 = ((n_mels - 30 + 1) // 2) * ((time_steps - 40 + 1) // 2)
        flattened_size = 200 * (size1 + size2 + size3 + size4)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(flattened_size, 400)
        self.bn1 = nn.BatchNorm1d(400)
        self.fc2 = nn.Linear(400, 200)
        self.bn2 = nn.BatchNorm1d(200)
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(200, 1)
    
    def forward(self, x):
        # Convolutional layers with ReLU activation
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        x4 = F.relu(self.conv4(x))
        
        # Max pooling layers
        x1 = self.pool1(x1)
        x2 = self.pool2(x2)
        x3 = self.pool3(x3)
        x4 = self.pool4(x4)
        
        # Flatten each parallel path
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)
        x4 = x4.view(x4.size(0), -1)
        
        # Concatenate all paths
        x = torch.cat((x1, x2, x3, x4), dim=1)
        
        # Fully connected layers with batch normalization and ReLU activation
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Output layer
        x = self.output(x)
        
        return x

# Create the model instance
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNModel2(frequency_bins, time_steps).to(device)

# Extract spectrograms and labels
train_spectrograms = np.array([entry for entry in X_train])
train_labels = np.array([entry for entry in y_train])
val_spectrograms = np.array([entry for entry in X_val])
val_labels = np.array([entry for entry in y_val])

# Convert numpy arrays to PyTorch tensors
X_train = torch.tensor(train_spectrograms, dtype=torch.float32).unsqueeze(1).to(device)  # Add a channel dimension
y_train = torch.tensor(train_labels, dtype=torch.float32).view(-1, 1).to(device)
X_val = torch.tensor(val_spectrograms, dtype=torch.float32).unsqueeze(1).to(device)
y_val = torch.tensor(val_labels, dtype=torch.float32).view(-1, 1).to(device)

# Create dataloaders
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

val_data = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

# Define loss function and optimizer
criterion = nn.L1Loss().to(device)  # MAE
optimizer = optim.Adam(model.parameters())

# Training loop
for epoch in range(20):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)  # Move data and targets to GPU
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
        val_loss /= len(val_loader)

    print(f'Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

PATH = ' /home/u387021/DL/model.pth'
torch.save(model.state_dict(), PATH)


# Determine the outliers
model.eval()
errors = []
predictions = []
with torch.no_grad():
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        abs_errors = torch.abs(output - target)
        errors.extend(abs_errors.cpu().numpy())
        predictions.extend(output.cpu().numpy())

errors = np.array(errors)
predictions = np.array(predictions)
outlier_indices = np.argsort(errors, axis=0)[-10:].flatten()  # Top 10 largest errors
outliers = X_val.cpu().numpy()[outlier_indices]
outlier_errors = errors[outlier_indices]
outlier_predictions = predictions[outlier_indices]
outlier_true_values = y_val.cpu().numpy()[outlier_indices]

# Visualize the outliers
plt.figure(figsize=(20, 15))
for i, idx in enumerate(outlier_indices):
    audio_data = dataset[idx]['audio_data']  # Make sure you have access to the original dataset
    mel_spectrogram = audio_to_melspectrogram(audio=audio_data, sr=8000)
    
    plt.subplot(2, 5, i + 1)
    librosa.display.specshow(mel_spectrogram, sr=8000, hop_length=512, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    
    # Extract scalar values for the title
    true_value = outlier_true_values[i].item() if isinstance(outlier_true_values[i], np.ndarray) else outlier_true_values[i]
    pred_value = outlier_predictions[i].item() if isinstance(outlier_predictions[i], np.ndarray) else outlier_predictions[i]
    error_value = outlier_errors[i].item() if isinstance(outlier_errors[i], np.ndarray) else outlier_errors[i]
    
    plt.title(f"True: {true_value:.4f}, Pred: {pred_value:.4f}, Err: {error_value:.4f}", fontsize=12)

plt.tight_layout()
plt.savefig('outliers_mel_spectrograms.png', dpi=300)
plt.show()
