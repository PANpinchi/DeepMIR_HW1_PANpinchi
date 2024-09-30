import os
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch


class MusicDataset(Dataset):
    def __init__(self, data_dir, label_file, transform=None):
        """
        Args:
            data_dir (str): Directory with all the .npy files.
            label_file (str): Path to the JSON file with labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform

        # Load labels from JSON
        with open(label_file, 'r') as f:
            self.labels = json.load(f)

        # List of all the .npy files available in the directory
        self.data_files = list(self.labels.keys())

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        # Get the file name based on index
        file_name = self.data_files[idx]

        # Load the .npy file
        file_path = os.path.join(self.data_dir, file_name)
        data = np.load(file_path)

        # Get the label for the file
        label = self.labels[file_name]

        # Convert data and label to tensors
        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        # Apply transformations if any
        if self.transform:
            data = self.transform(data)

        return data, label


# datasets.py
import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import torchaudio.transforms as T
from transformers import Wav2Vec2FeatureExtractor


class AudioDataset(Dataset):
    def __init__(self, data_dir, label_file, processor):
        self.data_dir = data_dir
        self.processor = processor
        self.resample_rate = processor.sampling_rate

        with open(label_file, 'r') as f:
            self.labels = json.load(f)

        self.data_files = list(self.labels.keys())

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_name = self.data_files[idx]
        file_path = os.path.join(self.data_dir, file_name)

        audio_data = np.load(file_path)
        audio_data = torch.from_numpy(audio_data)

        inputs = self.processor(audio_data, sampling_rate=self.resample_rate, return_tensors="pt")

        label = torch.tensor(self.labels[file_name], dtype=torch.float32)

        return inputs["input_values"].squeeze(0), label


# Usage in train.py
if __name__ == "__main__":
    # Example paths (adjust as needed)
    train_data_dir = 'hw1/slakh/train/'
    train_label_file = 'hw1/slakh/train_labels.json'

    # Initialize dataset
    train_dataset = MusicDataset(train_data_dir, train_label_file)

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Example usage of DataLoader
    for idx, (data, labels) in enumerate(train_loader):
        print("Batch data shape:", data.shape)
        print("Batch labels shape:", labels.shape)
        if idx == 5:
            break
