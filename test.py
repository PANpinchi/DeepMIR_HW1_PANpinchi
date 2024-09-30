from sklearn.metrics import classification_report
from transformers import Wav2Vec2FeatureExtractor, AutoModel
import torch
from torch.utils.data import DataLoader
from datasets import AudioDataset
import numpy as np
from tqdm import tqdm
from train import ClassificationHead
import os
import argparse


class_names = ['Piano', 'Percussion', 'Organ', 'Guitar', 'Bass', 'Strings', 'Voice', 'Wind Instruments', 'Synth']


def main(args):
    # Set up device (CUDA or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.frozen:
        results_dir = "./results_classifier_frozen"
    else:
        results_dir = "./results_classifier"

    # Loading model weights
    model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).to(device)
    model.eval()  # Set model to evaluation mode

    # Loading the corresponding preprocessor config
    processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)

    # Initialize validation dataset
    val_data_dir = "hw1/slakh/test"
    val_label_file = "hw1/slakh/test_labels.json"
    val_dataset = AudioDataset(val_data_dir, val_label_file, processor)

    # Create DataLoader for validation
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Load the trained classifier model
    classifier = ClassificationHead(input_dim=768, num_classes=9).to(device)
    classifier.load_state_dict(torch.load(os.path.join(results_dir, "classifier_model_best.pth")))
    classifier.eval()  # Set classifier to evaluation mode

    # Initialize lists for storing predictions and true labels
    all_predictions = []
    all_true_labels = []

    # Validation loop
    with torch.no_grad():
        for batch in tqdm(val_loader):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Get model outputs
            outputs = model(input_values=inputs, output_hidden_states=True)
            hidden_states = torch.stack(outputs.hidden_states, dim=1)
            time_reduced_hidden_states = hidden_states.mean(-2)
            logits = classifier(time_reduced_hidden_states)

            # Convert logits to binary predictions (or threshold for multi-class)
            probabilities = torch.sigmoid(logits)
            predicted_labels = (probabilities > args.threshold).float()

            # Store predicted and true labels for classification report
            all_predictions.append(predicted_labels.cpu().numpy())
            all_true_labels.append(labels.cpu().numpy())

    # Concatenate all batches to form the complete prediction and label arrays
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_true_labels = np.concatenate(all_true_labels, axis=0)

    # Generate classification report
    report = classification_report(all_true_labels, all_predictions, target_names=class_names)
    print("Classification Report:\n", report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test a Wav2Vec2 model with a classification head.")
    parser.add_argument('--frozen', action='store_true', help='Frozen backbone')
    parser.add_argument('--threshold', type=float, default=0.5, help='Logits to probabilities threshold')
    args = parser.parse_args()
    main(args)
