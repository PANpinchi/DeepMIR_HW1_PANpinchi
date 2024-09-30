import matplotlib.pyplot as plt  # Import matplotlib for plotting
from transformers import Wav2Vec2FeatureExtractor, AutoModel
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from datasets import AudioDataset
from tqdm import tqdm
import os
import argparse


class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.aggregator = nn.Conv1d(in_channels=13, out_channels=1, kernel_size=1)
        self.fc1 = nn.Linear(input_dim, 512)  # First fully connected layer
        self.fc2 = nn.Linear(512, 256)  # Second fully connected layer
        self.fc3 = nn.Linear(256, num_classes)  # Output layer
        self.relu = nn.ReLU()  # ReLU activation
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization

    def forward(self, x):
        # Apply the aggregator and remove the channel dimension
        x = self.aggregator(x).squeeze(1)  # Shape: [batch_size, 1, seq_length] -> [batch_size, seq_length]

        # Pass through the first fully connected layer and activation
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout

        # Pass through the second fully connected layer and activation
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout

        # Final output layer
        x = self.fc3(x)
        return x


def plot_and_save_loss_accuracy(train_losses, val_losses, val_accuracies, save_path):
    """Plot and save training and validation losses and accuracies"""
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'r', label='Training Loss')
    plt.plot(epochs, val_losses, 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, 'g', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()

    # Save the figure
    plt.savefig(save_path)
    plt.close()


def main(args):
    # Set up device (CUDA or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Loading model weights
    model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).to(device)

    # Freeze the model parameters
    if args.frozen:
        for param in model.parameters():
            param.requires_grad = False

    # Loading the corresponding preprocessor config
    processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)

    # Initialize training dataset using datasets.py's AudioDataset class
    train_data_dir = "hw1/slakh/train"
    train_label_file = "hw1/slakh/train_labels.json"
    train_dataset = AudioDataset(train_data_dir, train_label_file, processor)

    # Initialize validation dataset
    val_data_dir = "hw1/slakh/validation"
    val_label_file = "hw1/slakh/validation_labels.json"
    val_dataset = AudioDataset(val_data_dir, val_label_file, processor)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Create classification head
    num_classes = 9
    classifier = ClassificationHead(input_dim=768, num_classes=num_classes).to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(classifier.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # Set up the training loop
    num_epochs = 10
    if args.frozen:
        model.eval()
    else:
        model.train()
    classifier.train()

    # Lists to track loss and accuracy
    train_losses = []
    val_losses = []
    val_accuracies = []

    # Variable to track the best validation accuracy
    best_val_accuracy = 0.0
    best_model_path = None  # Variable to store the path of the best model

    # Ensure the results_classifier directory exists
    if args.frozen:
        results_dir = "./results_classifier_frozen"
    else:
        results_dir = "./results_classifier"
    os.makedirs(results_dir, exist_ok=True)  # Create the directory if it does not exist

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, disable=args.quiet)

        for batch in train_loader_tqdm:
            inputs, labels = batch
            # Move inputs and labels to the correct device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # outputs.keys(): ['last_hidden_state', 'hidden_states']
            outputs = model(input_values=inputs, output_hidden_states=True)

            # last_hidden_state = outputs.last_hidden_state
            hidden_states = torch.stack(outputs.hidden_states, dim=1)

            # Reduce in time
            time_reduced_hidden_states = hidden_states.mean(-2)

            # Pass through classification head
            logits = classifier(time_reduced_hidden_states)

            # Compute loss
            loss = criterion(logits, labels)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))  # Save training loss
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

        # Validate the model
        model.eval()
        val_running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        val_loader_tqdm = tqdm(val_loader, disable=args.quiet)

        with torch.no_grad():
            for batch in val_loader_tqdm:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(input_values=inputs, output_hidden_states=True)
                hidden_states = torch.stack(outputs.hidden_states, dim=1)
                time_reduced_hidden_states = hidden_states.mean(-2)
                logits = classifier(time_reduced_hidden_states)

                loss = criterion(logits, labels)
                val_running_loss += loss.item()

                # Calculate accuracy
                probabilities = torch.sigmoid(logits)
                predicted_labels = (probabilities > 0.5).float()
                correct_predictions += (predicted_labels == labels).sum().item()
                total_predictions += labels.numel()  # Total number of labels in the batch

        val_loss = val_running_loss / len(val_loader)
        val_accuracy = correct_predictions / total_predictions * 100
        val_losses.append(val_loss)  # Save validation loss
        val_accuracies.append(val_accuracy)  # Save validation accuracy
        print(f'Validation Loss: {val_loss}, Accuracy: {val_accuracy:.2f}%')

        # If the current validation accuracy is better than the previous best, save the model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_path = os.path.join(results_dir, "classifier_model_best.pth")
            torch.save(classifier.state_dict(), best_model_path)
            print(f"New best model saved with validation accuracy: {best_val_accuracy:.2f}% at {best_model_path}")

        model.train()  # Switch back to training mode

    # Plot and save loss and accuracy after training
    plot_path = os.path.join(results_dir, "training_plot.png")
    plot_and_save_loss_accuracy(train_losses, val_losses, val_accuracies, plot_path)

    print(f"Training complete. Best validation accuracy: {best_val_accuracy:.2f}%")
    print(f"Best model saved at: {best_model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Wav2Vec2 model with a classification head.")
    parser.add_argument('--quiet', action='store_true', help='Disable progress bars')
    parser.add_argument('--frozen', action='store_true', help='Frozen backbone')
    args = parser.parse_args()
    main(args)
