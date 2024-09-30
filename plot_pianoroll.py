import matplotlib.pyplot as plt
from glob import glob
import os
import json
import pretty_midi
import numpy as np
import argparse

from transformers import Wav2Vec2FeatureExtractor, AutoModel
import torch
from train import ClassificationHead
import librosa
from tqdm import tqdm

with open('./hw1/class_idx2MIDIClass.json') as f:
    class_idx2MIDIClass = json.load(f)
with open('./hw1/idx2instrument_class.json') as f:
    idx2instrument_class = json.load(f)
with open('./hw1/MIDIClassName2class_idx.json') as f:
    MIDIClassName2class_idx = json.load(f)

categories = [
    'Piano', 'Percussion', 'Organ', 'Guitar', 'Bass',
    'Strings', 'Voice', 'Wind Instruments', 'Synth'
]


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_track_dir', type=str,
                        help='source(test) midi track folder path', default='./hw1/test_track')
    parser.add_argument('--save_dir', type=str,
                        help='saved fig folder path', default='./')
    parser.add_argument('--frozen', action='store_true', help='Frozen backbone')
    parser.add_argument('--threshold', type=float, default=0.5, help='Logits to probabilities threshold')
    args = parser.parse_args()
    return args


def extract_pianoroll_from_midi(midi_file_path, time_step=5.0):
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    # print(midi_data)

    # Determine total duration in seconds
    total_time = midi_data.get_end_time()

    # Create an empty pianoroll matrix without the "Empty" class
    num_classes = len(class_idx2MIDIClass)
    num_time_steps = int(np.ceil(total_time / time_step))
    pianoroll = np.zeros((num_classes, num_time_steps))

    # Process each instrument in the MIDI file
    for instrument in midi_data.instruments:
        program_num = instrument.program

        if instrument.is_drum:
            instrument_class = 128
        else:
            # Determine the class for this instrument
            instrument_class = idx2instrument_class.get(str(program_num), None)
        if instrument_class and instrument_class in MIDIClassName2class_idx:
            class_idx = MIDIClassName2class_idx[instrument_class]

            # Fill the pianoroll for each note
            for note in instrument.notes:
                start_time = note.start
                end_time = note.end
                start_idx = int(np.floor(start_time / time_step))
                end_idx = int(np.ceil(end_time / time_step))
                pianoroll[class_idx, start_idx:end_idx] = 1  # Mark the note as present

    return pianoroll


def pianoroll_comparison(true_pianoroll, pred_pianoroll, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))

    # Plotting the true pianoroll
    axes[0].imshow(true_pianoroll, aspect='auto', cmap='Oranges', interpolation='nearest')
    axes[0].set_title('True Labels')
    axes[0].set_yticks(range(len(categories)))
    axes[0].set_yticklabels(categories)
    axes[0].set_xlabel('Time Steps')

    # Plotting the predicted pianoroll
    axes[1].imshow(pred_pianoroll, aspect='auto', cmap='Oranges', interpolation='nearest')
    axes[1].set_title('Predicted Labels')
    axes[1].set_yticks(range(len(categories)))
    axes[1].set_yticklabels(categories)
    axes[1].set_xlabel('Time Steps')

    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path)


def predict_instrument_activity(audio_path, model, processor, classifier, device, chunk_duration=5, threshold=0.5):
    # Load the entire audio file
    audio, sr = librosa.load(audio_path, sr=24000)

    # Calculate the total number of chunks based on the audio length
    total_duration = librosa.get_duration(y=audio, sr=sr)
    num_chunks = int(np.ceil(total_duration / chunk_duration))

    # Initialize an empty list to store the predictions
    all_predictions = []

    # Process each 5-second chunk
    for chunk_idx in tqdm(range(num_chunks)):
        # Extract the start and end of the current chunk
        start_sample = int(chunk_idx * chunk_duration * sr)
        end_sample = int(min((chunk_idx + 1) * chunk_duration * sr, len(audio)))

        # Extract the chunk
        audio_chunk = audio[start_sample:end_sample]

        # If the chunk is smaller than the required length (at the end), pad it
        if len(audio_chunk) < chunk_duration * sr:
            padding = np.zeros(int(chunk_duration * sr) - len(audio_chunk))
            audio_chunk = np.concatenate([audio_chunk, padding])

        # Process the chunk with the processor and make predictions
        inputs = processor(audio_chunk, return_tensors="pt", sampling_rate=24000, padding=True).input_values.to(device)

        with torch.no_grad():
            outputs = model(input_values=inputs, output_hidden_states=True)
            hidden_states = torch.stack(outputs.hidden_states, dim=1)
            time_reduced_hidden_states = hidden_states.mean(-2)
            logits = classifier(time_reduced_hidden_states)

        # Convert logits to binary predictions
        probabilities = torch.sigmoid(logits)
        predicted_labels = (probabilities > threshold).float().cpu().numpy()
        all_predictions.append(predicted_labels)

    # Concatenate all chunks to form the final predicted pianoroll
    pred_pianoroll = np.concatenate(all_predictions, axis=0)
    pred_pianoroll = np.transpose(pred_pianoroll, (1, 0))

    return pred_pianoroll


def main(opt):
    # Load your model (e.g., Wav2Vec2-based model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).to(device)
    model.eval()

    processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)

    classifier = ClassificationHead(input_dim=768, num_classes=9).to(device)  # Assuming 9 classes
    classifier.load_state_dict(torch.load(os.path.join(opt.save_dir, "classifier_model_best.pth")))
    classifier.eval()

    # Load the MIDI and corresponding audio files
    midi_path_list = glob(os.path.join(opt.test_track_dir, '*.mid'))
    audio_path_list = glob(os.path.join(opt.test_track_dir, '*.flac'))

    for midi_path, audio_path in tqdm(zip(midi_path_list, audio_path_list)):
        name = midi_path.split('/')[-1].split('.')[0]
        # Extract true pianoroll from MIDI file
        true_pianoroll = extract_pianoroll_from_midi(midi_path)

        # Predict the instrument activity every 5 seconds for the corresponding audio file
        pred_pianoroll = predict_instrument_activity(audio_path, model, processor, classifier, device,
                                                     chunk_duration=5, threshold=opt.threshold)

        # Ensure the predicted pianoroll has the same shape as the true pianoroll
        print('true_pianoroll.shape: ', true_pianoroll.shape)
        print('pred_pianoroll.shape: ', pred_pianoroll.shape)

        # pred_pianoroll is your model predict result please load your results here
        # pred_pianoroll.shape should be [9, L] and the L should be equal to true_pianoroll
        # pred_pianoroll = np.zeros(true_pianoroll.shape)
        save_path = os.path.join(opt.save_dir, name + '.png')
        pianoroll_comparison(true_pianoroll, pred_pianoroll, save_path)
    pass


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)