import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def load_model(model_name="theainerd/Wav2Vec2-large-xlsr-hindi"):
    # Load the Wav2Vec2 processor and model
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    return processor, model


def preprocess_audio(audio_tensor, processor, sample_rate=16000):
    # Load the audio file
    speech_array, sr = torchaudio.load(file_path)
    # Resample if necessary
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        speech_array = resampler(speech_array)
    # Process the audio input to fit the model
    input_values = processor(speech_array.squeeze().numpy(), return_tensors="pt", sampling_rate=sample_rate).input_values
    return input_values


def transcribe_audio(model, processor, input_values):
    # Get the logits (raw predictions)
    logits = model(input_values).logits
    # Take argmax to get the predicted IDs
    predicted_ids = torch.argmax(logits, dim=-1)
    # Decode the IDs to text
    transcription = processor.batch_decode(predicted_ids)
    return transcription[0]


file_path = "/home/shaukat/mycode/btp/BTP-backend/code/test.wav"
model_name = "theainerd/Wav2Vec2-large-xlsr-hindi"
processor, model = load_model(model_name)

input_values = preprocess_audio(file_path, processor)
predicted_text = transcribe_audio(model, processor, input_values)
print(f"Transcription: {predicted_text}")
