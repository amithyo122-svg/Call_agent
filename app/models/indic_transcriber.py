
import torch
import soundfile as sf
import numpy as np

from transformers import AutoModel
from app.config import settings


class IndicTranscriber:

    def __init__(self):
        self.model = None
        self.device = settings.DEVICE

    def load_model(self):

        if self.model is not None:
            return

        print("\nLoading AI4Bharat Indic-Conformer...")

        self.model = AutoModel.from_pretrained(
            settings.MODEL_ID,
            trust_remote_code=True
        )

        self.model.to(self.device)
        self.model.eval()

        print("Indic-Conformer loaded successfully!")

    def transcribe(self, audio_path):

        if self.model is None:
            self.load_model()

        print("\nPreparing audio...")

        # Read WAV file
        audio, sample_rate = sf.read(
            str(audio_path),
            dtype="float32",
            always_2d=True
        )

        # Convert stereo to mono
        audio = np.mean(audio, axis=1)

        # Convert NumPy array to PyTorch tensor
        waveform = torch.from_numpy(audio).unsqueeze(0)

        # Resample if required
        if sample_rate != 16000:
            import torchaudio.functional as AF

            waveform = AF.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=16000
            )

        waveform = waveform.to(self.device)

        print("Transcribing audio...")

        with torch.inference_mode():

            transcription = self.model(
                waveform,
                settings.LANGUAGE,
                settings.DECODING
            )

        return str(transcription).strip()
