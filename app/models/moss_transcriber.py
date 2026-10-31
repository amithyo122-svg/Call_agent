
import torch

from transformers import AutoModelForCausalLM, AutoProcessor

from moss_transcribe_diarize import parse_transcript

from moss_transcribe_diarize.inference_utils import (
    build_transcription_messages,
    generate_transcription,
    resolve_device,
)

from app.config import settings


class MossTranscriber:

    def __init__(self):

        self.model = None
        self.processor = None
        self.device = None
        self.dtype = None

    def load_model(self):

        if self.model is not None:
            return

        print("\nLoading MOSS transcription model...")

        self.device = resolve_device(settings.DEVICE)

        self.dtype = (
            torch.bfloat16
            if self.device.type == "cuda"
            else torch.float32
        )

        print(f"Device: {self.device}")
        print(f"Data type: {self.dtype}")

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.MODEL_ID,
            trust_remote_code=True,
            dtype="auto",
            attn_implementation="sdpa",
        ).to(dtype=self.dtype).to(self.device).eval()

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            settings.MODEL_ID,
            trust_remote_code=True,
        )

        print("MOSS model loaded successfully!")

    def transcribe(self, audio_path):

        if self.model is None:
            self.load_model()

        print("\nTranscribing audio...")

        # Prepare audio messages
        messages = build_transcription_messages(
            str(audio_path)
        )

        # Run inference
        result = generate_transcription(
            self.model,
            self.processor,
            messages,
            max_new_tokens=settings.MAX_NEW_TOKENS,
            do_sample=False,
            device=self.device,
            dtype=self.dtype,
        )

        raw_text = result["text"]

        # Extract speaker and timestamp information
        parsed_segments = parse_transcript(raw_text)

        segments = []

        for segment in parsed_segments:

            segments.append({
                "start": segment.start,
                "end": segment.end,
                "speaker": segment.speaker,
                "text": segment.text,
            })

        return {
            "text": raw_text,
            "segments": segments,
        }
