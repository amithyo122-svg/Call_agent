
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:

    # Audio configuration
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    AUDIO_DTYPE: str = "float32"
    RECORD_DURATION: int = 10

    # Model configuration
    MODEL_ID: str = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
    DEVICE: str = "auto"
    MAX_NEW_TOKENS: int = 2048

    # Project directories
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    RECORDINGS_DIR: Path = BASE_DIR / "recordings"
    TRANSCRIPTS_DIR: Path = BASE_DIR / "transcripts"


settings = Settings()
