from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    AUDIO_DTYPE: str = "float32"
    RECORD_DURATION: int = 10

    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    RECORDINGS_DIR: Path = BASE_DIR / "recordings"
    TRANSCRIPTS_DIR: Path = BASE_DIR / "transcripts"

    MODEL_ID: str = "ai4bharat/indic-conformer-600m-multilingual"
    LANGUAGE: str = "ml"
    DECODING: str = "ctc"
    DEVICE: str = "cpu"

    TRANSLATION_MODEL_ID: str = "ai4bharat/indictrans2-indic-en-dist-200M"
    SOURCE_LANGUAGE: str = "mal_Mlym"
    TARGET_LANGUAGE: str = "eng_Latn"


settings = Settings()
