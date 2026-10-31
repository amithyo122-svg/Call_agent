
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor

from app.config import settings


class IndicTranslator:

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.device = settings.DEVICE

    def load_model(self):

        if self.model is not None:
            return

        print("\nLoading IndicTrans2 translation model...")

        model_id = settings.TRANSLATION_MODEL_ID

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            trust_remote_code=True
        ).to(self.device)

        self.model.eval()

        self.processor = IndicProcessor(inference=True)

        print("IndicTrans2 loaded successfully!")

    def translate(self, text):

        if self.model is None:
            self.load_model()

        if not text or not text.strip():
            return ""

        print("\nTranslating Malayalam to English...")

        batch = self.processor.preprocess_batch(
            [text],
            src_lang=settings.SOURCE_LANGUAGE,
            tgt_lang=settings.TARGET_LANGUAGE
        )

        inputs = self.tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True
        ).to(self.device)

        with torch.inference_mode():

            generated_tokens = self.model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1
            )

        decoded = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        translations = self.processor.postprocess_batch(
            decoded,
            lang=settings.TARGET_LANGUAGE
        )

        return translations[0]
