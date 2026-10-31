
from app.models.indic_translator import IndicTranslator


def main():

    translator = IndicTranslator()

    malayalam_text = "ഇന്ന് എനിക്ക് ഒരു ജോലി ഉണ്ട്."

    english_text = translator.translate(malayalam_text)

    print("\n========== TRANSLATION ==========")
    print("Malayalam:", malayalam_text)
    print("English:", english_text)


if __name__ == "__main__":
    main()
