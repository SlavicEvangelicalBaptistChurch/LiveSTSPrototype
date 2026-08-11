from deep_translator import GoogleTranslator

class Translator:
    def __init__(self, source, target):
        self.source = source
        self.target = target

    def translate(self, text):
        try:
            translated_text = ""
            while(len(translated_text) == 0):
                translated_text = GoogleTranslator(source=self.source, target=self.target).translate(text)

            return translated_text
        except Exception as e:
            print(e)