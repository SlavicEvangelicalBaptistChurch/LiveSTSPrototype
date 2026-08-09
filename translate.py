from deep_translator import GoogleTranslator

class Translator:
    def __init__(self):
        self.source = "auto"
        self.target = "en"

    def translate(self, text, source = "", lang_dest=""):
        if (source == ""):
            source = self.source

        if(lang_dest == ""):
            lang_dest = self.target

        try:
            translated_text = ""
            while(len(translated_text) == 0):
                translated_text = GoogleTranslator(source="auto", target=lang_dest).translate(text)

            return translated_text
        except Exception as e:
            print(e)