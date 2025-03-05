import os
from spellchecker import SpellChecker


class LLM:
    letters: list[str]
    max_letters: int
    norwegian_letters: set[str]
    spell_checker: SpellChecker

    def __init__(
        self,
        max_letters: int = 5,
        norwegian_path: str = "model/norwegian_words.txt",
    ) -> None:
        print("Starting LLM")
        self.letters = []
        self.max_letters = max_letters
        self.spell_checker = SpellChecker(language=None)
        self.spell_checker.word_frequency.load_text_file(norwegian_path)

        if os.path.isfile(norwegian_path):
            with open(norwegian_path, "r+") as f:
                words = f.readlines()
                words = [word.strip() for word in words]
                self.norwegian_letters = set(words)
        else:
            self.norwegian_letters = set()

        pass

    def get_result(self) -> str:
        input_text = "".join(self.letters)
        self.letters = []

        closest_word = self.spell_checker.correction(input_text)
        closest_words = self.spell_checker.candidates(input_text)
        print(closest_words)
        if closest_word is None:
            return ""

        return closest_word

    def add_letter(self, letter: str) -> None:
        self.letters.append(letter)
        return

    pass
