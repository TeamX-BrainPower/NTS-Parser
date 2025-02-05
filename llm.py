import fasttext
import fasttext.util
import os


class LLM:
    letters: list[str]
    max_letters: int
    model: fasttext.FastText._FastText
    norwegian_letters: set[str]

    def __init__(
        self,
        max_letters: int = 5,
        model_path: str = "model/cc.no.300.bin",
        norwegian_path: str = "model/norwegian_words.txt",
    ) -> None:
        print("Starting LLM")
        self.letters = []
        self.max_letters = max_letters

        if not os.path.isfile(model_path):
            print("Downloading fasttext dataset")
            fasttext.util.download_model("no", if_exists="ignore")
            os.rename("cc.no.300.bin", model_path)
            os.remove("cc.no.300.bin.gz")

        if os.path.isfile(norwegian_path):
            with open(norwegian_path, "r+") as f:
                words = f.readlines()
                words = [word.strip() for word in words]
                self.norwegian_letters = set(words)
        else:
            self.norwegian_letters = set()
        self.model = fasttext.load_model(model_path)

        pass

    def get_result(self) -> str:
        input_text = "".join(self.letters)
        self.letters = []

        if input_text in self.norwegian_letters:
            return input_text

        closest_words = self.model.get_nearest_neighbors(input_text, k=5)

        return closest_words[0][-1]

    def add_letter(self, letter: str) -> None:
        self.letters.append(letter)
        return

    pass
