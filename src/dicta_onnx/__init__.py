from dicta_onnx.model import OnnxDiacritizationModel
import re

class Dicta:
    def __init__(self, model_path: str):
        self.model = OnnxDiacritizationModel(model_path)
    
    def add_diacritics(self, sentences: list | str, mark_matres_lectionis: str | None = None) -> str:
        """
        Adds niqqud (Hebrew diacritics) to the given text.

        Parameters:
        - sentences (list | str): A string or a list of strings to be processed. Each string should not exceed 2048 characters.
        - mark_matres_lectionis (str | None, optional): A string used to mark niqqud male. For example, if set to '|', 
            "לִימּוּדָיו" will be returned as "לִי|מּוּדָיו". Default is None (no marking).

        Returns:
        - str: The text with added diacritics.
        """

        if isinstance(sentences, str):
            sentences = [sentences]
        result = self.model.predict(sentences, mark_matres_lectionis=mark_matres_lectionis)
        return result[0]
    
    def get_niqqud_male(self, text: str, mark_matres_lectionis: str):
        """
        Based on given mark character remove the mark character to keep it as niqqud male
        """
        return text.replace(mark_matres_lectionis, '')
    
    def get_niqqud_haser(self, text: str, mark_matres_lectionis: str):
        """
        Based on given mark_matres_lectionis remove the niqqud niqqud male character along with the mark character
        """
        return re.sub(r'.\|', '', text) # Remove {char}{matres_lectionis}
        