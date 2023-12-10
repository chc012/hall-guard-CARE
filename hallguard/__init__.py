"""
Module for HallGuard, the wrapper class for hallucination guard with roberta.
"""
import os
from transformers import RobertaTokenizer
from transformers import RobertaForSequenceClassification
from transformers import TextClassificationPipeline

MODULE_PATH = os.path.abspath(os.path.dirname(__file__))
DEFAULT_MODEL_PATH = os.path.join(MODULE_PATH, "model")
SEP_TOKEN = "</s>"

class HallGuard:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH,
                 prob_threshold: float = 0.5):
        """
        Args:
            model_path (str): absolute path to the model directory.
            prob_threshold (float): threshold for the probability of the
                positive class.
        """
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = RobertaForSequenceClassification.from_pretrained(
            model_path)
        self.classifier = TextClassificationPipeline(
            model=model, tokenizer=tokenizer, top_k=None)
        self.prob_threshold = prob_threshold

    def predict(self, inputs: list[list[str]]) -> list[dict[str, float]]:
        """
        Make predictions on the given inputs.

        Args:
            inputs (list[list[str]]): list of conversation snippets
                (list of str).
        
        Returns:
            list[dict[str, float]]: list of predictions. Each prediction is
                a dictionary with keys "label" and "score".
                "label" is the predicted label based on prob_threshold;
                "score" is the probability of the positive class.
        """
        texts = self._convert_to_texts(inputs)
        return self._predict(texts)

    def _convert_to_texts(self, inputs: list[list[str]]) -> list[str]:
        """
        Helper. Convert the given inputs to digestable texts for roberta.

        Args:
            inputs (list[list[str]]): list of conversation snippets
                (list of str).

        Returns:
            list[str]: list of texts. Each text is a concatenation of
                conversation snippets with SEP_TOKEN in between.
        """
        return [(" " + SEP_TOKEN + " ").join(d) for d in inputs]

    def _predict(self, texts: list[str]) -> list[dict[str, float]]:
        """
        Helper. Make predictions on the given texts.

        Args:
            texts (list[str]): list of texts. See _convert_to_texts().

        Returns:
            list[dict[str, float]]: list of predictions. See predict().
        """
        outputs = self.classifier(texts)
        scores = [pred["score"] for output in outputs for pred in output
                       if pred["label"] == "POSITIVE"]
        preds = [int(score > self.prob_threshold)
                 for score in scores]
        return [{"label": pred, "score": score}
                for pred, score in zip(preds, scores)]
