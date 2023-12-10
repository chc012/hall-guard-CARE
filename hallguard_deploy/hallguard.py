from transformers import RobertaTokenizer
from transformers import RobertaForSequenceClassification
from transformers import TextClassificationPipeline
import numpy as np

DEFAULT_MODEL_PATH = "model"
SEP_TOKEN = "</s>"

class HallGuard:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH,
                 prob_threshold: float = 0.5):
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = RobertaForSequenceClassification.from_pretrained(
            model_path)
        self.classifier = TextClassificationPipeline(
            model=model, tokenizer=tokenizer, top_k=None)
        self.prob_threshold = prob_threshold

    def predict(self, inputs: list[list[str]]) -> list[list[float]]:
        texts = self._convert_to_texts(inputs)
        return self._predict(texts)

    def _convert_to_texts(self, inputs: list[list[str]]) -> list[str]:
        return [(" " + SEP_TOKEN + " ").join(d) for d in inputs]

    def _predict(self, texts: list[str]) -> list[list[float]]:
        outputs = self.classifier(texts)
        scores = [pred["score"] for output in outputs for pred in output
                       if pred["label"] == "POSITIVE"]
        preds = [int(score > self.prob_threshold)
                 for score in scores]
        return [{"label": pred, "score": score}
                for pred, score in zip(preds, scores)]
