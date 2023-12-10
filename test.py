from transformers import RobertaTokenizer
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from transformers import RobertaForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
import numpy as np
import evaluate
import json

MODEL_PATH = "hall_guard/checkpoint-6310"

def test_accuracy():
    # Load the dataset
    data_files = {"test": "data/test.csv"}
    dataset = load_dataset("csv", data_files=data_files)

    # Tokenize the dataset
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # create evaluation metrics
    accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    # evaluate model on test
    model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)

    training_args = TrainingArguments(output_dir="hall_guard")

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    result = trainer.evaluate()

    with open("test_eval.json", "w") as f:
        json.dump(result, f, indent=4)

def test_accuracy_by_category():
    # Load the dataset
    data_files = {
        "test_good": "data/test_good.csv",
        "test_bad_word": "data/test_bad_word.csv",
        "test_bad_response": "data/test_bad_response.csv"
    }
    dataset = load_dataset("csv", data_files=data_files)

    # Tokenize the dataset
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # create evaluation metrics
    accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    # evaluate model on test
    model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)

    training_args = TrainingArguments(output_dir="hall_guard")

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    result = {
        key: trainer.evaluate(eval_dataset=data)
        for key, data in tokenized_dataset.items()
    }

    with open("test_eval_category.json", "w") as f:
        json.dump(result, f, indent=4)

if __name__ == "__main__":
    test_accuracy()
    #test_accuracy_by_category()
