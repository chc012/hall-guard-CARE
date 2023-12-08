from transformers import RobertaTokenizer
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from transformers import RobertaForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
import numpy as np
import evaluate
import json

def main():
    # Load the dataset
    data_files = {
        "train": "data/train.csv",
        "val": "data/val.csv",
        "test": "data/test.csv"
    }
    dataset = load_dataset("csv", data_files=data_files)

    # Tokenize the dataset
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
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

    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    # hyperparameter search
    def model_init(trial):
        return RobertaForSequenceClassification.from_pretrained(
            "roberta-base", num_labels=2, id2label=id2label, label2id=label2id)

    # train with trainer
    training_args = TrainingArguments(
        output_dir="hall_guard",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="logs",
        logging_steps=100,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=None,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        model_init=model_init,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Hyperparameter search
    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-6, 1e-3, log=True),
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", [16, 32, 64, 128]),
        }

    best_trial = trainer.hyperparameter_search(
        direction="minimize",
        backend="optuna",
        hp_space=optuna_hp_space,
        n_trials=20
    )

    print("Best score: {:.3f}".format(best_trial.objective))
    print("Best hyperparameters:", best_trial.hyperparameters)

    best_trial.hyperparameters["score"] = best_trial.objective
    with open("best_model.json", "w") as f:
        json.dump()

    # trainer.train()

if __name__ == "__main__":
    main()
