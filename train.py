from transformers import RobertaTokenizer
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from transformers import RobertaForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
import numpy as np
import evaluate

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

    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=2, id2label=id2label, label2id=label2id)

    # train with trainer
    training_args = TrainingArguments(
        output_dir="hall_guard",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

if __name__ == "__main__":
    main()
