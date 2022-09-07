import config
import torch
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, BertTokenizer


class ModelTrainClass():
    def __init__(self, data, model_ckpt, device) -> None:
        self.data = data
        self.num_labels = self.data["label"].unique().size()[0]
        self.model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=self.num_labels).to(device)
        self.tokenizer = BertTokenizer.from_pretrained(model_ckpt)
        self.model_ckpt = model_ckpt
        

    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds, average="weighted")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1}

    def model_train_args(self, enc_data):
        batch_size = config.BATCH_SIZE
        logging_steps = len(enc_data) // batch_size
        model_name = config.MODEL_OUT_NAME
        training_args = TrainingArguments(output_dir=model_name,
                                  num_train_epochs=2,
                                  learning_rate=2e-5,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  weight_decay=0.01,
                                  evaluation_strategy="epoch",
                                  disable_tqdm=False,
                                  logging_steps=logging_steps,
                                  push_to_hub=True,
                                  log_level="error")
        return training_args

    def train_model(self):
        trainer = Trainer(model=self.model, args=self.model_train_args(self.data),
                  compute_metrics=self.compute_metrics,
                  train_dataset=self.data,
                  eval_dataset=self.data,
                  tokenizer=self.tokenizer)
        trainer.train()
        trainer.push_to_hub()