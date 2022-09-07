import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
import numpy as np
import gc

class Dataset:

    def __init__(self, model_ckpt, device):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(model_ckpt)
        self.model = BertModel.from_pretrained(model_ckpt, output_hidden_states=True).to(self.device)
        
    def create_data(self, dataset_id, split="train"):
        data = load_dataset(dataset_id, split=split)
        return data

    def tokenize_data_fn(self, batch):
        return self.tokenizer(batch["text"], padding=True, truncation=True)

    def tokenize(self, data, tokenize_fn, batched=True, batch_size = None):
        return data.map(tokenize_fn, batched=batched, batch_size=batch_size)

    def embedder_fn(self, batch):
        inputs = {k:v.to(self.device) for k,v in batch.items()
              if k in self.tokenizer.model_input_names}
        with torch.no_grad():
            last_hidden_state = self.model(**inputs).last_hidden_state
        return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}

    def run(self, dataset_id):
        data = self.create_data(dataset_id)
        print("Dataset Created")
        enc_data = self.tokenize(data, self.tokenize_data_fn)
        print("Dataset Tokenized")
        torch.cuda.empty_cache()
        gc.collect()
        enc_data.set_format("torch",
                            columns=["input_ids", "attention_mask", "label"])
        emb_data = np.array(enc_data.map(self.embedder_fn, batched=True)["hidden_state"])
        emb_label = np.array(enc_data["label"])
        print("Dataset Embedding generated")
        return enc_data, emb_data, emb_label