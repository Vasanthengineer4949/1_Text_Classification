import config
import torch
from dataset import Dataset
from model_train import ModelTrainClass

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Dataset(config.MODEL_CKPT, device)
    print("Data Class Object Created")
    enc_data, emb_data, emb_label = dataset.run(config.DATASET_ID)
    print(emb_data)
    print("Data Tokenization Completed")
    model_train_cls = ModelTrainClass(enc_data, config.MODEL_CKPT, device)
    model_train_cls.train_model()
