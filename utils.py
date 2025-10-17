import torch, numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score

def load_tweeteval_sentiment():
    return load_dataset("tweet_eval","sentiment")

class HFEncodedDataset(Dataset):
    def __init__(self, enc, labels):
        self.enc = enc; self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k,v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

@torch.no_grad()
def eval_loop(model, dl, device):
    model.eval()
    preds, gold = [], []
    for batch in dl:
        for k in ["input_ids","attention_mask","labels"]:
            batch[k] = batch[k].to(device)
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        p = out.logits.argmax(dim=-1)
        preds.extend(p.tolist()); gold.extend(batch["labels"].tolist())
    acc = accuracy_score(gold, preds)
    f1m = f1_score(gold, preds, average="macro")
    f1w = f1_score(gold, preds, average="weighted")
    return {"accuracy": acc, "f1_macro": f1m, "f1_weighted": f1w}
