import argparse, os, json, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from torch import nn
from utils import load_tweeteval_sentiment, HFEncodedDataset, eval_loop

def parse():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", choices=["train","validation","test"], default="test")
    return ap.parse_args()

def main():
    args = parse()
    tok = AutoTokenizer.from_pretrained(args.ckpt, use_fast=True)
    labels = json.load(open(os.path.join(args.ckpt,"labels.json")))
    ds = load_tweeteval_sentiment()
    def enc_split(split):
        enc = tok(list(ds[split]["text"]), truncation=True, padding=True, max_length=128)
        y = ds[split]["label"]
        return HFEncodedDataset(enc, y)
    dl = DataLoader(enc_split(args.split), batch_size=32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=None
    try:
        model = AutoModelForSequenceClassification.from_pretrained(args.ckpt).to(device)
    except Exception:
        base = AutoModel.from_pretrained(args.ckpt).to(device)
        hidden = base.config.hidden_size if hasattr(base.config,"hidden_size") else base.config.d_model
        class Head(nn.Module):
            def __init__(self): super().__init__(); self.drop=nn.Dropout(0.2); self.fc=nn.Linear(hidden,len(labels))
            def forward(self,x): return self.fc(self.drop(x))
        head = Head().to(device)
        head.load_state_dict(torch.load(os.path.join(args.ckpt,"head.pt"), map_location=device))
        @torch.no_grad()
        def get_pooled(outs):
            if hasattr(outs,"pooler_output") and outs.pooler_output is not None:
                return outs.pooler_output
            return outs.last_hidden_state.mean(dim=1)
        class Wrapped(nn.Module):
            def __init__(self, base, head): super().__init__(); self.base=base; self.head=head
            def forward(self, input_ids=None, attention_mask=None, labels=None):
                outs = self.base(input_ids=input_ids, attention_mask=attention_mask)
                pooled = get_pooled(outs); logits = self.head(pooled)
                return type("obj", (), {"logits": logits})
        model = Wrapped(base, head).to(device)
    metrics = eval_loop(model, dl, device)
    print(metrics)
    json.dump(metrics, open(os.path.join(args.ckpt, f"{args.split}_metrics.json"),"w"), indent=2)

if __name__ == "__main__":
    main()
