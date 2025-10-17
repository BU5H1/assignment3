# ============================================================
# This script evaluates a fine-tuned DistilBERT model (either
# the full fine-tuned version or the head-only version) on a
# specified dataset split (train, validation, or test).
#
# Usage examples:
#   python evaluate.py --ckpt results/distilbert_full --split test
#   python evaluate.py --ckpt results/distilbert_head --split validation
#
# The script loads the trained model and tokenizer, prepares
# the dataset, and computes evaluation metrics (accuracy, F1).
# ============================================================

# import and setup
import argparse, os, json, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from torch import nn
from utils import load_tweeteval_sentiment, HFEncodedDataset, eval_loop

# parse command-line arguments
def parse():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", choices=["train","validation","test"], default="test")
    return ap.parse_args()

# main evaluation function
def main():
    args = parse()
    # load tokenizer from the checkpoint directory
    tok = AutoTokenizer.from_pretrained(args.ckpt, use_fast=True)
    # Load label names (stored during training
    labels = json.load(open(os.path.join(args.ckpt,"labels.json")))
    # load dataset
    ds = load_tweeteval_sentiment()

    # function to tokenize a given dataset split
    def enc_split(split):
        enc = tok(list(ds[split]["text"]), truncation=True, padding=True, max_length=128)
        y = ds[split]["label"]
        return HFEncodedDataset(enc, y)
    dl = DataLoader(enc_split(args.split), batch_size=32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load model
    model=None
    try:
        model = AutoModelForSequenceClassification.from_pretrained(args.ckpt).to(device)
    except Exception:
        base = AutoModel.from_pretrained(args.ckpt).to(device)
        hidden = base.config.hidden_size if hasattr(base.config,"hidden_size") else base.config.d_model

        # Define classification head (same in train_head.py)
        class Head(nn.Module):
            def __init__(self): super().__init__(); self.drop=nn.Dropout(0.2); self.fc=nn.Linear(hidden,len(labels))
            def forward(self,x): return self.fc(self.drop(x))
        
        # load trained head weights
        head = Head().to(device)
        head.load_state_dict(torch.load(os.path.join(args.ckpt,"head.pt"), map_location=device))
        
        # helper: pooled representation (pooling if no pooler output)
        @torch.no_grad()
        def get_pooled(outs):
            if hasattr(outs,"pooler_output") and outs.pooler_output is not None:
                return outs.pooler_output
            return outs.last_hidden_state.mean(dim=1)
        
        # wrap base model + head for unified forward interface
        class Wrapped(nn.Module):
            def __init__(self, base, head): super().__init__(); self.base=base; self.head=head
            def forward(self, input_ids=None, attention_mask=None, labels=None):
                outs = self.base(input_ids=input_ids, attention_mask=attention_mask)
                pooled = get_pooled(outs); logits = self.head(pooled)
                return type("obj", (), {"logits": logits})
        model = Wrapped(base, head).to(device)
    
    # evaluation model
    metrics = eval_loop(model, dl, device)
    print(metrics)
    # save to json file
    json.dump(metrics, open(os.path.join(args.ckpt, f"{args.split}_metrics.json"),"w"), indent=2)

if __name__ == "__main__":
    main()
