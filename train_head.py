# ============================================================
# This script trains only a small classification head on top of a frozen
# pretrained DistilBERT encoder. The goal is to compare performance and
# efficiency against full fine-tuning.
#
# Usage:
#     python train_head.py --model_name distilbert-base-uncased --out results/distilbert_head
# ============================================================

# imports and setup
import argparse, os, json, torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch import nn
from tqdm import tqdm
from utils import load_tweeteval_sentiment, HFEncodedDataset, eval_loop

# Parse command-line arguments
def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="distilbert-base-uncased") # base model
    ap.add_argument("--out", default="results/distilbert_head") # output folder
    ap.add_argument("--epochs", type=int, default=3) # training epochs
    ap.add_argument("--lr", type=float, default=1e-3) # learning rate
    ap.add_argument("--batch", type=int, default=32) # batch size
    ap.add_argument("--max_len", type=int, default=128) # max token length
    return ap.parse_args()

# Define classification head
class Head(nn.Module):
    def __init__(self, hidden, num_labels):
        super().__init__(); self.drop=nn.Dropout(0.2); self.fc=nn.Linear(hidden, num_labels)
    def forward(self, x): return self.fc(self.drop(x))

# Main training function
def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True)
    
    # load dataset
    ds = load_tweeteval_sentiment()
    
    # save label names for references
    labels = ds["train"].features["label"].names
    json.dump(labels, open(os.path.join(args.out,"labels.json"),"w"), indent=2)

    # Initialize tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    # Helper function to tokenize text data
    def enc_split(split):
        enc = tok(list(ds[split]["text"]), truncation=True, padding=True, max_length=args.max_len)
        y = ds[split]["label"]
        return HFEncodedDataset(enc, y)
    
    # Encode train / validation / test splits
    train_ds = enc_split("train"); val_ds = enc_split("validation"); test_ds = enc_split("test")

    # Model and optimizer setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load base (pretrained) transformer encoder
    base = AutoModel.from_pretrained(args.model_name).to(device)
    # Freeze all encoder parameters (we only train the head)
    for p in base.parameters(): p.requires_grad=False

    # Get hidden dimension (depends on model type)
    hidden = base.config.hidden_size if hasattr(base.config,"hidden_size") else base.config.d_model
    
    # Initialize classification head and loss/optimizer
    head = Head(hidden, len(labels)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = AdamW(head.parameters(), lr=args.lr)

    # Helper to get a pooled representation (mean pooling if no pooler output)
    def get_pooled(outs):
        if hasattr(outs,"pooler_output") and outs.pooler_output is not None:
            return outs.pooler_output
        return outs.last_hidden_state.mean(dim=1)

    # Prepare DataLoaders
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=args.batch)
    test_dl  = DataLoader(test_ds, batch_size=args.batch)

    # Training Loop
    best = -1.0
    for e in range(args.epochs):
        head.train()
        for batch in tqdm(train_dl, desc=f"epoch {e+1}/{args.epochs}"):
            batch = {k:v.to(device) for k,v in batch.items()}
            # Forward pass through frozen base + classification head
            outs = base(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            pooled = get_pooled(outs)
            logits = head(pooled)
            # Compute loss and update head parameters only
            loss = loss_fn(logits, batch["labels"])
            opt.zero_grad(); loss.backward(); opt.step()
        # wrap for eval
        class Wrapped(nn.Module):
            def __init__(self, base, head): super().__init__(); self.base=base; self.head=head
            def forward(self, input_ids=None, attention_mask=None, labels=None):
                outs = self.base(input_ids=input_ids, attention_mask=attention_mask)
                pooled = get_pooled(outs); logits = self.head(pooled)
                return type("obj", (), {"logits": logits})
        wrapped = Wrapped(base, head).to(device)
        # Evaluate on validation set
        metrics = eval_loop(wrapped, val_dl, device)
        json.dump(metrics, open(os.path.join(args.out,"eval_metrics.json"),"w"), indent=2)
        
        # Save best model (based on macro F1)
        if metrics["f1_macro"]>best:
            best=metrics["f1_macro"]
            base.save_pretrained(args.out); tok.save_pretrained(args.out)
            torch.save(head.state_dict(), os.path.join(args.out,"head.pt"))

    # Final evaluation on the test
    wrapped = Wrapped(base, head).to(device)
    test_metrics = eval_loop(wrapped, test_dl, device)
    json.dump(test_metrics, open(os.path.join(args.out,"test_metrics.json"),"w"), indent=2)
    print("Done", args.out, test_metrics)

if __name__ == "__main__":
    main()
