import argparse, json, os, random, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from torch.utils.data import DataLoader
from torch import nn

def load_labels(ckpt):
    with open(os.path.join(ckpt, "labels.json")) as f:
        return json.load(f)

def build_model(ckpt, num_labels):
    """
    Works with BOTH:
    - full FT checkpoints (AutoModelForSequenceClassification)
    - head checkpoints (base encoder + head.pt)
    """
    try:
        m = AutoModelForSequenceClassification.from_pretrained(ckpt)
        tok = AutoTokenizer.from_pretrained(ckpt, use_fast=True)
        return m, tok, "full"
    except Exception:
        tok = AutoTokenizer.from_pretrained(ckpt, use_fast=True)
        base = AutoModel.from_pretrained(ckpt)

        hidden = base.config.hidden_size if hasattr(base.config,"hidden_size") else base.config.d_model
        class Head(nn.Module):
            def __init__(self):
                super().__init__(); self.drop = nn.Dropout(0.2); self.fc = nn.Linear(hidden, num_labels)
            def forward(self, x): return self.fc(self.drop(x))

        head = Head()
        head.load_state_dict(torch.load(os.path.join(ckpt, "head.pt"), map_location="cpu"))

        class Wrapped(nn.Module):
            def __init__(self, base, head):
                super().__init__(); self.base = base; self.head = head
            def forward(self, input_ids=None, attention_mask=None, labels=None):
                outs = self.base(input_ids=input_ids, attention_mask=attention_mask)
                pooled = outs.pooler_output if getattr(outs, "pooler_output", None) is not None else outs.last_hidden_state.mean(dim=1)
                logits = self.head(pooled)
                return type("obj", (), {"logits": logits})

        return Wrapped(base, head), tok, "head"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Checkpoint folder (e.g., results/distilbert_full)")
    ap.add_argument("--split", default="test", choices=["train","validation","test"])
    ap.add_argument("--k", type=int, default=10, help="How many misclassified examples to print")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--true_label", type=str, default=None, help="Only show errors with this TRUE label (optional)")
    ap.add_argument("--pred_label", type=str, default=None, help="Only show errors with this PRED label (optional)")
    args = ap.parse_args()

    random.seed(args.seed)

    labels = load_labels(args.ckpt)
    num_labels = len(labels)

    model, tok, kind = build_model(args.ckpt, num_labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device); model.eval()

    ds = load_dataset("tweet_eval","sentiment")[args.split]
    texts = list(ds["text"])
    y_true = list(ds["label"])

    # batch predict for speed
    batch_size = 64
    mis = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = tok(batch_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k,v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            preds = logits.argmax(dim=-1).cpu().tolist()
        for j, t in enumerate(batch_texts):
            true_id = y_true[i+j]; pred_id = preds[j]
            if true_id != pred_id:
                true_name = labels[true_id]; pred_name = labels[pred_id]
                mis.append((t, true_name, pred_name))

    # optional filtering
    if args.true_label:
        mis = [m for m in mis if m[1].lower() == args.true_label.lower()]
    if args.pred_label:
        mis = [m for m in mis if m[2].lower() == args.pred_label.lower()]

    random.shuffle(mis)
    take = mis[:args.k]

    print(f"Checkpoint: {args.ckpt}  (type={kind})")
    print(f"Split: {args.split} | Total misclassifications found: {len(mis)} | Showing: {len(take)}")
    print("-"*80)
    for idx, (text, tname, pname) in enumerate(take, 1):
        print(f"[{idx}] TRUE={tname}  PRED={pname}")
        print(f"TEXT: {text}")
        print("-"*80)

if __name__ == "__main__":
    main()