# ============================================================
# This script fine-tunes all parameters of a pretrained DistilBERT model
# for sentiment classification (negative / neutral / positive) using the
# TweetEval sentiment dataset.
# 
# Usage:
#     python train_full.py --model_name distilbert-base-uncased --out results/distilbert_full
# ============================================================

# imports and setup
import argparse, os, json, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from utils import load_tweeteval_sentiment, HFEncodedDataset, eval_loop

# Arguments parser
def parse():
    # define command-line arguments for training configuration
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="distilbert-base-uncased") # pretrained model name
    ap.add_argument("--out", default="results/distilbert_full") # output folder for model and metrics
    ap.add_argument("--epochs", type=int, default=3) # number of training epochs
    ap.add_argument("--lr", type=float, default=2e-5) # learning rate
    ap.add_argument("--batch", type=int, default=16) # batch size
    ap.add_argument("--max_len", type=int, default=128) # max token length
    return ap.parse_args()

# Main traing loop
def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True) # Create output directory if it doesn’t exist

    # Load the TweetEval sentiment dataset
    ds = load_tweeteval_sentiment()

    # Extract label names (e.g., ["negative", "neutral", "positive"])
    labels = ds["train"].features["label"].names
    json.dump(labels, open(os.path.join(args.out,"labels.json"),"w"), indent=2)

    # Initialize the tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # Encode function to tokenize each data split
    def enc_split(split):
        # Apply tokenizer to the text and create a dataset with encoded inputs + labels
        enc = tok(list(ds[split]["text"]), truncation=True, padding=True, max_length=args.max_len)
        y = ds[split]["label"]
        return HFEncodedDataset(enc, y)
    
    # Tokenize training, validation, and test sets
    train_ds = enc_split("train")
    val_ds   = enc_split("validation")
    test_ds  = enc_split("test")
    
    # Model, optimizer and scheduler setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(labels)).to(device)

    # Create DataLoaders for efficient batch processing
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=args.batch)
    test_dl  = DataLoader(test_ds, batch_size=args.batch)

    # Define optimizer and linear learning rate scheduler with warm-up
    opt = AdamW(model.parameters(), lr=args.lr)
    steps = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06*steps), steps)

    # Training loop
    best = -1.0
    for e in range(args.epochs):
        model.train()
        for batch in tqdm(train_dl, desc=f"epoch {e+1}/{args.epochs}"):
            batch = {k:v.to(device) for k,v in batch.items()}
            # Forward pass and compute loss
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            loss = out.loss
            # Backpropagation and optimization
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        
        # Evaluate after each epoch on validation set
        metrics = eval_loop(model, val_dl, device)
        json.dump(metrics, open(os.path.join(args.out,"eval_metrics.json"),"w"), indent=2)
        
        # Save model if it achieves best macro-F1 so far
        if metrics["f1_macro"]>best:
            best = metrics["f1_macro"]
            model.save_pretrained(args.out); tok.save_pretrained(args.out)
    
    # Final evaluation on test set
    test_metrics = eval_loop(model, test_dl, device)
    json.dump(test_metrics, open(os.path.join(args.out,"test_metrics.json"),"w"), indent=2)
    print("Done", args.out, test_metrics)

if __name__ == "__main__":
    main()
