# Assignment 3 — Simple Training Loop with Pretrained Transformers

Two strategies on TweetEval sentiment:
1) Full fine-tuning (update all params)
2) Linear probe (freeze transformer, train small head)

## PowerShell quickstart
```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

python train_full.py --model_name distilbert-base-uncased --out results\distilbert_full
python train_head.py --model_name distilbert-base-uncased --out results\distilbert_head

python evaluate.py --ckpt results\distilbert_full --split test
python evaluate.py --ckpt results\distilbert_head --split test

python scripts\plot_compare.py --roots results\distilbert_full results\distilbert_head --out results\compare.png
```
