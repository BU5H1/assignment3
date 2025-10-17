import argparse, json, os, pandas as pd, matplotlib.pyplot as plt
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--roots', nargs='+', required=True); ap.add_argument('--out', default='results/compare.png'); args=ap.parse_args()
    rows=[]
    for r in args.roots:
        p=os.path.join(r,'test_metrics.json')
        if os.path.exists(p):
            m=json.load(open(p)); m['run']=os.path.basename(r); rows.append(m)
    if not rows: print('No metrics'); return
    import pandas as pd; df=pd.DataFrame(rows); print(df[['run','accuracy','f1_macro','f1_weighted']])
    import matplotlib.pyplot as plt
    fig=plt.figure(); ax=fig.gca(); idx=range(len(df))
    ax.bar(list(idx), list(df['accuracy']), label='Accuracy'); ax.bar(list(idx), list(df['f1_macro']), alpha=0.6, label='F1 Macro')
    ax.set_xticks(list(idx)); ax.set_xticklabels(list(df['run']), rotation=15, ha='right'); ax.legend(); plt.tight_layout(); fig.savefig(args.out, dpi=180); print('Saved:', args.out)
if __name__=='__main__': main()
