import argparse, json
import numpy as np
from .core import discover_slices

def main():
    ap=argparse.ArgumentParser(description="Discover compact high-error slices")
    ap.add_argument("npz",help="NPZ with X and errors"); ap.add_argument("--depth",type=int,default=2); ap.add_argument("--top",type=int,default=10); ap.add_argument("--min-support",type=float,default=.05)
    args=ap.parse_args(); d=np.load(args.npz,allow_pickle=False); names=[str(x) for x in d["feature_names"]] if "feature_names" in d else None
    out=discover_slices(d["X"],d["errors"],names,max_depth=args.depth,min_support=args.min_support)
    print(json.dumps([{"rule":r.rule,"support":r.support,"error_rate":r.error_rate,"lift":r.lift,"score":r.score} for r in out[:args.top]],indent=2))
if __name__=="__main__": main()
