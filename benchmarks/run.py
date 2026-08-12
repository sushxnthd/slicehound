from pathlib import Path
import json, numpy as np
from slicehound import discover_slices

def main():
    rng=np.random.default_rng(77); n=3200; X=rng.random((n,4)); errors=rng.random(n)<.07
    hotspot=(X[:,0]>.6)&(X[:,1]<=.4); errors[hotspot]=rng.random(hotspot.sum())<.86
    out=discover_slices(X,errors,["latency","confidence","length","noise"],max_depth=2,beam_width=30,min_support=.04)
    best_pair=next((s for s in out if len(s.predicates)==2 and {p.feature for p in s.predicates}=={0,1}),None)
    assert best_pair is not None and best_pair.lift>2.5
    result={"n":n,"global_error_rate":round(float(errors.mean()),4),"best_rule":out[0].rule,"best_lift":round(out[0].lift,4),"target_pair_rule":best_pair.rule,"target_pair_lift":round(best_pair.lift,4),"target_pair_support":best_pair.support}
    Path(__file__).with_name("results.json").write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
