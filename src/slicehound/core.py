from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Predicate:
    feature: int
    name: str
    op: str
    threshold: float

    def apply(self, X: np.ndarray) -> np.ndarray:
        col=np.asarray(X)[:,self.feature]
        return col <= self.threshold if self.op=="<=" else col > self.threshold

    def __str__(self) -> str:
        return f"{self.name} {self.op} {self.threshold:.3g}"


@dataclass(frozen=True)
class SliceResult:
    predicates: tuple[Predicate,...]
    support: int
    support_fraction: float
    error_rate: float
    lift: float
    excess_error: float
    score: float

    @property
    def rule(self) -> str:
        return " AND ".join(str(p) for p in self.predicates)


def _metrics(mask: np.ndarray, errors: np.ndarray, predicates: tuple[Predicate,...], global_rate: float) -> SliceResult | None:
    n=int(mask.sum())
    if n==0: return None
    rate=float(np.mean(errors[mask])); excess=rate-global_rate; lift=rate/max(global_rate,1e-12)
    score=excess*np.sqrt(n)
    return SliceResult(predicates,n,n/len(mask),rate,lift,excess,float(score))


def _predicate_pool(X: np.ndarray, names: Sequence[str], quantiles: Sequence[float]) -> list[Predicate]:
    pool=[]
    for j,name in enumerate(names):
        vals=np.unique(np.quantile(X[:,j],quantiles))
        for t in vals:
            pool.append(Predicate(j,name,"<=",float(t))); pool.append(Predicate(j,name,">",float(t)))
    return pool


def discover_slices(X: np.ndarray, errors: np.ndarray, feature_names: Sequence[str] | None = None, *, max_depth: int = 2, beam_width: int = 24, min_support: int | float = 0.05, quantiles: Sequence[float] = (.2,.4,.6,.8)) -> list[SliceResult]:
    X=np.asarray(X,dtype=float); e=np.asarray(errors).astype(bool).reshape(-1)
    if X.ndim!=2 or len(X)!=len(e): raise ValueError("X must be 2D and match errors")
    if not 1 <= max_depth <= 4: raise ValueError("max_depth must be between 1 and 4")
    names=list(feature_names) if feature_names is not None else [f"x{i}" for i in range(X.shape[1])]
    if len(names)!=X.shape[1]: raise ValueError("feature_names length does not match X")
    min_n=max(1, int(np.ceil(min_support*len(X)))) if isinstance(min_support,float) and min_support < 1 else int(min_support)
    global_rate=float(e.mean()); pool=_predicate_pool(X,names,quantiles); seen=set(); results=[]

    beam=[(tuple(), np.ones(len(X),dtype=bool))]
    for _depth in range(1,max_depth+1):
        candidates=[]
        for preds,base_mask in beam:
            used={(p.feature,p.op,p.threshold) for p in preds}
            for pred in pool:
                key=(pred.feature,pred.op,pred.threshold)
                if key in used: continue
                if any(p.feature==pred.feature and p.op!=pred.op for p in preds):
                    continue
                mask=base_mask & pred.apply(X)
                if mask.sum() < min_n: continue
                mask_key=np.packbits(mask).tobytes()
                if mask_key in seen: continue
                seen.add(mask_key)
                new_preds=preds+(pred,)
                m=_metrics(mask,e,new_preds,global_rate)
                if m is None or m.excess_error <= 0: continue
                results.append(m); candidates.append((m.score,new_preds,mask))
        candidates.sort(key=lambda x:x[0],reverse=True)
        beam=[(preds,mask) for _,preds,mask in candidates[:beam_width]]
        if not beam: break
    results.sort(key=lambda r:(r.score,r.lift),reverse=True)
    return results
