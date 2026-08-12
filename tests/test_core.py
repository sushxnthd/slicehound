import numpy as np
import pytest
from slicehound import discover_slices


def test_finds_single_feature_hotspot():
    rng=np.random.default_rng(1); X=rng.random((1000,2)); e=rng.random(1000)<.05; e[X[:,0]>.8]=True
    r=discover_slices(X,e,["x","y"],max_depth=1,min_support=.05)
    assert r[0].predicates[0].feature==0
    assert r[0].lift > 2


def test_finds_conjunction():
    rng=np.random.default_rng(3); X=rng.random((1600,3)); e=rng.random(1600)<.08; hotspot=(X[:,0]>.6)&(X[:,1]<=.4); e[hotspot]=rng.random(hotspot.sum())<.9
    r=discover_slices(X,e,["a","b","c"],max_depth=2,min_support=.04)
    assert any(len(s.predicates)==2 and {p.feature for p in s.predicates}=={0,1} and s.lift>2 for s in r[:15])


def test_min_support_respected():
    rng=np.random.default_rng(4); X=rng.random((200,2)); e=np.zeros(200,bool); e[:3]=True
    r=discover_slices(X,e,min_support=40)
    assert all(s.support>=40 for s in r)


def test_bad_shapes_rejected():
    with pytest.raises(ValueError): discover_slices(np.zeros((3,2)),np.zeros(2))
