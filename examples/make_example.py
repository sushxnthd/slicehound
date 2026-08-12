from pathlib import Path
import numpy as np


def main():
    rng = np.random.default_rng(11)
    X = rng.random((500, 3))
    errors = rng.random(500) < .06
    hotspot = (X[:, 0] > .6) & (X[:, 1] <= .4)
    errors[hotspot] = rng.random(hotspot.sum()) < .85
    out = Path(__file__).with_name("failures.npz")
    np.savez(out, X=X, errors=errors, feature_names=np.array(["latency", "confidence", "noise"]))
    print(out)


if __name__ == "__main__":
    main()
