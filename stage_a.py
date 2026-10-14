"""Stage A: detector-level Monte-Carlo database + spurious-pulse classifiers.
For each hold-off value and incident-rate value, two independent simulations
(train / test). A gradient-boosting discriminator is trained per hold-off on the
pooled training simulations and evaluated on the test simulations."""
import pickle, sys, time
import numpy as np
from multiprocessing import Pool
from sklearn.ensemble import HistGradientBoostingClassifier
from spad import DEFAULT, simulate, features

Q = np.logspace(-6, np.log10(3e-2), 14)
HOLDS = np.array([0.2, 0.5, 1, 2, 5, 10, 20]) * 1e-6
SNSPD = dict(DEFAULT, eta_d=0.80, dcr=100.0, jit_sigma=20e-12, tail_frac=0.0, p_ap0=0.0)


def run(args):
    q, h, seed, det = args
    p = SNSPD if det == "snspd" else DEFAULT
    s = simulate(q, h, p=p, n_target=200_000, rng=np.random.default_rng(seed))
    f = features(s)
    return dict(q=q, hold=h, seed=seed, det=det, F=f.astype(np.float32), lab=s["lab"],
                n_ph=s["n_ph"], n_slots=s["n_slots"])


if __name__ == "__main__":
    jobs = [(q, h, seed, "spad") for h in HOLDS for q in Q for seed in (1, 2)]
    jobs += [(q, 50e-9, seed, "snspd") for q in Q for seed in (1, 2)]
    t0 = time.time()
    with Pool(2) as pool:
        res = pool.map(run, jobs, chunksize=2)
    print("sims", len(res), round(time.time() - t0), "s", flush=True)
    db = {}
    for r in res:
        db[(r["det"], r["hold"], r["q"], r["seed"])] = r
    # classifiers: one per hold-off, trained on seed-1 data, balanced sample per q
    clf = {}
    rng = np.random.default_rng(0)
    for h in HOLDS:
        X, y = [], []
        for q in Q:
            r = db[("spad", h, q, 1)]
            n = len(r["lab"]); k = min(n, 40_000)
            idx = rng.choice(n, k, replace=False)
            X.append(r["F"][idx]); y.append((r["lab"][idx] == 0).astype(int))
        X = np.vstack(X); y = np.concatenate(y)
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                           l2_regularization=1.0, early_stopping=True,
                                           validation_fraction=0.15, random_state=0)
        m.fit(X, y)
        clf[h] = m
        print("hold %.1fus trained on %d, iters %d" % (h * 1e6, len(y), m.n_iter_), flush=True)
    pickle.dump(dict(db=db, clf=clf, Q=Q, HOLDS=HOLDS), open("stage_a.pkl", "wb"))
    print("done", round(time.time() - t0), "s")
