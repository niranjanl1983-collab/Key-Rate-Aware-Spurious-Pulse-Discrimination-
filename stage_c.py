"""Stage C: sensitivity of performance to detector parameters (dark-count rate,
afterpulse probability, timing jitter), using the optimised window + hold-off rule."""
import json, time
import numpy as np
from multiprocessing import Pool
from spad import DEFAULT, simulate, features
from mdi import key_rate
import stage_b as B

QS = np.logspace(-6, np.log10(3e-2), 8)
HS = np.array([0.2, 1, 5, 20]) * 1e-6
WS = B.W; H2S = np.array([0, 1, 5, 20]) * 1e-6


def table(p, seed=3):
    tab = np.zeros((len(HS), len(QS), len(WS) * len(H2S), 2))
    for hi, h in enumerate(HS):
        for qi, q in enumerate(QS):
            s = simulate(q, h, p=p, n_target=100_000, rng=np.random.default_rng(seed))
            F = features(s); sig = s["lab"] == 0; off = np.abs(F[:, 0]); dt = 10 ** F[:, 1]
            k = 0
            for w in WS:
                for x in H2S:
                    acc = (off <= w * 500) & (dt >= x)
                    tab[hi, qi, k] = (np.sum(acc & sig) / max(s["n_ph"], 1), np.sum(acc & ~sig) / s["n_slots"]); k += 1
    return tab


def interp(tab, q):
    lq = np.log10(np.clip(q, QS[0], QS[-1])); lQ = np.log10(QS)
    i = np.clip(np.searchsorted(lQ, lq) - 1, 0, len(QS) - 2); w = (lq - lQ[i]) / (lQ[i + 1] - lQ[i])
    a = np.log10(np.maximum(tab[:, i], 1e-14)); b = np.log10(np.maximum(tab[:, i + 1], 1e-14))
    return 10 ** ((1 - w) * a + w * b)


def perf(p):
    tab = table(p)
    eta_d = p["eta_d"]
    res = {}
    maxL = 0
    for L in np.arange(0, 301, 5):
        best = 0.0
        for mu in B.MUS:
            eta_ch = 10 ** (-0.2 * (L / 2) / 10)
            q = eta_d * 2 * eta_ch / 4 * (B.PMU * mu + B.PNU * B.NU)
            v = interp(tab, q).reshape(-1, 2)
            ri = B.r_inf(L, mu, eta_d, v[:, 0], v[:, 1])
            j = int(np.argmax(ri))
            if ri[j] <= 0:
                continue
            e, pd = eta_d * v[j, 0], v[j, 1]
            R, _, _ = key_rate(L, lambda *a: (e, pd), mu, B.NU, mode="lp")
            best = max(best, R)
        res[int(L)] = best
        if best > 0:
            maxL = L
        elif L > maxL + 10 and L > 30:
            break
    return dict(maxL=float(maxL), R50=res.get(50, 0.0), R100=res.get(100, 0.0))


def job(a):
    kind, val, val2 = a
    p = dict(DEFAULT)
    if kind == "dcr_ap":
        p["dcr"] = val; p["p_ap0"] = val2
    elif kind == "jitter":
        p["jit_sigma"] = val; p["tail_tau"] = val2
    r = perf(p); r.update(kind=kind, v1=val, v2=val2)
    return r


if __name__ == "__main__":
    t0 = time.time()
    jobs = [("dcr_ap", d, a) for d in (300, 1000, 3000, 10000, 30000) for a in (0.1, 0.3, 0.6, 0.9)]
    jobs += [("jitter", s * 1e-12, t * 1e-12) for s, t in ((20, 100), (50, 300), (100, 500), (200, 800))]
    with Pool(2) as pool:
        out = pool.map(job, jobs, chunksize=1)
    json.dump(out, open("sensitivity.json", "w"), indent=1)
    print("done", round(time.time() - t0))
