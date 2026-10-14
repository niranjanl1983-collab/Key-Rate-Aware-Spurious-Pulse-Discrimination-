"""Stage B: operating points of spurious-pulse discriminators and key-rate
optimisation versus distance."""
import pickle, json, time
import numpy as np
from mdi import gains, y11_exact, h2, key_rate
from spad import DEFAULT

A = pickle.load(open("stage_a.pkl", "rb"))
db, clf, Q, HOLDS = A["db"], A["clf"], A["Q"], A["HOLDS"]
W = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])  # ns (full width)
H2 = np.array([0, 0.5, 1, 2, 5, 10, 20, 50]) * 1e-6
TH = np.unique(np.concatenate([[0.0], np.logspace(-4, -1, 15), 1 - np.logspace(-5, np.log10(0.9), 70)]))
NU, PMU, PNU = 0.05, 0.5, 0.3


def op_points(r, h=None, fam="win"):
    F, lab = r["F"], r["lab"]
    sig = lab == 0
    off = np.abs(F[:, 0]); dt = 10 ** F[:, 1]
    res = []
    if fam == "raw":
        acc = np.ones(len(lab), bool); res.append(acc)
    elif fam == "win":
        for w in W:
            res.append(off <= w * 500)
    elif fam == "winho":
        for w in W:
            for x in H2:
                res.append((off <= w * 500) & (dt >= x))
    elif fam == "gbm":
        s = clf[h].predict_proba(F)[:, 1]
        for t in TH:
            res.append(s >= t)
    out = []
    for acc in res:
        out.append((np.sum(acc & sig) / max(r["n_ph"], 1), np.sum(acc & ~sig) / r["n_slots"]))
    return np.array(out)  # (n_params, 2)


def build_tables():
    T = {}
    for fam in ("raw", "win", "winho", "gbm"):
        arr = []
        for h in HOLDS:
            arr.append([op_points(db[("spad", h, q, 2)], h, fam) for q in Q])
        T[fam] = np.array(arr)  # (hold, q, param, 2)
    T["snspd"] = np.array([[op_points(db[("snspd", 50e-9, q, 2)], None, "win") for q in Q]])
    return T


def interp_q(tab, q):
    """tab: (..., nq, P, 2) -> values at q via log-log interpolation."""
    lq = np.log10(np.clip(q, Q[0], Q[-1]))
    i = np.clip(np.searchsorted(np.log10(Q), lq) - 1, 0, len(Q) - 2)
    w = (lq - np.log10(Q[i])) / (np.log10(Q[i + 1]) - np.log10(Q[i]))
    a = np.log10(np.maximum(tab[..., i, :, :], 1e-14)); b = np.log10(np.maximum(tab[..., i + 1, :, :], 1e-14))
    return 10 ** ((1 - w) * a + w * b)


def r_inf(L, mu, eta_d, rho, pd, ed=0.015, f=1.16):
    eta_ch = 10 ** (-0.2 * (L / 2) / 10)
    e = eta_ch * eta_d * rho
    QZ, EQZ, _, _ = gains(mu, mu, e, e, pd, ed)
    Y11, e11 = y11_exact(e, e, pd, ed)
    return np.maximum(mu * mu * np.exp(-2 * mu) * Y11 * (1 - h2(e11)) - QZ * f * h2(EQZ / QZ), 0)


CONFIGS = {
    "Noise-free detector": dict(kind="ideal"),
    "Raw SPAD (no filtering)": dict(kind="spad", fam="raw", holds=[10e-6]),
    "Fixed 1-ns window": dict(kind="spad", fam="win", holds=[10e-6], params=[list(W).index(1.0)]),
    "Optimised window + hold-off": dict(kind="spad", fam="winho"),
    "KR-SPD (proposed)": dict(kind="spad", fam="gbm"),
    "SNSPD (optimised window)": dict(kind="snspd", fam="win"),
}
MUS = np.round(np.arange(0.05, 0.81, 0.05), 3)
LS = np.arange(0, 451, 5)


def best_detector(cfg, T, L, mu):
    """Return (rate_inf, eta_eff, pd_eff, desc) of the best detector setting."""
    if cfg["kind"] == "ideal":
        return r_inf(L, mu, DEFAULT["eta_d"], 1.0, 0.0), DEFAULT["eta_d"], 0.0, "-"
    snspd = cfg["kind"] == "snspd"
    eta_d = 0.80 if snspd else DEFAULT["eta_d"]
    eta_ch = 10 ** (-0.2 * (L / 2) / 10)
    q = eta_d * (2 * eta_ch) / 4 * (PMU * mu + PNU * NU)
    tab = T["snspd"] if snspd else T[cfg["fam"]]
    holds = [50e-9] if snspd else HOLDS
    vals = interp_q(tab, q)  # (hold, P, 2)
    best = (-1, None)
    for hi, h in enumerate(holds):
        if (not snspd) and "holds" in cfg and not np.any(np.isclose(h, cfg["holds"])):
            continue
        pidx = cfg.get("params", range(vals.shape[1]))
        for pi in pidx:
            rho, pd = vals[hi, pi]
            r = r_inf(L, mu, eta_d, rho, pd)
            if r > best[0]:
                best = (r, (eta_d * rho, pd, h, pi))
    r, (e, pd, h, pi) = best
    return r, e, pd, (h, pi)


def run_configs(T):
    out = {}
    for name, cfg in CONFIGS.items():
        t0 = time.time()
        rows = []
        for L in LS:
            best = None
            for mu in MUS:
                ri, e, pd, desc = best_detector(cfg, T, L, mu)
                if ri <= 0:
                    continue
                det = lambda L_, m_, n_: (e, pd)
                R, EZ, e11 = key_rate(L, det, mu, NU, mode="lp")
                if best is None or R > best["R"]:
                    best = dict(L=float(L), R=float(R), R_inf=float(ri), mu=float(mu), eta=float(e), pd=float(pd),
                                EZ=float(EZ), e11=float(e11) if e11 == e11 else None, desc=str(desc))
            if best is None or best["R"] <= 0:
                rows.append(dict(L=float(L), R=0.0))
                if L > 50 and all(r["R"] == 0 for r in rows[-3:]):
                    break
            else:
                rows.append(best)
        out[name] = rows
        print(name, "max L", max([r["L"] for r in rows if r["R"] > 0] or [0]), round(time.time() - t0), "s", flush=True)
    return out


def finite(T, name="KR-SPD (proposed)"):
    cfg = CONFIGS[name]
    res = {}
    for N in (1e11, 1e12, 1e13, 1e14):
        rows = []
        for L in LS:
            best = 0.0
            for mu in MUS:
                ri, e, pd, desc = best_detector(cfg, T, L, mu)
                if ri <= 0:
                    continue
                det = lambda L_, m_, n_: (e, pd)
                R, _, _ = key_rate(L, det, mu, NU, mode="lp", fluct=(N, 0.8, (0.5, 0.3, 0.2), 5.0))
                best = max(best, R)
            rows.append(dict(L=float(L), R=float(best)))
            if L > 30 and best == 0 and rows[-2]["R"] == 0:
                break
        res["%.0e" % N] = rows
        print("finite", N, max([r["L"] for r in rows if r["R"] > 0] or [0]), flush=True)
    return res


if __name__ == "__main__":
    t0 = time.time()
    import os
    if os.path.exists("tables.pkl"):
        T = pickle.load(open("tables.pkl", "rb"))
    else:
        T = build_tables()
        pickle.dump(T, open("tables.pkl", "wb"))
    print("tables", round(time.time() - t0), "s", flush=True)
    out = run_configs(T)
    json.dump(out, open("keyrates.json", "w"), indent=1)
    fin = finite(T)
    json.dump(fin, open("finite.json", "w"), indent=1)
    print("done", round(time.time() - t0))
