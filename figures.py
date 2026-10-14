"""Figures and summary numbers for the paper."""
import json, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mdi import gains, mc_bsm
from spad import DEFAULT, simulate, features, p_afterpulse
import stage_b as B

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 300, "savefig.bbox": "tight"})
C = {"Noise-free detector": "#555555", "Raw SPAD (no filtering)": "#d62728", "Fixed 1-ns window": "#ff7f0e",
     "Optimised window + hold-off": "#1f77b4", "KR-SPD (proposed)": "#2ca02c", "SNSPD (optimised window)": "#9467bd"}
LS_ = {"Noise-free detector": "--", "SNSPD (optimised window)": ":"}
S = {}

# ---------------------------------------------------------------- validation
rng = np.random.default_rng(7)
pd, ed = 1e-5, 0.0
Ls = np.arange(0, 61, 10)
val = {"L": Ls.tolist()}
for basis in "ZX":
    mcq, mce, aq, ae = [], [], [], []
    for L in Ls:
        e = 10 ** (-0.2 * L / 2 / 10) * 0.145
        q, er = mc_bsm(0.5, e, e, pd, basis, 6_000_000, rng)
        QZ, EQZ, QX, EQX = gains(0.5, 0.5, e, e, pd, ed)
        mcq.append(q); mce.append(er)
        aq.append(QZ if basis == "Z" else QX); ae.append(EQZ / QZ if basis == "Z" else EQX / QX)
    val[basis] = dict(mc_q=mcq, mc_e=mce, an_q=aq, an_e=ae)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
for b, mk in (("Z", "o"), ("X", "s")):
    ax[0].semilogy(Ls, val[b]["an_q"], "-", color="k" if b == "Z" else "#1f77b4", lw=1)
    ax[0].semilogy(Ls, val[b]["mc_q"], mk, color="k" if b == "Z" else "#1f77b4", ms=4, mfc="none", label=f"{b} basis")
    ax[1].plot(Ls, np.array(val[b]["an_e"]) * 100, "-", color="k" if b == "Z" else "#1f77b4", lw=1)
    ax[1].plot(Ls, np.array(val[b]["mc_e"]) * 100, mk, color="k" if b == "Z" else "#1f77b4", ms=4, mfc="none", label=f"{b} basis")
ax[0].set_xlabel("Total distance Alice–Bob (km)"); ax[0].set_ylabel("Gain per pulse pair"); ax[0].legend(frameon=False)
ax[1].set_xlabel("Total distance Alice–Bob (km)"); ax[1].set_ylabel("QBER (%)")
ax[0].set_title("(a) Gain: analytic (lines) vs Monte-Carlo (markers)", fontsize=8.5)
ax[1].set_title("(b) QBER: analytic (lines) vs Monte-Carlo (markers)", fontsize=8.5)
fig.tight_layout(); fig.savefig("figs/fig3_validation.png"); plt.close(fig)
S["validation"] = val

# ---------------------------------------------------------------- avalanche features
s = simulate(1e-4, 1e-6, n_target=300_000, rng=np.random.default_rng(11))
F = features(s); lab = s["lab"]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
names = ["Signal", "Dark count", "Afterpulse"]; cols = ["#2ca02c", "#7f7f7f", "#d62728"]
for k in range(3):
    ax[0].hist(F[lab == k, 0] / 1000, bins=np.linspace(-5, 5, 201), histtype="step", lw=1.1, color=cols[k], label=names[k], density=True)
    ax[1].hist(F[lab == k, 1], bins=np.linspace(-6.2, -1, 120), histtype="step", lw=1.1, color=cols[k], label=names[k], density=True)
ax[0].set_yscale("log"); ax[0].set_xlabel("Offset from slot centre (ns)"); ax[0].set_ylabel("Probability density")
ax[1].set_yscale("log"); ax[1].set_xlabel("log$_{10}$ time since previous avalanche (s)")
ax[0].legend(frameon=False, fontsize=7); ax[0].set_title("(a) Arrival-time offset", fontsize=8.5); ax[1].set_title("(b) Inter-avalanche interval", fontsize=8.5)
fig.tight_layout(); fig.savefig("figs/fig4_features.png"); plt.close(fig)
S["feat_counts"] = {names[k]: int((lab == k).sum()) for k in range(3)}

# ---------------------------------------------------------------- trade-off curves
T = pickle.load(open("tables.pkl", "rb"))
hi = 2  # hold-off 1 us
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
S["tradeoff"] = {}
for a, qi in zip(ax, (4, 8)):
    q = B.Q[qi]
    w = T["win"][hi, qi]; wh = T["winho"][hi, qi]; g = T["gbm"][hi, qi]

    def front(pts):
        o = np.argsort(pts[:, 0]); pts = pts[o]; out = []; best = np.inf
        for r, p in pts[::-1]:
            if p < best:
                out.append((r, p)); best = p
        return np.array(out[::-1])
    for pts, lab_, c in ((w, "Timing window", "#ff7f0e"), (wh, "Window + software hold-off", "#1f77b4"), (g, "KR-SPD discriminator", "#2ca02c")):
        f_ = front(pts[pts[:, 1] > 0])
        a.semilogy(f_[:, 0], f_[:, 1], "-", color=c, label=lab_, lw=1.3)
    a.set_xlim(0.5, 1.0); a.set_xlabel("Signal retention ρ"); a.set_ylabel("Accepted spurious clicks per slot")
    a.set_title(f"({'a' if qi == 4 else 'b'}) photon-detection probability q = {q:.1e}", fontsize=8.5)
    # matched-retention comparison
    rec = {}
    for target in (0.8, 0.9, 0.95):
        def at(pts):
            ok = pts[:, 0] >= target
            return float(pts[ok, 1].min()) if ok.any() else None
        rec[str(target)] = dict(win=at(w), winho=at(wh), gbm=at(g))
    S["tradeoff"]["%.1e" % q] = rec
ax[0].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig("figs/fig5_tradeoff.png"); plt.close(fig)

# ---------------------------------------------------------------- key rate vs distance
K = json.load(open("keyrates.json"))
fig, ax = plt.subplots(figsize=(5.6, 3.6))
for n, rows in K.items():
    L = [r["L"] for r in rows if r["R"] > 0]; R = [r["R"] for r in rows if r["R"] > 0]
    ax.semilogy(L, R, LS_.get(n, "-"), color=C[n], lw=2 if "proposed" in n else 1.2, label=n)
ax.set_xlabel("Total distance Alice–Bob (km)"); ax.set_ylabel("Secret key rate (bits per pulse)")
ax.set_ylim(1e-10, 5e-2); ax.legend(frameon=False, fontsize=7); ax.grid(alpha=.25, which="both")
fig.savefig("figs/fig6_keyrate.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.0, 3.2))
for n, rows in K.items():
    if n.startswith("Noise-free"):
        continue
    L = [r["L"] for r in rows if r["R"] > 0]; E = [r["EZ"] * 100 for r in rows if r["R"] > 0]
    ax.plot(L, E, LS_.get(n, "-"), color=C[n], lw=2 if "proposed" in n else 1.2, label=n)
ax.set_xlabel("Total distance Alice–Bob (km)"); ax.set_ylabel("Z-basis QBER $E_Z$ (%)")
ax.legend(frameon=False, fontsize=7); ax.grid(alpha=.25)
fig.savefig("figs/fig7_qber.png"); plt.close(fig)

# optimal settings vs distance
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
for n in ("Optimised window + hold-off", "KR-SPD (proposed)"):
    rows = [r for r in K[n] if r["R"] > 0]
    L = [r["L"] for r in rows]
    hold = [float(r["desc"].split("(")[2].split(")")[0]) * 1e6 for r in rows]
    ax[0].plot(L, hold, "-o", ms=2.5, color=C[n], label=n)
    ax[1].plot(L, [r["eta"] / 0.145 for r in rows], "-", color=C[n], label=n + ": ρ")
    ax[1].plot(L, [r["pd"] / 3e-5 for r in rows], "--", color=C[n], label=n + ": spurious / raw dark")
ax[0].set_yscale("log"); ax[0].set_xlabel("Total distance (km)"); ax[0].set_ylabel("Optimal hold-off (µs)")
ax[1].set_yscale("log"); ax[1].set_xlabel("Total distance (km)"); ax[1].set_ylabel("Relative value")
ax[0].set_title("(a) Key-rate-optimal hold-off", fontsize=8.5); ax[1].set_title("(b) Retention ρ and residual spurious rate", fontsize=8.5)
ax[0].legend(frameon=False, fontsize=6.5)
h_, l_ = ax[1].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=2, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.08))
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig("figs/fig8_settings.png"); plt.close(fig)

# finite size
FN = json.load(open("finite.json"))
fig, ax = plt.subplots(figsize=(5.0, 3.2))
rows = [r for r in K["KR-SPD (proposed)"] if r["R"] > 0]
ax.semilogy([r["L"] for r in rows], [r["R"] for r in rows], "-", color="k", label="Asymptotic")
for (n, rows_), c in zip(FN.items(), ("#c6dbef", "#6baed6", "#2171b5", "#08306b")):
    rr = [r for r in rows_ if r["R"] > 0]
    if rr:
        ax.semilogy([r["L"] for r in rr], [r["R"] for r in rr], "-", color=c, label=f"N = {n.replace('e+', '×10^')}")
ax.set_xlabel("Total distance (km)"); ax.set_ylabel("Secret key rate (bits per pulse)"); ax.legend(frameon=False, fontsize=7)
ax.grid(alpha=.25, which="both"); fig.savefig("figs/fig9_finite.png"); plt.close(fig)

# sensitivity
SE = json.load(open("sensitivity.json"))
d = [r for r in SE if r["kind"] == "dcr_ap"]
dcrs = sorted({r["v1"] for r in d}); aps = sorted({r["v2"] for r in d})
M = np.zeros((len(aps), len(dcrs)))
for r in d:
    M[aps.index(r["v2"]), dcrs.index(r["v1"])] = r["maxL"]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.3, 1]})
im = ax[0].imshow(M, origin="lower", cmap="viridis", aspect="auto")
ax[0].set_xticks(range(len(dcrs))); ax[0].set_xticklabels([f"{int(x):,}" for x in dcrs], fontsize=7)
ax[0].set_yticks(range(len(aps))); ax[0].set_yticklabels([f"{x:.1f}" for x in aps])
for i in range(len(aps)):
    for j in range(len(dcrs)):
        ax[0].text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", color="w" if M[i, j] < M.max() * 0.7 else "k", fontsize=7)
ax[0].set_xlabel("Dark-count rate (counts/s)"); ax[0].set_ylabel("Trap-release probability $P_{ap,0}$")
ax[0].set_title("(a) Maximum distance (km)", fontsize=8.5); fig.colorbar(im, ax=ax[0], fraction=.046)
jj = [r for r in SE if r["kind"] == "jitter"]
ax[1].bar([f"{r['v1'] * 1e12:.0f}" for r in jj], [r["maxL"] for r in jj], color="#1f77b4")
ax[1].set_xlabel("Gaussian jitter σ (ps)"); ax[1].set_ylabel("Maximum distance (km)"); ax[1].set_title("(b) Timing jitter", fontsize=8.5)
for i, r in enumerate(jj):
    ax[1].text(i, r["maxL"] + 2, f"{r['maxL']:.0f}", ha="center", fontsize=7)
fig.tight_layout(); fig.savefig("figs/fig10_sensitivity.png"); plt.close(fig)
S["sens"] = SE

# ---------------------------------------------------------------- summary numbers
summ = {}
for n, rows in K.items():
    d_ = {r["L"]: r for r in rows}
    summ[n] = dict(maxL=max([r["L"] for r in rows if r["R"] > 0] or [0]),
                   **{f"R{L}": d_.get(float(L), {"R": 0})["R"] for L in (0, 25, 50, 100, 150, 200)},
                   **{f"E{L}": d_.get(float(L), {}).get("EZ") for L in (0, 50, 100, 150, 200)},
                   **{f"set{L}": {k: d_.get(float(L), {}).get(k) for k in ("mu", "eta", "pd", "desc")} for L in (0, 50, 100, 150, 200)})
S["summary"] = summ
S["finite"] = {n: max([r["L"] for r in rows if r["R"] > 0] or [0]) for n, rows in FN.items()}
S["finite_R50"] = {n: {r["L"]: r["R"] for r in rows}.get(50.0, 0) for n, rows in FN.items()}
S["pap"] = {f"{h * 1e6:g}": p_afterpulse(h) for h in B.HOLDS}
json.dump(S, open("summary.json", "w"), indent=1, default=float)
print(json.dumps(summ, indent=0, default=float)[:2500])
print(S["tradeoff"]); print(S["finite"])
