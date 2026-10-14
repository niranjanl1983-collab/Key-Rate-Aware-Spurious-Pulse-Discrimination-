"""Event-driven Monte-Carlo of a free-running InGaAs/InP SPAD with time tagging.

Physical processes: photon detections (Bernoulli per time slot, timing jitter
with Gaussian core + exponential diffusion tail), thermally generated dark
counts (Poisson), afterpulsing (two trap levels, release after hold-off),
active hold-off (dead time). Every avalanche is time-tagged and labelled
(signal / dark / afterpulse) so that spurious-pulse discriminators can be
trained and evaluated.
"""
import heapq
import numpy as np

DEFAULT = dict(
    f_rep=100e6,          # pulse repetition rate (Hz)
    eta_d=0.145,          # detection efficiency
    dcr=3000.0,           # dark-count rate (counts/s)
    jit_sigma=50e-12,     # Gaussian timing jitter (s)  (FWHM ~ 118 ps)
    tail_frac=0.10,       # fraction of detections in the diffusion tail
    tail_tau=300e-12,     # tail time constant (s)
    p_ap0=0.60,           # total trap-release probability per avalanche
    trap_w=(0.6, 0.4),    # weights of the two trap levels
    trap_tau=(0.5e-6, 5e-6),  # trap lifetimes (s)
)


def p_afterpulse(hold, p=DEFAULT):
    """Probability that an avalanche is followed by an afterpulse after hold-off."""
    return p["p_ap0"] * sum(w * np.exp(-hold / t) for w, t in zip(p["trap_w"], p["trap_tau"]))


def simulate(q, hold, p=DEFAULT, n_target=300_000, max_time=None, rng=None):
    """q: probability per slot that an incident photon would be detected
    (eta_d already included). hold: hold-off time (s).
    Returns dict with avalanche times, labels and bookkeeping counts."""
    rng = rng or np.random.default_rng(0)
    T = 1.0 / p["f_rep"]
    rate = q * p["f_rep"] + p["dcr"]
    dur = n_target / rate
    if max_time:
        dur = min(dur, max_time)
    n_slots = int(dur / T)
    dur = n_slots * T
    # photon detection candidates
    n_ph = rng.binomial(n_slots, q)
    slots = np.sort(rng.choice(n_slots, n_ph, replace=False)) if n_ph < n_slots // 4 else \
        np.flatnonzero(rng.random(n_slots) < q)
    n_ph = len(slots)
    jit = rng.normal(0, p["jit_sigma"], n_ph)
    tail = rng.random(n_ph) < p["tail_frac"]
    jit[tail] += rng.exponential(p["tail_tau"], tail.sum())
    t_ph = slots * T + jit
    # dark-count candidates
    n_dk = rng.poisson(p["dcr"] * dur)
    t_dk = rng.uniform(0, dur, n_dk)
    times = np.concatenate([t_ph, t_dk]); lab = np.concatenate([np.zeros(n_ph, np.int8), np.ones(n_dk, np.int8)])
    o = np.argsort(times); times = times[o]; lab = lab[o]
    # event loop (afterpulses generated on the fly)
    heap = []
    out_t, out_l = [], []
    dead_until = -1.0
    i, n = 0, len(times)
    tw, ttau = np.array(p["trap_w"]), np.array(p["trap_tau"])
    while i < n or heap:
        if heap and (i >= n or heap[0][0] < times[i]):
            t, l = heapq.heappop(heap)
        else:
            t, l = times[i], lab[i]; i += 1
        if t < dead_until or t > dur:
            continue
        out_t.append(t); out_l.append(l)
        dead_until = t + hold
        if rng.random() < p["p_ap0"]:
            d = rng.exponential(ttau[int(rng.random() < tw[1])])
            if d >= hold:  # released after the hold-off -> afterpulse avalanche
                heapq.heappush(heap, (t + d, 2))
    out_t = np.array(out_t); out_l = np.array(out_l, np.int8)
    return dict(t=out_t, lab=out_l, n_slots=n_slots, n_ph=n_ph, T=T, dur=dur, q=q, hold=hold)


def features(sim):
    """Per-avalanche features: offset to the nearest slot centre, time since the
    previous avalanche, and the local avalanche rate (last 50 avalanches)."""
    t, T = sim["t"], sim["T"]
    off = t - np.round(t / T) * T
    dt = np.diff(t, prepend=-1.0)
    dt[0] = 1.0
    k = 50
    lr = np.empty_like(t)
    lr[k:] = k / (t[k:] - t[:-k])
    lr[:k] = len(t) / sim["dur"]
    return np.stack([off * 1e12, np.log10(dt), np.log10(lr)], 1)  # ps, log10 s, log10 Hz


def evaluate(sim, accept):
    """Return (rho_sig, p_spur): fraction of photon-detection opportunities that
    yield an accepted signal avalanche, and accepted spurious avalanches per slot."""
    lab = sim["lab"]
    sig = np.sum(accept & (lab == 0))
    spur = np.sum(accept & (lab != 0))
    return sig / max(sim["n_ph"], 1), spur / sim["n_slots"]
