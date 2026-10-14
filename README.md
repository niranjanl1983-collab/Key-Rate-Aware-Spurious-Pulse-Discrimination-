# KR-SPD: detector-aware MDI-QKD simulation
Requires Python 3.10+, numpy, scipy, scikit-learn, matplotlib.
1. `python stage_a.py`  – SPAD event Monte-Carlo database (224 streams) + gradient-boosting discriminators (~1.5 min, 2 cores)
2. `python stage_b.py`  – discriminator operating points, key rate vs distance for 6 configurations, finite-size (~2 min)
3. `python stage_c.py`  – sensitivity to dark-count rate, afterpulsing and jitter (~3 min)
4. `python figures.py`  – all figures (figs/) and summary.json (includes the BSM Monte-Carlo validation)
`mdi.py`: analytic BSM gains, slot-level BSM Monte-Carlo, LP decoy bounds, key rate. `spad.py`: free-running SPAD event simulator.
