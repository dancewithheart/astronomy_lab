"""Compare BLS and Lomb--Scargle on a transit and a pulsating star.

Run in astronomy_visual repo: python -m experiments.tess_signal_shapes
Requires previous experiments/tess_transits.py, lightkurve and astropy.
"""
from pathlib import Path

import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares, LombScargle

from experiments.tess_transits import load_tess  # reuse cached WASP-18 data

OUT = Path("reports/tess_signal_shapes")
HD_CACHE = Path("data/tess_hd42608.csv")


def load_pulsator():
    if HD_CACHE.exists():
        print(f"[TESS] reading {HD_CACHE}", flush=True)
        return pd.read_csv(HD_CACHE)

    print("[TESS] searching for HD 42608 (TIC 37498433), sector 6...", flush=True)
    products = lk.search_lightcurve(
        "TIC 37498433", mission="TESS", author="SPOC", sector=6, exptime=120
    )
    if not len(products):
        raise RuntimeError("No SPOC 2-minute light curve found for HD 42608")

    lightcurve = products[0].download()
    if lightcurve is None:
        raise RuntimeError("TESS light curve download failed")
    lightcurve = lightcurve.remove_nans().normalize()
    data = pd.DataFrame({
        "time_btjd": np.asarray(lightcurve.time.value, dtype=float),
        "flux": np.asarray(lightcurve.flux.value, dtype=float),
    }).dropna().sort_values("time_btjd")

    HD_CACHE.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(HD_CACHE, index=False)
    print(f"[TESS] cached {len(data):,} cadences to {HD_CACHE}", flush=True)
    return data


def rmse_ppt(observed, predicted):
    return 1000 * np.sqrt(np.mean((observed - predicted) ** 2))


def compare(label, data, frequency_range, durations, file_stem):
    # Same preprocessing for both algorithms within each star.
    raw = data.sort_values("time_btjd")
    lc = lk.LightCurve(time=raw["time_btjd"].to_numpy(),
                       flux=raw["flux"].to_numpy())
    flat = lc.flatten(window_length=1201, niters=2, sigma=10).remove_nans()
    t = np.asarray(flat.time.value, dtype=float)
    y = np.asarray(flat.flux.value, dtype=float)

    # Chronological split: fit/search on earlier observations only.
    cut = int(len(t) * 0.75)
    t_train, t_test = t[:cut], t[cut:]
    y_train, y_test = y[:cut], y[cut:]
    if len(t_test) < 100:
        raise ValueError("Not enough observations for a time-based holdout")

    fmin, fmax = frequency_range  # cycles per day
    frequency = np.linspace(fmin, fmax, 6000)
    periods = 1 / frequency

    print(f"\n[{label}] cadences={len(t):,}, train={cut:,}, test={len(t_test):,}")
    print(f"[{label}] frequency search={fmin:.1f}–{fmax:.1f} cycles/day")

    ls = LombScargle(t_train, y_train)
    ls_power = ls.power(frequency)
    ls_idx = int(np.argmax(ls_power))
    ls_freq = float(frequency[ls_idx])
    ls_prediction = ls.model(t_test, ls_freq)

    bls = BoxLeastSquares(t_train, y_train)
    bls_result = bls.power(periods, durations, objective="snr", oversample=10)
    bls_idx = int(np.argmax(bls_result.power))
    bls_period = float(bls_result.period[bls_idx])
    bls_prediction = bls.model(
        t_test, bls_period,
        float(bls_result.duration[bls_idx]),
        float(bls_result.transit_time[bls_idx]),
    )

    baseline = rmse_ppt(y_test, np.full_like(y_test, np.median(y_train)))
    ls_rmse = rmse_ppt(y_test, ls_prediction)
    bls_rmse = rmse_ppt(y_test, bls_prediction)

    print(f"  baseline_test_RMSE_ppt: {baseline:.3f}")
    print(f"  LS:  period={1 / ls_freq:.6f} d  test_RMSE={ls_rmse:.3f} ppt")
    print(f"  BLS: period={bls_period:.6f} d  test_RMSE={bls_rmse:.3f} ppt")
    print(f"  best_holdout_fit: {'LS' if ls_rmse < bls_rmse else 'BLS'}")

    # Plot peak locations. Relative peak heights across methods are NOT comparable.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(frequency, ls_power / np.max(ls_power), label="Lomb–Scargle")
    ax.plot(frequency, bls_result.power / np.max(bls_result.power),
            label="BLS", alpha=0.7)
    ax.set(xlabel="Frequency [cycles/day]", ylabel="Power / maximum power",
           title=f"{label}: periodicity search (training data)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"{file_stem}_periodograms.png", dpi=160)
    plt.close(fig)

    return {
        "target": label,
        "cadences": len(t),
        "train_count": len(t_train),
        "test_count": len(t_test),
        "baseline_test_rmse_ppt": baseline,
        "ls_period_days": 1 / ls_freq,
        "ls_test_rmse_ppt": ls_rmse,
        "bls_period_days": bls_period,
        "bls_test_rmse_ppt": bls_rmse,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = [
        compare(
            "WASP-18 (transit)", load_tess(),
            frequency_range=(1 / 5, 1 / 0.5),
            durations=[0.05, 0.08, 0.11, 0.14],
            file_stem="wasp18",
        ),
        compare(
            "HD 42608 (pulsation)", load_pulsator(),
            frequency_range=(35, 60),
            durations=[0.004, 0.005, 0.006, 0.007],
            file_stem="hd42608",
        ),
    ]
    output = pd.DataFrame(results)
    output.to_csv(OUT / "comparison.csv", index=False)
    print("\n[Summary: held-out RMSE; lower is better]")
    print(output[["target", "baseline_test_rmse_ppt", "ls_test_rmse_ppt",
                  "bls_test_rmse_ppt"]].round(3).to_string(index=False))
    print(f"\nSaved to {OUT}/")


if __name__ == "__main__":
    main()
