"""Recover a known exoplanet's transit period from TESS photometry.

Run from the astronomy_visual repository root:
    python -m experiments.tess_transits
    python -m experiments.tess_transits --refresh
"""

import argparse
from pathlib import Path

import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares
from astropy.timeseries import LombScargle

DATA_DIR = Path("data")
REPORT_DIR = Path("reports/tess_transits")
TIC = "TIC 100100827"  # WASP-18; identifier, NOT the planet's period
KNOWN_PERIOD_DAYS = 0.9414524  # used only AFTER detection for validation


def load_tess(refresh=False):
    cache = DATA_DIR / "tess_wasp18.csv"
    if cache.exists() and not refresh:
        print(f"[TESS] reading {cache}", flush=True)
        return pd.read_csv(cache)

    print(f"[TESS] searching MAST for {TIC}...", flush=True)
    results = lk.search_lightcurve(TIC, mission="TESS", author="SPOC", exptime=120)
    print(f"[TESS] found {len(results)} light-curve products", flush=True)
    if not len(results):
        raise RuntimeError("No TESS SPOC 2-minute observations found for target")

    # Start with a single observing sector: small download, enough transits.
    print(f"[TESS] downloading first sector: {results.table['mission'][0]}", flush=True)
    lc = results[0].download()
    if lc is None:
        raise RuntimeError("TESS download returned no light curve")
    lc = lc.remove_nans().normalize()  # typically uses corrected PDCSAP flux

    data = pd.DataFrame({
        "time_btjd": np.asarray(lc.time.value, dtype=float),
        "flux": np.asarray(lc.flux.value, dtype=float),
    }).dropna()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(cache, index=False)
    print(f"[TESS] cached {len(data):,} measurements to {cache}", flush=True)
    return data


def plot_scatter(x, y, xlabel, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(x, y, s=3, alpha=0.45)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / filename, dpi=160)
    plt.close(fig)


def main(refresh=False):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_tess(refresh)

    time = data["time_btjd"].to_numpy()
    raw_flux = data["flux"].to_numpy()
    lc = lk.LightCurve(time=time, flux=raw_flux)

    # Smooth away long-term trends (~13h window for 2-minute cadence).
    # Do NOT sigma-clip away downward transits!
    flat = lc.flatten(
        window_length=1201,  # Previously 401
        niters=2,
        sigma=10,
    ).remove_nans()
    t = np.asarray(flat.time.value, dtype=float)
    flux = np.asarray(flat.flux.value, dtype=float)

    span = float(t.max() - t.min())
    noise = 1.4826 * np.median(np.abs(flux - np.median(flux)))
    if not np.isfinite(noise) or noise <= 0 or span <= 2:
        raise ValueError("Invalid light curve: too little data or no measurable variation")

    print(f"\n[TESS] cadences={len(t):,}, span={span:.2f} days")
    print(f"[TESS] scatter (robust)={noise * 1000:.2f} ppt")

    plot_scatter(time, raw_flux, "Time [BTJD]", "Normalized flux",
                 "WASP-18: raw TESS brightness", "lightcurve.png")

    print("[BLS] searching for periodic transit-shaped dips...", flush=True)
    bls = BoxLeastSquares(t, flux, dy=np.full_like(flux, noise))
    periods = bls.autopower(
        [0.05, 0.08, 0.11, 0.14],  # possible transit durations, days
        minimum_period=0.5,
        maximum_period=min(5.0, span / 3),
        frequency_factor=2.0,
        objective="snr",
    )
    i = int(np.nanargmax(periods.power))
    period = float(periods.period[i])
    duration = float(periods.duration[i])
    midpoint = float(periods.transit_time[i])
    depth = float(periods.depth[i])
    score = float(periods.depth_snr[i])

    # Compare box-shaped transit model to a constant-flux baseline.
    box = bls.model(t, period, duration, midpoint)
    baseline_rmse = float(np.sqrt(np.mean((flux - np.median(flux)) ** 2)))
    bls_rmse = float(np.sqrt(np.mean((flux - box) ** 2)))
    error_minutes = abs(period - KNOWN_PERIOD_DAYS) * 24 * 60

    print("\n[BLS] best transit candidate")
    print(f"  period_days:        {period:.6f}")
    print(f"  duration_hours:     {duration * 24:.2f}")
    print(f"  depth_percent:      {depth * 100:.3f}")
    print(f"  depth_snr:          {score:.1f} (approximate)")
    print(f"  constant_RMSE_ppt:  {baseline_rmse * 1000:.2f}")
    print(f"  transit_RMSE_ppt:   {bls_rmse * 1000:.2f} (in-sample)")
    print("\n[Validation] compared AFTER fitting")
    print(f"  published_period:   {KNOWN_PERIOD_DAYS:.7f} days")
    print(f"  period_error:       {error_minutes:.2f} minutes")

    # Plot BLS scores over candidate periods.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(periods.period, periods.power)
    ax.axvline(period, color="tab:orange", linestyle="--", label="Detected period")
    ax.set(xlabel="Trial period [days]", ylabel="BLS transit-depth SNR",
           title="WASP-18: search over orbital periods")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "periodogram.png", dpi=160)
    plt.close(fig)

    # Fold all observations into one orbit; transit should align at hour zero.
    phase_hours = (((t - midpoint + period / 2) % period) - period / 2) * 24
    show = np.abs(phase_hours) <= 5
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(phase_hours[show], flux[show], s=3, alpha=0.18, label="Observations")
    bins = np.linspace(-5, 5, 61)
    summary = (pd.DataFrame({"phase": phase_hours[show], "flux": flux[show]})
               .assign(bin=lambda frame: pd.cut(frame.phase, bins))
               .groupby("bin", observed=True)["flux"].median())
    centers = np.array([v.mid for v in summary.index])
    ax.plot(centers, summary.to_numpy(), color="tab:orange",
            linewidth=2, label="Binned median")
    ax.set(xlabel="Hours from mid-transit", ylabel="Normalized flux",
           title=f"WASP-18: phase-folded light curve (P={period:.4f} d)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "folded.png", dpi=160)
    plt.close(fig)

    pd.DataFrame({"period_days": periods.period,
                  "bls_snr": periods.power}).to_csv(
                      REPORT_DIR / "periodogram.csv", index=False)
    pd.DataFrame([{
        "target": TIC, "cadences": len(t), "span_days": span,
        "period_days": period, "duration_hours": duration * 24,
        "depth_percent": depth * 100, "depth_snr": score,
        "published_period_days": KNOWN_PERIOD_DAYS,
        "period_error_minutes": error_minutes,
    }]).to_csv(REPORT_DIR / "summary.csv", index=False)


    # Compare transit detection against sinusoidal periodicity detection.
    frequency = np.linspace(1 / 5, 1 / 0.5, 20_000)

    ls = LombScargle(t, flux)
    ls_power = ls.power(frequency)

    best = np.argmax(ls_power)
    ls_period = 1 / frequency[best]

    print("\n[Lomb-Scargle] best periodic signal")
    print(f"  period_days: {ls_period:.6f}")
    print(f"  difference_from_BLS_minutes: {abs(ls_period - period) * 1440:.2f}")

    # Plot the second algorithm's periodogram.
    trial_periods = 1 / frequency
    order = np.argsort(trial_periods)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(trial_periods[order], ls_power[order])
    ax.axvline(
        period,
        color="tab:orange",
        linestyle="--",
        label="BLS detected period",
    )

    ax.set(
        xlabel="Trial period [days]",
        ylabel="Lomb-Scargle power",
        title="WASP-18: sinusoidal periodicity search",
    )

    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "lomb-scargle.png", dpi=160)
    plt.close(fig)

    print(f"\nSaved CSVs and 3 plots to {REPORT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Fetch TESS again")
    main(refresh=parser.parse_args().refresh)
