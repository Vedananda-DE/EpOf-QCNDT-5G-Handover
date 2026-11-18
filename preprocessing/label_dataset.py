"""
Label the 5G production traces for EpOf-QCNDT handover prediction.

The logs contain the serving cell only, so the ground truth is the usual
serving-measurement rule (3GPP Event A2 plus a speed-based target tier):

    0  Maintain current link
    1  Handover to a macro cell
    2  Handover to a small cell

The link is kept when RSRP is at least -100 dBm, SNR is at least -5 dB
and RSRP has not fallen by more than the 2 dB hysteresis in one step.
Otherwise a user at 30 km/h or faster is sent to a macro cell, and a
slower user is sent to a small cell. A new decision is held for 2 s so
the label does not ping-pong. delta_RSRP is the one-step change of the
logged serving RSRP. The model features are RSRP, delta_RSRP, SNR and Speed.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import io as sio
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "dataset" / "input" / "5G-production-dataset"
OUT_LABELED = ROOT / "dataset" / "processed" / "labeled_data"
OUT_NORM = ROOT / "dataset" / "processed" / "normalized_data"
OUT_CLEAN = ROOT / "dataset" / "processed" / "cleaned_data"
OUT_ANALYSIS = ROOT / "dataset" / "dataset_analysis"

SEED = 42
HOM_DB = 2.0
RSRP_KEEP_DBM = -100.0
SNR_MIN_DB = -5.0
SPEED_MACRO_KMH = 30.0
PINGPONG_S = 2.0
GAP_RESET_S = 5.0
NOISE_DBM = -94.0
D0_M = 1.0
DMIN_M = 10.0
EARTH_R = 6371000.0

TIERS = [
    ("macro", 6, 1500.0, 900e6, 2.1, 36.0, 0.0),
    ("bs", 7, 500.0, 2.4e9, 3.0, 22.0, 0.0),
    ("femto", 5, 100.0, 18e9, 4.0, 20.0, 0.0),
]
CLASS_NAME = {
    0: "Maintain Current Link",
    1: "Handover to Macro BS",
    2: "Handover to Small Cell",
}
SELECTED = ["RSRP", "delta_RSRP", "SNR", "Speed"]


def local_xy(lat, lon, lat0, lon0):
    x = np.deg2rad(lon - lon0) * EARTH_R * np.cos(np.deg2rad(lat0))
    y = np.deg2rad(lat - lat0) * EARTH_R
    return np.column_stack([x, y])


def haversine(lat1, lon1, lat2, lon2):
    p1 = np.deg2rad(lat1)
    p2 = np.deg2rad(lat2)
    dp = np.deg2rad(lat2 - lat1)
    dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def discover_files():
    rows = []
    for dirpath, dirnames, filenames in os.walk(DATA_ROOT):
        dirnames[:] = [d for d in dirnames if d != "__MACOSX" and not d.startswith(".")]
        for name in filenames:
            if not name.endswith(".csv") or name.startswith("._"):
                continue
            path = Path(dirpath) / name
            rel = path.relative_to(DATA_ROOT)
            parts = rel.parts
            application = parts[0]
            mobility = "Driving" if "Driving" in parts else "Static"
            rows.append(
                {
                    "path": path,
                    "session_id": rel.as_posix()[:-4],
                    "mobility": mobility,
                    "application": application,
                }
            )
    rows.sort(key=lambda r: r["session_id"])
    return rows


def read_session(meta):
    df = pd.read_csv(meta["path"], na_values=["-", "NA", "nan", "NaN"])
    df["session_id"] = meta["session_id"]
    df["mobility"] = meta["mobility"]
    df["application"] = meta["application"]
    df["timestamp"] = pd.to_datetime(
        df["Timestamp"], format="%Y.%m.%d_%H.%M.%S", errors="coerce"
    )
    for col in ["Longitude", "Latitude", "Speed", "RSRP", "RSRQ", "SNR", "CQI",
                "DL_bitrate", "UL_bitrate", "CellID"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["_order"] = np.arange(len(df))
    df = df.sort_values(["timestamp", "_order"], kind="mergesort")
    return df


def clean_session(df):
    invalid = df["RSRP"].isna() | (df["RSRP"] < -160) | (df["RSRP"] > -20)
    df.loc[invalid, "RSRP"] = np.nan
    df["SNR"] = df["SNR"].interpolate(limit=3)
    df["SNR"] = df["SNR"].ffill(limit=2).bfill(limit=2)
    dt = df["timestamp"].diff().dt.total_seconds().to_numpy()
    if np.isnan(dt).any():
        dt = np.where(np.isnan(dt), 1.0, dt)
    dt[0] = 0.0
    rsrp = df["RSRP"].to_numpy(dtype=float)
    delta = np.zeros(len(df), dtype=float)
    for i in range(1, len(df)):
        if dt[i] <= PINGPONG_S and np.isfinite(rsrp[i]) and np.isfinite(rsrp[i - 1]):
            delta[i] = rsrp[i] - rsrp[i - 1]
    df["delta_RSRP"] = delta
    df["dt_s"] = dt
    return df


def place_sites(frames):
    all_df = pd.concat(
        [f[["Latitude", "Longitude", "mobility"]] for f in frames], ignore_index=True
    )
    lat0 = float(all_df["Latitude"].median())
    lon0 = float(all_df["Longitude"].median())
    xy = local_xy(
        all_df["Latitude"].to_numpy(), all_df["Longitude"].to_numpy(), lat0, lon0
    )
    rng = np.random.RandomState(SEED)
    take = rng.choice(len(xy), size=min(40000, len(xy)), replace=False)
    sample = xy[take]
    static_mask = all_df["mobility"].to_numpy() == "Static"
    static_xy = xy[static_mask]
    if len(static_xy) > 20000:
        static_xy = static_xy[rng.choice(len(static_xy), 20000, replace=False)]

    pools = {"macro": sample, "bs": sample, "femto": static_xy if len(static_xy) else sample}
    centers = []
    occupied = []
    for tier, count, radius, freq, alpha, aref, ptx in TIERS:
        km = KMeans(n_clusters=count, n_init=10, random_state=SEED)
        km.fit(pools[tier])
        for idx, center in enumerate(km.cluster_centers_):
            point = center.astype(float).copy()
            for other in occupied:
                shift = point - other
                dist = np.linalg.norm(shift)
                if dist < 80.0:
                    direction = shift if dist > 1.0 else np.array([0.0, 1.0])
                    direction = direction / np.linalg.norm(direction)
                    point = other + direction * 80.0
            occupied.append(point)
            prefix = {"macro": "M", "bs": "B", "femto": "F"}[tier]
            dlat = (point[1] / EARTH_R) * 180.0 / np.pi
            dlon = (point[0] / (EARTH_R * np.cos(np.deg2rad(lat0)))) * 180.0 / np.pi
            centers.append(
                {
                    "cell_name": f"{prefix}{idx + 1}",
                    "tier": tier,
                    "radius_m": radius,
                    "frequency_hz": freq,
                    "alpha": alpha,
                    "reference_attenuation_db": aref,
                    "tx_power_dbm": ptx,
                    "x_m": point[0],
                    "y_m": point[1],
                    "Latitude": lat0 + dlat,
                    "Longitude": lon0 + dlon,
                }
            )
    return pd.DataFrame(centers), lat0, lon0


def radio_maps(df, sites):
    lat = df["Latitude"].to_numpy(dtype=float)
    lon = df["Longitude"].to_numpy(dtype=float)
    n = len(df)
    m = len(sites)
    rsrp = np.empty((n, m), dtype=float)
    for j, site in sites.iterrows():
        dist = haversine(lat, lon, site["Latitude"], site["Longitude"])
        dist = np.maximum(dist, DMIN_M)
        pl = site["reference_attenuation_db"] + 10.0 * site["alpha"] * np.log10(dist / D0_M)
        rsrp[:, j] = site["tx_power_dbm"] - pl
    power = np.power(10.0, rsrp / 10.0)
    noise = 10.0 ** (NOISE_DBM / 10.0)
    interference = power.sum(axis=1, keepdims=True) - power + noise
    sinr = 10.0 * np.log10(power / interference)
    return rsrp, sinr


def measurement_decision(rsrp, delta, snr, speed):
    ok = np.isfinite(rsrp) & np.isfinite(delta) & np.isfinite(snr) & np.isfinite(speed)
    keep = ok & (rsrp >= RSRP_KEEP_DBM) & (snr >= SNR_MIN_DB) & (delta > -HOM_DB)
    raw = np.full(len(rsrp), -1, dtype=np.int8)
    raw[keep] = 0
    handover = ok & ~keep
    raw[handover & (speed >= SPEED_MACRO_KMH)] = 1
    raw[handover & (speed < SPEED_MACRO_KMH)] = 2
    return raw


def apply_hold(raw, dt):
    """Hold a decision for the ping-pong window so the class cannot chatter."""
    labels = np.full(len(raw), -1, dtype=np.int8)
    committed = None
    since = 0.0
    elapsed = 0.0
    for i in range(len(raw)):
        if i:
            elapsed += float(dt[i])
        if raw[i] < 0:
            continue
        if committed is None or dt[i] > GAP_RESET_S:
            committed = int(raw[i])
            since = elapsed
        elif int(raw[i]) != committed and (elapsed - since) >= PINGPONG_S:
            committed = int(raw[i])
            since = elapsed
        labels[i] = committed
    return labels


def label_session(df, sites, rsrp, sinr):
    n = len(df)
    nearest = np.argmax(rsrp, axis=1)
    raw = measurement_decision(
        df["RSRP"].to_numpy(dtype=float),
        df["delta_RSRP"].to_numpy(dtype=float),
        df["SNR"].to_numpy(dtype=float),
        df["Speed"].to_numpy(dtype=float),
    )
    labels = apply_hold(raw, df["dt_s"].to_numpy(dtype=float))
    df = df.copy()
    df["HO_label"] = labels
    df["HO_class"] = [CLASS_NAME[int(v)] if int(v) >= 0 else "" for v in labels]
    df["serving_cell"] = sites["cell_name"].to_numpy()[nearest]
    df["serving_tier"] = sites["tier"].to_numpy()[nearest]
    df["synth_RSRP"] = rsrp[np.arange(n), nearest]
    df["synth_SINR"] = sinr[np.arange(n), nearest]
    return df


def assign_splits(session_meta):
    rng = np.random.RandomState(SEED)
    split_of = {}
    groups = {}
    for meta in session_meta:
        groups.setdefault((meta["mobility"], meta["application"]), []).append(meta["session_id"])
    for key in sorted(groups):
        ids = sorted(groups[key])
        order = rng.permutation(len(ids))
        ids = [ids[i] for i in order]
        n = len(ids)
        n_val = max(1, int(np.floor(n * 0.15))) if n >= 3 else 0
        n_test = max(1, int(np.floor(n * 0.15))) if n >= 3 else (1 if n == 2 else 0)
        if n_val + n_test >= n:
            n_val = 1 if n >= 4 else 0
            n_test = 1 if n >= 3 else 0
        n_train = n - n_val - n_test
        for sid in ids[:n_train]:
            split_of[sid] = "train"
        for sid in ids[n_train:n_train + n_val]:
            split_of[sid] = "val"
        for sid in ids[n_train + n_val:]:
            split_of[sid] = "test"
    return split_of


def feature_ranking(df):
    train = df[df["split"] == "train"].copy()
    work = pd.DataFrame(
        {
            "RSRP": train["RSRP"],
            "delta_RSRP": train["delta_RSRP"],
            "SNR": train["SNR"],
            "Speed": train["Speed"],
            "RSRQ": train["RSRQ"],
            "CQI": train["CQI"],
            "Longitude": train["Longitude"],
            "Latitude": train["Latitude"],
            "CellID": train["CellID"],
        }
    )
    mode_codes = {name: i for i, name in enumerate(sorted(train["NetworkMode"].dropna().unique()))}
    work["NetworkMode"] = train["NetworkMode"].map(mode_codes)
    for col in work.columns:
        work[col] = work[col].fillna(work[col].median())
    tree = DecisionTreeClassifier(
        criterion="gini", max_depth=5, min_samples_leaf=50, random_state=SEED
    )
    tree.fit(work.to_numpy(), train["HO_label"].to_numpy())
    rank = (
        pd.DataFrame({"feature": work.columns, "gini_importance": tree.feature_importances_})
        .sort_values("gini_importance", ascending=False)
        .reset_index(drop=True)
    )
    rank["rank"] = np.arange(1, len(rank) + 1)
    rank["selected_for_model"] = rank["feature"].isin(SELECTED)
    return rank


def scaling_params(df):
    train = df[df["split"] == "train"]
    rows = []
    scaled = pd.DataFrame(index=df.index)
    for name in SELECTED:
        lo = float(train[name].min())
        hi = float(train[name].max())
        span = hi - lo if hi > lo else 1.0
        values = ((df[name] - lo) / span).clip(0.0, 1.0)
        scaled[name] = values
        rows.append({"feature": name, "min": lo, "max": hi})
    return scaled, pd.DataFrame(rows)


def write_reports(raw_rows, kept, sites, rank, scale, split_of):
    lines = []
    lines.append("5G production dataset analysis and handover labeling")
    lines.append("")
    lines.append("Source: Raca et al., MMSys 2020, 5G production dataset.")
    lines.append("Original files were not modified. A directory junction points at them.")
    lines.append(f"Sessions: {len(split_of)}")
    lines.append(f"Rows read: {raw_rows}")
    lines.append(f"Rows kept for supervised learning: {len(kept)}")
    lines.append(f"Rows removed (invalid RSRP, missing SNR, or missing position): {raw_rows - len(kept)}")
    lines.append("")
    lines.append("Label convention")
    lines.append("  Serving-cell Event A2 with a speed-based target tier.")
    lines.append("  Maintain when RSRP >= -100 dBm, SNR >= -5 dB and delta_RSRP > -2 dB.")
    lines.append("  Otherwise Speed >= 30 km/h selects a macro handover, and a lower speed")
    lines.append("  selects a small-cell handover. The decision is held for 2.0 s.")
    lines.append("  delta_RSRP(t) = RSRP(t) - RSRP(t-1) inside a session when the step is <= 2 s.")
    lines.append("  The Table 1 sites are stored for the later network simulation.")
    lines.append("  Their path loss is PL = A_ref + 10 * alpha * log10(d / 1 m), d >= 10 m.")
    lines.append("")
    lines.append("Selected decision-tree features: RSRP, delta_RSRP, SNR, Speed.")
    lines.append("")
    lines.append("Class counts")
    counts = kept["HO_label"].value_counts().sort_index()
    for label, count in counts.items():
        lines.append(
            f"  {label} {CLASS_NAME[int(label)]}: {count} ({100.0 * count / len(kept):.2f}%)"
        )
    lines.append("")
    lines.append("Class counts by mobility")
    for mobility, part in kept.groupby("mobility"):
        lines.append(f"  {mobility} ({len(part)} rows)")
        vc = part["HO_label"].value_counts().sort_index()
        for label, count in vc.items():
            lines.append(f"    {label}: {count} ({100.0 * count / len(part):.2f}%)")
    lines.append("")
    lines.append("Session split (70/15/15 by mobility and application)")
    for split, part in kept.groupby("split"):
        lines.append(f"  {split}: {part['session_id'].nunique()} sessions, {len(part)} rows")
    lines.append("")
    lines.append("Gini importance on the training sessions (depth 5)")
    for _, row in rank.iterrows():
        mark = " selected" if row["selected_for_model"] else ""
        lines.append(f"  {int(row['rank']):2d}. {row['feature']}: {row['gini_importance']:.4f}{mark}")
    lines.append("")
    lines.append("Training min/max used to scale the four selected features to [0, 1]")
    for _, row in scale.iterrows():
        lines.append(f"  {row['feature']}: [{row['min']:.6g}, {row['max']:.6g}]")
    lines.append("")
    lines.append("Synthetic sites")
    for _, site in sites.iterrows():
        lines.append(
            f"  {site['cell_name']} {site['tier']} lat {site['Latitude']:.6f} lon {site['Longitude']:.6f}"
        )
    text = "\n".join(lines) + "\n"
    (OUT_ANALYSIS / "dataset_analysis.txt").write_text(text, encoding="utf-8")
    return text


def main():
    for folder in (OUT_LABELED, OUT_NORM, OUT_CLEAN, OUT_ANALYSIS):
        folder.mkdir(parents=True, exist_ok=True)

    metas = discover_files()
    split_of = assign_splits(metas)
    frames = []
    raw_rows = 0
    for meta in metas:
        session = clean_session(read_session(meta))
        raw_rows += len(session)
        session["split"] = split_of[meta["session_id"]]
        frames.append(session)

    sites, lat0, lon0 = place_sites(frames)
    labeled_frames = []
    for session in frames:
        usable = session["Latitude"].notna() & session["Longitude"].notna()
        work = session.loc[usable].copy()
        if work.empty:
            continue
        rsrp, sinr = radio_maps(work, sites)
        labeled_frames.append(label_session(work, sites, rsrp, sinr))

    labeled = pd.concat(labeled_frames, ignore_index=True)
    before = len(labeled)
    labeled = labeled[
        labeled["RSRP"].notna()
        & labeled["SNR"].notna()
        & labeled["Speed"].notna()
        & (labeled["HO_label"] >= 0)
    ]
    labeled = labeled.reset_index(drop=True)

    rank = feature_ranking(labeled)
    scaled, scale = scaling_params(labeled)

    keep_cols = [
        "session_id", "mobility", "application", "split", "Timestamp",
        "Longitude", "Latitude", "Speed", "Operatorname", "CellID", "NetworkMode",
        "RSRP", "delta_RSRP", "RSRQ", "SNR", "CQI", "DL_bitrate", "UL_bitrate", "State",
        "serving_cell", "serving_tier", "synth_RSRP", "synth_SINR", "HO_label", "HO_class",
    ]
    labeled[keep_cols].to_csv(OUT_LABELED / "labeled_traces.csv", index=False)
    scaled.assign(HO_label=labeled["HO_label"], split=labeled["split"], session_id=labeled["session_id"]).to_csv(
        OUT_NORM / "selected_features_scaled.csv", index=False
    )
    scale.to_csv(OUT_NORM / "feature_scaling.csv", index=False)
    rank.to_csv(OUT_ANALYSIS / "feature_ranking.csv", index=False)
    sites.to_csv(OUT_LABELED / "cell_layout.csv", index=False)

    split_rows = (
        labeled.groupby(["session_id", "mobility", "application", "split"], as_index=False)
        .size()
        .rename(columns={"size": "rows"})
    )
    split_rows.to_csv(OUT_LABELED / "session_split.csv", index=False)

    summary = (
        labeled.groupby(["mobility", "application", "HO_label"], as_index=False)
        .agg(
            rows=("HO_label", "size"),
            sessions=("session_id", "nunique"),
            mean_rsrp=("RSRP", "mean"),
            mean_snr=("SNR", "mean"),
            mean_speed=("Speed", "mean"),
            mean_delta_rsrp=("delta_RSRP", "mean"),
        )
    )
    summary.to_csv(OUT_ANALYSIS / "dataset_summary.csv", index=False)

    mobility_code = labeled["mobility"].map({"Static": 0, "Driving": 1}).to_numpy(dtype=np.int8)
    split_code = labeled["split"].map({"train": 0, "val": 1, "test": 2}).to_numpy(dtype=np.int8)
    sio.savemat(
        OUT_LABELED / "labeled_traces.mat",
        {
            "RSRP": labeled["RSRP"].to_numpy(dtype=np.float64),
            "delta_RSRP": labeled["delta_RSRP"].to_numpy(dtype=np.float64),
            "SNR": labeled["SNR"].to_numpy(dtype=np.float64),
            "Speed": labeled["Speed"].to_numpy(dtype=np.float64),
            "features_scaled": scaled[SELECTED].to_numpy(dtype=np.float64),
            "HO_label": labeled["HO_label"].to_numpy(dtype=np.int8),
            "split_code": split_code,
            "mobility_code": mobility_code,
            "feature_min": scale["min"].to_numpy(dtype=np.float64),
            "feature_max": scale["max"].to_numpy(dtype=np.float64),
            "selected_features": np.array(SELECTED, dtype=object),
            "layout_lat0": np.array([lat0]),
            "layout_lon0": np.array([lon0]),
        },
        do_compression=True,
    )
    report = write_reports(raw_rows, labeled, sites, rank, scale, split_of)
    removed = before - len(labeled)
    print(report)
    print(f"removed_after_radio {removed}")
    print(f"wrote {OUT_LABELED / 'labeled_traces.csv'}")


if __name__ == "__main__":
    main()
