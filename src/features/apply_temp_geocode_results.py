from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TEMP = ROOT / "data" / "temp"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "reports"

PANEL_PATH = PROCESSED / "schooldata_modeling_panel_2008_2025_geocoded.csv"
OUT_PANEL = PROCESSED / "schooldata_modeling_panel_2008_2025_geocoded.csv"
APPLIED_RESULTS = PROCESSED / "schooldata_temp_geocode_results_applied.csv"
CLOSED_UNMATCHED = PROCESSED / "schooldata_closed_unmatched_after_geocode.csv"
REPORT_PATH = REPORTS / "schooldata_temp_geocode_apply_report.csv"

COL_YEAR = "데이터_연도"
COL_SIDO = "시도"
COL_SGG = "행정구"
COL_LEVEL = "학교급"
COL_NAME = "학교명"
COL_STATUS = "상태"
COL_ADDRESS = "주소"

INPUTS = [
    ("latest", 1, TEMP / "closed_latest_geocode_results_all_batches.csv"),
    ("recheck", 2, TEMP / "closed_recheck_geocode_results_all_batches.csv"),
    ("manual", 3, TEMP / "manual_closed_geocode_results.csv"),
]


def load_success_rows() -> pd.DataFrame:
    frames = []
    for source_label, priority, path in INPUTS:
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        required = {"school_key", "status", "lat", "lon"}
        if not required.issubset(df.columns):
            continue
        ok = df[df["status"].eq("ok") & df["lat"].notna() & df["lon"].notna()].copy()
        if ok.empty:
            continue
        ok["temp_result_source"] = source_label
        ok["temp_priority"] = priority
        if "source" in ok.columns:
            ok = ok.rename(columns={"source": "geocode_provider"})
        else:
            ok["geocode_provider"] = ""
        if "matched_address" not in ok.columns:
            ok["matched_address"] = ""
        frames.append(
            ok[
                [
                    "school_key",
                    "school_name",
                    "lat",
                    "lon",
                    "geocode_provider",
                    "matched_address",
                    "temp_result_source",
                    "temp_priority",
                ]
            ]
        )
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["school_key", "temp_priority"]).drop_duplicates("school_key", keep="last")
    return combined


def summarize(panel: pd.DataFrame, label: str) -> dict[str, float | int | str]:
    has_coord = panel["lttud"].notna() & panel["lgtud"].notna()
    closed = panel[panel[COL_STATUS].astype(str).str.contains("폐", na=False)]
    closed_has_coord = closed["lttud"].notna() & closed["lgtud"].notna()
    return {
        "stage": label,
        "rows": len(panel),
        "coordinate_rows": int(has_coord.sum()),
        "missing_coordinate_rows": int((~has_coord).sum()),
        "coordinate_rate": float(has_coord.mean()),
        "closed_rows": len(closed),
        "closed_coordinate_rows": int(closed_has_coord.sum()),
        "closed_missing_coordinate_rows": int((~closed_has_coord).sum()),
        "closed_coordinate_rate": float(closed_has_coord.mean()) if len(closed) else 0.0,
    }


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    panel["lttud"] = pd.to_numeric(panel["lttud"], errors="coerce")
    panel["lgtud"] = pd.to_numeric(panel["lgtud"], errors="coerce")

    before = summarize(panel, "before")
    success = load_success_rows()
    if success.empty:
        pd.DataFrame([before]).to_csv(REPORT_PATH, index=False, encoding="utf-8-sig")
        print("No temp geocode successes found.")
        return 0

    panel = panel.merge(success, on="school_key", how="left", suffixes=("", "_temp"))
    missing_mask = panel["lttud"].isna() | panel["lgtud"].isna()
    fill_mask = missing_mask & panel["lat"].notna() & panel["lon"].notna()
    panel.loc[fill_mask, "lttud"] = panel.loc[fill_mask, "lat"]
    panel.loc[fill_mask, "lgtud"] = panel.loc[fill_mask, "lon"]
    panel.loc[fill_mask, "coordinate_source"] = "temp_" + panel.loc[fill_mask, "temp_result_source"].astype(str)
    panel.loc[fill_mask, "matched_address"] = panel.loc[fill_mask, "matched_address"].astype(str)

    applied = panel.loc[
        fill_mask,
        [
            COL_YEAR,
            COL_SIDO,
            COL_SGG,
            COL_LEVEL,
            COL_NAME,
            COL_STATUS,
            COL_ADDRESS,
            "school_key",
            "lttud",
            "lgtud",
            "coordinate_source",
            "geocode_provider",
            "matched_address",
        ],
    ].copy()

    drop_cols = [
        "school_name",
        "lat",
        "lon",
        "geocode_provider",
        "matched_address",
        "temp_result_source",
        "temp_priority",
    ]
    panel = panel.drop(columns=[c for c in drop_cols if c in panel.columns])
    panel.to_csv(OUT_PANEL, index=False, encoding="utf-8-sig")
    applied.to_csv(APPLIED_RESULTS, index=False, encoding="utf-8-sig")

    closed = panel[panel[COL_STATUS].astype(str).str.contains("폐", na=False)].copy()
    closed_missing = closed[closed["lttud"].isna() | closed["lgtud"].isna()].copy()
    cols = [
        c
        for c in [
            COL_YEAR,
            COL_SIDO,
            COL_SGG,
            COL_LEVEL,
            COL_NAME,
            COL_STATUS,
            COL_ADDRESS,
            "school_key",
            "coordinate_source",
        ]
        if c in closed_missing.columns
    ]
    closed_missing[cols].to_csv(CLOSED_UNMATCHED, index=False, encoding="utf-8-sig")

    after = summarize(panel, "after")
    report = pd.DataFrame(
        [
            before,
            after
            | {
                "temp_success_unique_school_keys": int(success["school_key"].nunique()),
                "temp_applied_rows": int(fill_mask.sum()),
                "temp_applied_unique_school_keys": int(panel.loc[fill_mask, "school_key"].nunique()),
            },
        ]
    )
    report.to_csv(REPORT_PATH, index=False, encoding="utf-8-sig")
    print(report.to_string(index=False))
    print("saved panel:", OUT_PANEL)
    print("saved unmatched closed:", CLOSED_UNMATCHED)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
