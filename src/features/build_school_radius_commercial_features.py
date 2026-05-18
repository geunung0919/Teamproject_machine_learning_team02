from __future__ import annotations

from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from project_config import valid_sido_coord_mask


ROOT = SRC.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "reports"

EARTH_RADIUS_KM = 6371.0088
RADII_KM = [0.5, 1.0, 2.0]
CHUNK_SIZE = 250_000


def classify_shop(frame: pd.DataFrame) -> pd.DataFrame:
    text = (
        frame["indsLclsNm"].fillna("")
        + " "
        + frame["indsMclsNm"].fillna("")
        + " "
        + frame["indsSclsNm"].fillna("")
        + " "
        + frame["bizesNm"].fillna("")
    )
    out = frame.copy()
    out["is_education"] = text.str.contains("교육|학원|교습|독서실|공부방", regex=True)
    out["is_kids"] = text.str.contains("아동|어린이|유아|키즈|문구|완구|장난감", regex=True)
    out["is_medical"] = text.str.contains("병원|의원|약국|의료|치과|한의", regex=True)
    return out


def count_radius_features() -> pd.DataFrame:
    schools = pd.read_csv(PROCESSED / "final_national_current_school_features.csv", low_memory=False)
    schools["lttud"] = pd.to_numeric(schools["lttud"], errors="coerce")
    schools["lgtud"] = pd.to_numeric(schools["lgtud"], errors="coerce")
    schools = schools[valid_sido_coord_mask(schools)].copy().reset_index(drop=True)

    coords_rad = np.deg2rad(schools[["lttud", "lgtud"]].to_numpy(dtype=float))
    result = schools[["schlCd", "schlNm", "requested_sido_name", "sgg_code", "school_level", "lttud", "lgtud"]].copy()
    for radius in RADII_KM:
        suffix = str(radius).replace(".", "_")
        for col in ["all", "education", "kids", "medical"]:
            result[f"radius_{suffix}km_{col}_shops"] = 0

    usecols = [
        "bizesId",
        "bizesNm",
        "indsLclsNm",
        "indsMclsNm",
        "indsSclsNm",
        "ctprvnNm",
        "signguCd",
        "signguNm",
        "lon",
        "lat",
    ]
    shop_path = RAW / "national_small_shop.csv"
    for chunk_idx, chunk in enumerate(pd.read_csv(shop_path, usecols=usecols, chunksize=CHUNK_SIZE, low_memory=False), start=1):
        chunk["lat"] = pd.to_numeric(chunk["lat"], errors="coerce")
        chunk["lon"] = pd.to_numeric(chunk["lon"], errors="coerce")
        chunk = chunk.dropna(subset=["lat", "lon"])
        chunk = chunk[chunk["lat"].between(33.0, 38.7) & chunk["lon"].between(124.0, 131.5)].copy()
        if chunk.empty:
            continue

        chunk = classify_shop(chunk)
        shop_coords = np.deg2rad(chunk[["lat", "lon"]].to_numpy(dtype=float))
        tree = BallTree(shop_coords, metric="haversine")
        for radius in RADII_KM:
            suffix = str(radius).replace(".", "_")
            ind = tree.query_radius(coords_rad, r=radius / EARTH_RADIUS_KM)
            result[f"radius_{suffix}km_all_shops"] += np.fromiter((len(i) for i in ind), dtype=int, count=len(ind))
            for flag, name in [("is_education", "education"), ("is_kids", "kids"), ("is_medical", "medical")]:
                values = chunk[flag].to_numpy()
                counts = np.fromiter((int(values[i].sum()) if len(i) else 0 for i in ind), dtype=int, count=len(ind))
                result[f"radius_{suffix}km_{name}_shops"] += counts
        print(f"processed shop chunks: {chunk_idx}", flush=True)

    return result


def main() -> int:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    features = count_radius_features()
    features.to_csv(PROCESSED / "school_radius_commercial_features.csv", index=False, encoding="utf-8-sig")
    summary = features.describe().T.reset_index().rename(columns={"index": "feature"})
    summary.to_csv(REPORTS / "school_radius_commercial_feature_summary.csv", index=False, encoding="utf-8-sig")
    print("saved:", PROCESSED / "school_radius_commercial_features.csv")
    print("rows:", len(features))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

