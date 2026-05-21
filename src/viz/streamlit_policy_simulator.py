from __future__ import annotations

from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from project_config import (
    RISK_COLORS,
    RISK_LABEL_KO,
    RISK_SCORE_WEIGHTS,
    assign_policy_risk_label,
    valid_sido_coord_mask,
)


ROOT = SRC.parent
SCENARIO_PATH = ROOT / "data" / "processed" / "final_national_school_scenario_2026_2040.csv"


@st.cache_data(show_spinner=False)
def load_scenario() -> pd.DataFrame:
    df = pd.read_csv(SCENARIO_PATH, low_memory=False)
    df["lttud"] = pd.to_numeric(df["lttud"], errors="coerce")
    df["lgtud"] = pd.to_numeric(df["lgtud"], errors="coerce")
    return df[valid_sido_coord_mask(df)].copy()


def weighted_score(df: pd.DataFrame, weights: dict[str, int]) -> pd.Series:
    return (
        df["pred_low_student_flag"] * weights["low_student"]
        + df["long_term_decline_flag"] * weights["long_decline"]
        + df["severe_decline_flag"] * weights["severe_decline"]
        + df["isolation_high_flag"] * weights["isolation"]
        + df["commercial_vulnerable_flag"] * weights["commercial"]
        + df["regional_decline_high_flag"] * weights["regional"]
        + df["objective_top10_flag"] * weights["edss"]
        + df["replacement_near_flag"] * weights["replacement"]
    )


def apply_policy(df: pd.DataFrame, weights: dict[str, int]) -> pd.DataFrame:
    out = df.copy()
    out["risk_score"] = weighted_score(out, weights)
    out["risk_label"] = out.apply(assign_policy_risk_label, axis=1)
    out["risk_label_ko"] = out["risk_label"].map(RISK_LABEL_KO).fillna(out["risk_label"])
    return out


def marker_popup(row: pd.Series) -> str:
    change = 0.0
    if row["student_count_2025"]:
        change = (row["forecast_student_count"] / row["student_count_2025"] - 1) * 100
    return f"""
    <b>{row['schlNm']}</b><br>
    {row['requested_sido_name']} / {row['sgg_code']} · {row['school_level']} · {int(row['forecast_year'])}년<br>
    2025 학생수: {int(row['student_count_2025'])}명<br>
    예측 학생수: {int(row['forecast_student_count'])}명 ({change:.1f}%)<br>
    위험등급: <b>{row['risk_label_ko']}</b><br>
    위험점수: {row['risk_score']:.0f}<br>
    고립도: {row['school_isolation_score']:.1f} · 상권취약도: {row['commercial_vulnerability_score']:.1f}<br>
    지역 학령수요 감소압력: {row['regional_decline_risk_score']:.1f}
    """


def build_map(df: pd.DataFrame) -> folium.Map:
    if df.empty:
        return folium.Map(location=[36.5, 127.8], zoom_start=7, tiles="CartoDB positron")
    center = [df["lttud"].mean(), df["lgtud"].mean()]
    fmap = folium.Map(location=center, zoom_start=9, tiles="CartoDB positron")
    plot_df = df.sort_values("risk_score", ascending=False).head(1500)
    for _, row in plot_df.iterrows():
        color = RISK_COLORS.get(row["risk_label"], "#64748b")
        folium.CircleMarker(
            location=[row["lttud"], row["lgtud"]],
            radius=5 if row["risk_score"] < 60 else 7,
            color="white",
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.86,
            popup=folium.Popup(marker_popup(row), max_width=360),
        ).add_to(fmap)
    return fmap


def main() -> None:
    st.set_page_config(page_title="통폐합 위험 What-If 시뮬레이터", layout="wide")
    st.title("통폐합 위험 What-If 정책 시뮬레이터")
    st.caption("슬라이더로 정책 가중치를 바꾸면 위험점수와 위험등급을 즉시 다시 계산합니다. 회귀 예측 학생수 자체는 바뀌지 않습니다.")

    df = load_scenario()
    with st.sidebar:
        st.header("필터")
        year = st.selectbox("연도", sorted(df["forecast_year"].dropna().astype(int).unique()), index=14)
        sidos = ["전체"] + sorted(df["requested_sido_name"].dropna().astype(str).unique())
        sido_default = sidos.index("충남") if "충남" in sidos else 0
        sido = st.selectbox("지역", sidos, index=sido_default)
        levels = ["전체"] + sorted(df["school_level"].dropna().astype(str).unique())
        level = st.selectbox("학교급", levels)

        st.header("정책 가중치")
        weights = {
            key: st.slider(label, 0, 40, int(default))
            for key, label, default in [
                ("low_student", "저학생수", RISK_SCORE_WEIGHTS["low_student"]),
                ("long_decline", "장기 감소", RISK_SCORE_WEIGHTS["long_decline"]),
                ("severe_decline", "급격 감소", RISK_SCORE_WEIGHTS["severe_decline"]),
                ("isolation", "학교 고립도", RISK_SCORE_WEIGHTS["isolation"]),
                ("commercial", "상권 취약도", RISK_SCORE_WEIGHTS["commercial"]),
                ("regional", "지역 학령수요 감소압력", RISK_SCORE_WEIGHTS["regional"]),
                ("edss", "EDSS 유사도", RISK_SCORE_WEIGHTS["edss"]),
                ("replacement", "대체학교 근접성", RISK_SCORE_WEIGHTS["replacement"]),
            ]
        }

    view = df[df["forecast_year"].eq(year)].copy()
    if sido != "전체":
        view = view[view["requested_sido_name"].eq(sido)].copy()
    if level != "전체":
        view = view[view["school_level"].eq(level)].copy()
    view = apply_policy(view, weights)

    counts = view["risk_label_ko"].value_counts()
    total_students = int(view["forecast_student_count"].fillna(0).sum())
    total_2025 = int(view["student_count_2025"].fillna(0).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("표시 학교", f"{len(view):,}개")
    c2.metric("선택 연도 예측 학생수", f"{total_students:,}명")
    c3.metric("2025 기준 학생수", f"{total_2025:,}명")
    c4.metric("통폐합 가능 고위험", f"{int(counts.get('통폐합 가능 고위험', 0)):,}개")

    map_col, table_col = st.columns([2.1, 1])
    with map_col:
        st_folium(build_map(view), height=720, use_container_width=True)
    with table_col:
        st.subheader("위험등급 분포")
        st.dataframe(counts.rename_axis("위험등급").reset_index(name="학교 수"), use_container_width=True)
        st.subheader("상위 위험 학교")
        cols = [
            "schlNm",
            "requested_sido_name",
            "sgg_code",
            "school_level",
            "forecast_student_count",
            "risk_label_ko",
            "risk_score",
        ]
        st.dataframe(
            view.sort_values(["risk_score", "forecast_student_count"], ascending=[False, True])[cols].head(30),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
