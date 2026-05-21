from __future__ import annotations

from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import joblib
import numpy as np
import pandas as pd
from models.train_temporal_closure_classifier import build_temporal_dataset


ROOT = SRC.parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "reports"
MODELS = ROOT / "outputs" / "models"


def evaluate_topk():
    print("[Top-K Evaluation] Loading temporal dataset...")
    df = build_temporal_dataset()
    
    # 2022년을 기준년도(Base Year)로 설정하여 2023년의 폐교 여부를 예측
    # df['closed_next_year_proxy']는 2022년 데이터 중 2023년에 사라진 학교를 1로, 남아있는 학교를 0으로 마킹함
    df_2022 = df[df["base_year"] == 2022].copy()
    
    # 학제명 필터링 (초/중/고)
    df_2022 = df_2022[df_2022["학제명"].isin(["초등학교", "중학교", "고등학교"])].copy()
    
    total_schools = len(df_2022)
    actual_closures = int(df_2022["closed_next_year_proxy"].sum())
    
    print(f"[Top-K Evaluation] Total Active Schools in 2022: {total_schools:,}")
    print(f"[Top-K Evaluation] Actual Closures in 2023: {actual_closures} schools (Rate: {actual_closures/total_schools*100:.3f}%)")
    
    # 2. 저장된 훈련 모델(tuned_histgb_temporal_closure.pkl) 로드
    model_path = MODELS / "tuned_histgb_temporal_closure.pkl"
    if not model_path.exists():
        print(f"[ERROR] Model file not found at {model_path}")
        return
        
    model = joblib.load(model_path)
    
    feature_cols = [
        "sido_name",
        "학제명",
        "설립구분명",
        "student_count",
        "students_per_class",
        "students_per_teacher",
        "student_growth_1yr",
        "birth_count",
        "avg_total_fertility_rate",
        "avg_birth_count_yoy_rate"
    ]
    
    # 3. 2022년 데이터를 모델에 통과시켜 2023년 폐교 확률 추론
    print("[Top-K Evaluation] Predicting closure probabilities for 2023...")
    df_2022["pred_prob"] = model.predict_proba(df_2022[feature_cols])[:, 1]
    
    # 4. 예측 확률 기준 내림차순 정렬
    df_sorted = df_2022.sort_values("pred_prob", ascending=False).copy()
    
    # 5. Top-K 평가지표 계산 (K = 10, 30, 50, 100, 200, 300, 500)
    k_list = [10, 30, 50, 100, 200, 300, 500]
    results = []
    
    for k in k_list:
        top_k = df_sorted.head(k)
        hits = int(top_k["closed_next_year_proxy"].sum())
        recall = hits / actual_closures if actual_closures > 0 else 0.0
        precision = hits / k
        lift = precision / (actual_closures / total_schools) if actual_closures > 0 else 0.0
        
        results.append({
            "K": k,
            "Hits (Actual Closed)": hits,
            "Precision (Hit Rate)": f"{precision * 100:.2f}%",
            "Recall (Sensitivity)": f"{recall * 100:.2f}%",
            "Lift Factor": f"{lift:.2f}x"
        })
        
    results_df = pd.DataFrame(results)
    print("\n=================== TOP-K EVALUATION RESULTS (2022 -> 2023 Closures) ===================")
    print(results_df.to_string(index=False))
    print("========================================================================================\n")
    
    # 결과 파일 저장
    results_df.to_csv(REPORTS / "temporal_closure_topk_evaluation.csv", index=False, encoding="utf-8-sig")
    
    # 가장 높은 확률로 폐교를 예측한 Top 10 학교 샘플 출력 (보안 마스킹 상태)
    print("Top 10 High-Risk School Predictions:")
    top_10_samples = df_sorted.head(10)[["school_id", "sido_name", "학제명", "student_count", "student_growth_1yr", "pred_prob", "closed_next_year_proxy"]]
    print(top_10_samples.to_string(index=False))


if __name__ == "__main__":
    evaluate_topk()
