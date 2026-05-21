from __future__ import annotations

from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder


ROOT = SRC.parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "reports"
MODELS = ROOT / "outputs" / "models"


def train_policy_multiclass_model() -> Pipeline:
    # 1. 시나리오 데이터 로드 (수식에 의해 임시 생성된 라벨 포함)
    # 임시 라벨링된 데이터를 불러와 머신러닝의 훈련셋으로 활용
    scenario_path = PROCESSED / "final_national_school_scenario_2026_2040.csv"
    if not scenario_path.exists():
        # 만약 파일이 없는 경우, 변화량 버전 사용 시도
        scenario_path = PROCESSED / "final_national_school_scenario_change_model_2026_2040.csv"
    
    df = pd.read_csv(scenario_path, low_memory=False)
    
    # 2. 필수 변수 numeric 변환 및 필터링
    df = df[df["school_level"].isin(["초등학교", "중학교", "고등학교"])].copy()
    df = df.dropna(subset=["risk_label"]).copy()
    
    # Train/Test 분할 (2026~2035년: Train / 2036~2040년: Test)
    train = df[df["forecast_year"].between(2026, 2035)].copy()
    test = df[df["forecast_year"].between(2036, 2040)].copy()
    
    # 피처 정의
    feature_cols = [
        "requested_sido_name",
        "school_level",
        "foundation",
        "student_count_2025",
        "forecast_student_count",
        "population_pressure_ratio",
        "school_isolation_score",
        "commercial_vulnerability_score",
        "regional_decline_risk_score",
        "objective_closure_percentile",
        "nearest_same_level_school_km",
        "same_level_school_count_5km",
        "risk_score"
    ]
    
    categorical = ["requested_sido_name", "school_level", "foundation"]
    numeric = [col for col in feature_cols if col not in categorical]
    
    # 라벨 인코더 정의
    le = LabelEncoder()
    train_y = le.fit_transform(train["risk_label"])
    test_y = le.transform(test["risk_label"])
    
    # scikit-learn Pipeline 구축
    prep = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ]
    )
    
    # 다중 클래스 기계학습 모델 설정 (Gradient Boosting Classifier)
    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=300,
        max_leaf_nodes=41,
        l2_regularization=0.1,
        class_weight="balanced",
        random_state=42
    )
    
    pipe = Pipeline([("prep", prep), ("clf", model)])
    
    print(f"[Policy ML Model] Training Multiclass HistGradientBoostingClassifier on {len(train)} rows...")
    pipe.fit(train[feature_cols], train_y)
    
    # 검증 및 예측 성능 평가
    pred_y = pipe.predict(test[feature_cols])
    acc = accuracy_score(test_y, pred_y)
    macro_f1 = f1_score(test_y, pred_y, average="macro")
    
    print(f"[Policy ML Model] Validation Accuracy: {acc * 100:.2f}% | Macro F1-Score: {macro_f1:.4f}")
    
    # 평가지표 리포트 생성 및 저장
    report = classification_report(test_y, pred_y, target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(REPORTS / "policy_multiclass_classifier_metrics.csv", encoding="utf-8-sig")
    
    # 모델 및 라벨 인코더 직렬화 저장
    joblib.dump(pipe, MODELS / "tuned_histgb_policy_multiclass_classifier.pkl")
    joblib.dump(le, MODELS / "policy_multiclass_label_encoder.pkl")
    
    print(f"[Policy ML Model] Model and LabelEncoder saved to {MODELS}")
    return pipe


if __name__ == "__main__":
    train_policy_multiclass_model()
