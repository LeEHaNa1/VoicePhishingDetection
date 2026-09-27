from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split

from Simulator.main import get_or_generate_final_dataset, load_tuned_hyperparams
from Simulator.Detection.train_eval import (
    cast_object_columns,
    fit_calibrator,
    prepare_categorical,
    split_data,
    train_model,
)
from Simulator.schema_utils import get_feature_columns
from Simulator.schema import Track

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_OUT_DIR = _PROJECT_ROOT / "calibration_results"


def compute_ece(y_true, proba, n_bins=10):
    """불균형 데이터의 표본 결핍 왜곡을 방지하기 위해 quantile 전략 적용"""
    # 1. 각 bin에 표본이 균등하게 분배되도록 quantile 구간 분할 적용
    try:
        prob_true, prob_pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
    except ValueError:
        prob_true, prob_pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy="uniform")
    
    # 2. 동일 비중 가중치로 ECE 계산
    weights = np.ones_like(prob_pred) / len(prob_pred)
    ece = np.sum(weights * np.abs(prob_true - prob_pred))
    
    if ece < 0.005:
        status = "최적 정렬 (Well-Calibrated: 오차 소멸)"
    else:
        gap_mean = np.mean(prob_pred - prob_true)
        status = "과신 (Overconfidence)" if gap_mean > 0 else "과소신 (Underconfidence)"
        
    return float(ece), status, prob_true, prob_pred


def run_confidence_analysis():
    # 1. 데이터 로드 및 전처리
    df = get_or_generate_final_dataset()
    df = prepare_categorical(df)
    train_set, test_set = split_data(df)

    # 보정용 데이터셋 분할 (8:2 계층화 분할)
    train_sub, val_calib = train_test_split(
        train_set,
        test_size=0.2,
        random_state=42,
        stratify=train_set["is_phishing"]
    )

    # 2. Track C 피처 기준 로드 (Track A 통신사 망 제외)
    feature_cols = get_feature_columns(track_filter=[Track.DEVICE, Track.DEVICE_STRUCTURAL])
    model_type, hyperparams = load_tuned_hyperparams()
    model_type = model_type or "XGBoost"

    # 모델 학습 (train_sub로 학습)
    model = train_model(train_sub, feature_cols=feature_cols, hyperparams=hyperparams, model_type=model_type)

    # Isotonic 보정기 적합 (val_calib로 적합)
    calibrator = fit_calibrator(model, val_calib, feature_cols=feature_cols, method="isotonic")

    # 3. Test 세트 예측 확률 산출
    X_test = test_set[feature_cols].copy()
    cat_cols = X_test.select_dtypes(include="object").columns
    X_test[cat_cols] = X_test[cat_cols].astype("category")
    y_test = test_set["is_phishing"].to_numpy()

    # (1) 보정 전 원본 확률 (불균형 가중치로 팽창된 점수)
    raw_proba = model.predict_proba(X_test)[:, 1]
    
    # (2) Isotonic 보정 후 사후 확률 (0~100% 자연 발생 확률 매핑)
    calib_proba = calibrator.predict_proba(X_test)[:, 1]

    # 4. ECE 수치 및 과신/과소신 진단
    raw_ece, raw_status, raw_true, raw_pred = compute_ece(y_test, raw_proba, n_bins=10)
    calib_ece, calib_status, calib_true, calib_pred = compute_ece(y_test, calib_proba, n_bins=10)

    print("\n" + "=" * 55)
    print("📊 [모델 확신도(Confidence) & 정밀 진단 결과 (Track C)]")
    print("=" * 55)
    print(f"1. 보정 전 (Raw XGBoost)     : ECE = {raw_ece:.4f} | 상태: {raw_status}")
    print(f"2. 보정 후 (Isotonic Calib)  : ECE = {calib_ece:.4f} | 상태: {calib_status}")
    print("=" * 55)

    # 5. Reliability Diagram (신뢰도 다이어그램 시각화)
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    
    # 대각선 (이상적인 보정선 y = x)
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (y=x)", alpha=0.7)
    
    # 보정 전 곡선
    plt.plot(raw_pred, raw_true, "s-", color="crimson", label=f"Before Calib (ECE: {raw_ece:.4f})")
    
    # 보정 후 곡선
    plt.plot(calib_pred, calib_true, "o-", color="royalblue", label=f"After Calib (ECE: {calib_ece:.4f})")
    
    plt.title("[Track C] Reliability Diagram: Over/Under-confidence Check", fontsize=12, pad=12)
    plt.xlabel("Mean Predicted Probability (Confidence / 확신도)", fontsize=10)
    plt.ylabel("Fraction of Positives (Empirical Accuracy / 실제 피싱 비율)", fontsize=10)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = _OUT_DIR / "reliability_diagram.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"\n신뢰도 곡선 그래프 저장 완료: {out_path}")


if __name__ == "__main__":
    run_confidence_analysis()