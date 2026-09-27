# Simulator 코드 작성을 위한 main.py
# 역할: 전체 파이프라인의 진입점. mode에 따라 Generation과 Detection을 순서대로 호출.

import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from Simulator.Generation import config
from Simulator.Generation.dataset_builder import build_dataset
from Simulator.Detection.train_eval import (
    SCALAR_METRIC_KEYS,
    cast_object_columns,
    evaluate,
    fit_calibrator,
    prepare_categorical,
    split_data,
    train_model,
)
from Simulator.Detection.sensitivity_analysis import run_sensitivity, run_stress_sensitivity
from Simulator.Detection.feature_ablation import (
    run_feature_selection_pipeline,
    run_stability_check,
    CANDIDATE_DERIVED_FEATURES,
)
from Simulator.Detection.optuna_apply import run_optuna_tuning_and_compare
from Simulator.Detection.derived_features import add_candidate_features
from Simulator.schema import Track
from Simulator.schema_utils import get_feature_columns, get_schema_column_names

# 최종 dataset 저장 폴더: 실행 위치(cwd)와 무관하게 항상 프로젝트 루트 기준으로 고정.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATASET_DIR = _PROJECT_ROOT / "DataSet"
_TUNE_RESULTS_DIR = _PROJECT_ROOT / "optuna_results"


# 최종 모드 실행 파라미터. "이번 최종 실행을 어떻게 돌릴지"에 대한 파이프라인 설정값이라 main.py에 지역 상수로 둠.
N = 100000                  # 생성할 사건 수
SUBGROUP_RATIO_KEY = "D"   # 정상 이벤트 하위집단(지인/기관) 비중 키 (민감도 분석 결과 B->D로 확정)
GEN_SEED = 42              # build_dataset()용 random_state

# confidence calibration(isotonic) 관련 스위치 2개. 하나로 합쳐두면 "학습 데이터가 줄어드는 효과"와
# "calibration 자체 효과"가 뒤섞여서 비교가 안 되므로(run_final_calibration_compare() 참고),
# 독립적으로 켜고 끌 수 있게 분리함:
#   CALIBRATION_SPLIT_TRAIN: train_set을 train_sub/val_calib으로 나눌지(=학습 데이터를 줄일지) 여부
#   CALIBRATION_APPLY      : val_calib으로 calibrator를 적합해서 evaluate()에 넘길지 여부
#                            (True면 CALIBRATION_SPLIT_TRAIN도 반드시 True여야 함 -> run_final() 참고)
# 기본(True/True)은 현재 운영 중인 방식과 동일. run_final(split_for_calibration=..., apply_calibrator=...)
# 로 직접 override해서 세 버전(아래 A/B/C)을 나란히 비교할 수 있음(CLI -m final은 이 상수값을 그대로 씀).
CALIBRATION_SPLIT_TRAIN = True
CALIBRATION_APPLY = True
CALIBRATION_VAL_SIZE = 0.3  # train_set 중 val_calib(보정용)으로 떼어낼 비율

# run_final(compute_permutation=True)에서 permutation_importance() 반복 횟수(클수록 안정적이지만 느려짐).
# url_reliability_recheck.py의 PERM_N_REPEATS와 동일한 값 사용.
PERMUTATION_N_REPEATS = 10

# DataSet/final_dataset 캐시 무효화용 버전. N/GEN_SEED/SUBGROUP_RATIO_KEY/FINAL_SOPHISTICATION은
# 바뀌면 자동으로 감지되지만(fingerprint 비교), "생성 로직 자체"가 바뀌는 경우(코드는 바뀌었는데
# 위 파라미터 값은 그대로인 경우)는 자동으로 감지가 안 되므로, 그럴 때 이 숫자를 수동으로 올려야 dataset을 재사용하지 않고 강제로 다시 생성함.
# 2026-09-19: add_candidate_features()를 generate_final_dataset()에 연결(파생 feature 4개 포함)
#             하면서 1 -> 2로 올림 -> 기존(파생 feature 미포함) 캐시가 재사용되지 않고 새로 생성됨.
DATASET_GEN_VERSION = 2

# 최종 dataset 생성 시 각 feature에 적용할 sophistication. 
# 민감도 분석(필요 시 K=10/stress 모드로 재검증까지 거쳐) 결과로 확정한 값.
FINAL_SOPHISTICATION = {
    config.SOPH_NUMBER_TYPE_BAND: "mid",
    config.SOPH_FIRST_CONTACT: "high",
    config.SOPH_REPEAT_GAP: "low",           # K=10 재검증 완료
    config.SOPH_REPEAT_CONTACT: "mid",
    config.SOPH_NUM_IN_MSG: "mid",           # stress 모드로 재검증
    config.SOPH_INNER_NUM_DIFFERS: "mid",
    config.SOPH_MSG_OFFICIAL_MATCH: "high",
    config.SOPH_SMS_TO_CALL_GAP: "mid",      # K=10 재검증 완료
    config.SOPH_URL_RATE: "mid",             # stress 모드로 재검증
    config.SOPH_APP_INSTALL: "low",
    config.SOPH_SEQUENTIAL_CALLERS: "mid",
    config.SOPH_SMS_TO_CALL: "high",
}

# Track별 접근 가능 feature 시나리오.
# 최종 dataset 1개를 그대로 두고, 시나리오별로 사용하는 컬럼만 달라짐.
TRACK_SCENARIOS = {
   # "A": [Track.CARRIER, Track.DEVICE, Track.DEVICE_STRUCTURAL],  # 통신사: A+B+C 전부 접근 가능 (현재로써는 접근 가능한 데이터 X)
    "B": [Track.DEVICE],                                          # 단말: B만 접근 가능
    "C": [Track.DEVICE, Track.DEVICE_STRUCTURAL],                 # 단말+구조적신호: B+C 접근 가능
}


# 민감도 분석 대상 feature 목록.
SENSITIVITY_PARAMS = [
    # 하나
    config.SOPH_NUMBER_TYPE_BAND,
    config.SOPH_FIRST_CONTACT,
    config.SOPH_REPEAT_GAP,          # 정상 전용 (재연락 간격 배율/theta 크기)
    config.SOPH_REPEAT_CONTACT,      # 정상+피싱 공용 (사건 내 반복 접촉 존재확률)
    # 지민
    config.SOPH_NUM_IN_MSG,
    config.SOPH_INNER_NUM_DIFFERS,
    config.SOPH_MSG_OFFICIAL_MATCH,
    config.SOPH_SMS_TO_CALL_GAP,     # 정상 전용
    # 다은
    config.SOPH_URL_RATE,            # 정상 전용
    config.SOPH_APP_INSTALL,
    config.SOPH_SEQUENTIAL_CALLERS,
    config.SOPH_SMS_TO_CALL,         # 피싱 전용
]

# mode="each_sen"일 때 확인할 feature 1개. 바꾸고 싶으면 이 줄만 수정(config.SOPH_* 중 하나).
EACH_SENSITIVITY_PARAM = config.SOPH_SMS_TO_CALL_GAP

# mode="each_sen"일 때 쓸 K(K-fold). 
# 기본 5(표준). 일부 feature에 대해 재검증하려면 이 값만 10으로 수정 -> 기존 K=5 결과와 별도 저장됨.
EACH_SENSITIVITY_K = 5


# mode="stress" 대상 시나리오 목록. 
# config.py의 STRESS_*_KEY / STRESS_*_LEVELS 상수와 짝을 맞춰서 등록.
# "순수가정"으로 정한 feature의 절대값 자체가 틀렸을 때 성능이 얼마나 흔들리는지 확인하기 위한 것.
# low/mid/high 3-preset 비교인 SENSITIVITY_PARAMS와는 별개.
STRESS_SCENARIOS = {
    config.STRESS_NUM_IN_MSG_KEY: config.STRESS_NUM_IN_MSG_LEVELS,
    config.STRESS_URL_RATE_KEY: config.STRESS_URL_RATE_LEVELS,
}

# mode="stress"일 때 확인할 시나리오 1개. 바꾸고 싶으면 이 줄만 수정(STRESS_SCENARIOS의 key 중 하나).
STRESS_SCENARIO_KEY = config.STRESS_URL_RATE_KEY

# mode="stress"일 때 쓸 K(K-fold). 기본 5(표준).
STRESS_K = 5

# mode="ablation"일 때 쓸 다중공선성/데이터 누수 판정 임계값.
# feature_ablation.py의 기본값(0.8/0.95)을 그대로 쓰고 싶으면 None으로 둠 -> 바꾸고 싶을 때만 수정.
ABLATION_MULTICOLLINEARITY_THRESHOLD = None
ABLATION_LEAKAGE_THRESHOLD = None

# mode="ablation_stability"일 때 쓸 K(K-Fold). "애매한 단일 실행 결과 재검증" 성격이라
# 민감도분석 기본값(5)이 아니라 재검증 관례(K=10)를 따름.
ABLATION_STABILITY_K = 10

# mode="ablation_stability"에서 baseline과 비교할 feature 조합들.
# 2026-09-19 SCHEMA 연결 이후 최종 확정 4개(구 FINAL_CANDIDATE_FEATURES)는 이미 baseline_cols에
# 포함돼 있어서 더 이상 비교 대상으로 넣으면 안 됨(넣으면 컬럼 중복 -> run_stability_check()가
# ValueError로 막음). 그래서 여기서는 아직 SCHEMA 미등록인 나머지 9개(CANDIDATE_DERIVED_FEATURES,
# 1단계 선정에서 탈락/대기 중인 후보)를 하나의 묶음으로 재검증하도록 둠 -> 혹시 그중 일부가
# 단일 실행의 우연 때문에 탈락한 건 아닌지 K-Fold로 다시 확인하는 용도.
ABLATION_STABILITY_FEATURE_SETS = {
    "remaining_candidates": CANDIDATE_DERIVED_FEATURES,
}

# mode="tune"일 때 Optuna trial 횟수 / validation 비율
# (train_set를 train_sub/val_sub로 나눠서, val_sub로만 탐색 평가 -> test_set은 안 건드림).
TUNE_N_TRIALS = 50
TUNE_VAL_SIZE = 0.2


# ------------------------------------------------------------
# 최종 dataset 캐싱 메커니즘 (generate_final_dataset / get_or_generate_final_dataset)
# ------------------------------------------------------------
# 문제: run_final()이 매번 build_dataset()으로 처음부터 다시 생성하면, N=100000 기준
#      몇 초씩 낭비됨. 그런데 GEN_SEED 등이 고정이라 매번 만들어도 "완전히 동일한" dataset이
#      나오니, 이미 저장된 파일이 있으면 해당 파일을 가져와서 사용하는 게 더 효율적임.
# 해결: 저장할 때(save_final_dataset) dataset과 함께 "이번에 쓴 생성 파라미터"를
#      final_dataset.meta.json에 지문(fingerprint)으로 같이 남김.
#      다음에 get_or_generate_final_dataset()이 불릴 때, 지금 설정의 지문과 저장된 지문을 비교해서:
#        - 지문이 같음 -> 파라미터가 하나도 안 바뀌었다는 뜻 -> parquet 파일을 그대로 읽어서 반환
#        - 지문이 다름(또는 파일이 아예 없음) -> N/시드/subgroup/sophistication 중 뭔가
#          바뀌었다는 뜻 -> generate_final_dataset()으로 새로 생성해서 덮어씀
# 한계: 이 방식은 "숫자로 바뀌는 파라미터"만 감지한다. 파생 feature를 여기 새로 연결하는
#      것처럼, 파라미터 값은 그대로인데 "생성 함수 안의 코드 자체"가 바뀌는 경우는 안 잡힌다 
#      -> 그럴 때는 DATASET_GEN_VERSION 숫자를 수동으로 올려서 지문을 강제로 바꿔야 재사용되지 않고 새로 생성된다.
# mode="generate"는 이 캐시를 무시하고 generate_final_dataset()을 직접 불러 항상 강제로
# 새로 만든다(예: 캐시가 있어도 일부러 재현성만 다시 확인하고 싶을 때).
def _dataset_fingerprint() -> dict:
    """지금 이 실행의 생성 파라미터 지문. get_or_generate_final_dataset()이 저장된 dataset을
    재사용해도 되는지 판단하는 기준으로 씀 -> N/GEN_SEED/SUBGROUP_RATIO_KEY/FINAL_SOPHISTICATION
    중 하나라도 달라지면 값이 달라져서 자동으로 재생성됨."""
    return {
        "version": DATASET_GEN_VERSION,
        "n": N,
        "gen_seed": GEN_SEED,
        "subgroup_ratio_key": SUBGROUP_RATIO_KEY,
        "sophistication": FINAL_SOPHISTICATION,
    }


def save_final_dataset(df, out_dir: Path = _DATASET_DIR) -> None:
    """생성 직후(카테고리 변환/분할 전)의 원본 dataset을 DataSet/ 폴더에 CSV+Parquet 둘 다로 저장.
    같은 이름으로 덮어씀(재실행 시 항상 최신 dataset만 유지, 버전별 파일 누적 안 함).
    이번 생성에 쓴 파라미터 지문도 final_dataset.meta.json으로 같이 저장 -> 나중에
    get_or_generate_final_dataset()이 재사용 가능 여부를 판단하는 데 씀."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "final_dataset.csv", index=False, encoding="utf-8-sig")  # utf-8-sig: Excel에서 한글 깨짐 방지
    df.to_parquet(out_dir / "final_dataset.parquet", index=False)  # dtype(카테고리 등) 그대로 보존
    with open(out_dir / "final_dataset.meta.json", "w", encoding="utf-8") as f:
        json.dump(_dataset_fingerprint(), f, ensure_ascii=False, indent=2)


def generate_final_dataset():
    """최종 확정한 config 값으로 dataset 1개를 새로 생성해서 DataSet/ 폴더에 저장하고 반환
    (기존 저장분이 있어도 무조건 다시 만듦 -> mode="generate"에서 "강제로 새로 만들고 싶을 때" 씀).
    GEN_SEED/FINAL_SOPHISTICATION/SUBGROUP_RATIO_KEY가 고정이라 몇 번을 다시 돌려도 완전히
    동일한 dataset이 나옴(재현성 보장). 평소엔 이 함수를 직접 부르기보다
    get_or_generate_final_dataset()을 쓰는 걸 권장(불필요한 재생성 방지)."""
    df = build_dataset(
        n=N,
        phishing_rate=config.CLASS_IMBALANCE,
        subgroup_ratio_key=SUBGROUP_RATIO_KEY,
        random_state=GEN_SEED,
        config=config,
        sophistication=FINAL_SOPHISTICATION,
    )
    df = add_candidate_features(df)  # 파생 feature 13개 후보 전부 계산 (원본 23개 + 파생 13개 + 라벨 2개 = 38개 column)

    # SCHEMA에 이름이 등록되어 있는 column 전부 반환(원본 23개 + 파생 4개 = 27개) + 라벨 별도로 추가(SCHEMA에 없으므로)
    keep_cols = get_schema_column_names() + ["is_phishing", "incident_type"]

    # keep_cols에 있는 컬럼만 남긴다. 목록에 없는 파생 후보는 여기서 빠진다.
    df = df[[c for c in keep_cols if c in df.columns]]
    save_final_dataset(df)  # Detection 쪽 가공(category 변환/분할) 전, 생성 직후의 원본을 저장
    return df


def get_or_generate_final_dataset():
    """DataSet/final_dataset.parquet + final_dataset.meta.json이 있고, 그 지문이 지금
    _dataset_fingerprint()와 완전히 같으면 저장된 dataset을 그대로 읽어서 반환(재생성 생략).
    하나라도 없거나 지문이 다르면(N/시드/sophistication 변경, 또는 DATASET_GEN_VERSION을
    수동으로 올린 경우) generate_final_dataset()으로 새로 생성."""
    parquet_path = _DATASET_DIR / "final_dataset.parquet"
    meta_path = _DATASET_DIR / "final_dataset.meta.json"

    if parquet_path.exists() and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            saved_fingerprint = json.load(f)
        if saved_fingerprint == _dataset_fingerprint():
            return pd.read_parquet(parquet_path)

    return generate_final_dataset()


def run_final(split_for_calibration=None, apply_calibrator=None, compute_permutation=False):
    """최종 모드: get_or_generate_final_dataset()으로 dataset 확보(저장된 게 있고 파라미터가
    그대로면 재사용, 아니면 새로 생성) -> split_data() 2분할(train/test) -> optuna_results/
    best_params.json이 있으면 그 모델 종류(XGBoost/LightGBM 중 승자)+하이퍼파라미터를 사용
    (없으면 train_model() 기본값=XGBoost) -> Track 시나리오(B/C)별로 feature만 다르게 골라
    train_model()/evaluate() 반복.

    split_for_calibration/apply_calibrator: 각각 None(기본)이면 모듈 상단의
    CALIBRATION_SPLIT_TRAIN/CALIBRATION_APPLY 상수를 따름. 명시적으로 넘기면 그 값으로 override.
    두 스위치를 독립적으로 조합하면 아래 세 버전을 만들 수 있음(둘 다 True인 조합이 현재 운영 방식):
      - A(원본)        : split_for_calibration=False, apply_calibrator=False
                         -> train_set 전체로 학습, threshold도 train_set에서 탐색, calibrator 없음.
      - B(데이터만 축소): split_for_calibration=True,  apply_calibrator=False
                         -> train_set을 train_sub/val_calib으로 나눠 train_sub로만 학습(C와
                         동일한 크기)하지만, val_calib으로 calibrator는 적합하지 않음.
      - C(calibration) : split_for_calibration=True,  apply_calibrator=True
                         -> B와 동일하게 학습하고, val_calib으로 적합한 calibrator를 evaluate()에
                         넘김(threshold 탐색·최종 평가 모두 calibrator를 통과한 확률을 씀).
    (split_for_calibration=False, apply_calibrator=True 조합은 calibrator를 적합할 val_calib이
    없어서 성립하지 않음 -> ValueError.)

    A/B/C 세 버전을 비교하면 "학습 데이터가 줄어든 효과"와 "calibration 자체 효과"를 분리해서
    볼 수 있음(비교1=B-A, 비교2=C-B) -> run_final_calibration_compare() 참고.

    compute_permutation(기본 False, -m final의 기존 속도/동작에 영향 없음): True면 Track별로
    test_set 기준 permutation_importance()를 추가 계산해서 evaluate() 결과 dict에
    "permutation_importance" 키로 붙임. calibrator가 있으면(버전 C) calibrator를, 없으면(버전 A/B)
    model을 estimator로 씀 -> B/C를 비교하면(둘 다 같은 model이므로) "calibration의 확률 재정렬이
    feature별 실질 기여도(순위 기반)를 얼마나 바꾸는가"만 순수하게 분리해서 볼 수 있음(gain은 B/C가
    거의 동일해서 이 비교가 안 됨 -> run_final_calibration_compare() 참고)."""
    if split_for_calibration is None:
        split_for_calibration = CALIBRATION_SPLIT_TRAIN
    if apply_calibrator is None:
        apply_calibrator = CALIBRATION_APPLY
    if apply_calibrator and not split_for_calibration:
        raise ValueError(
            "apply_calibrator=True이려면 split_for_calibration=True여야 함"
            "(calibrator를 적합할 val_calib이 필요)."
        )

    df = get_or_generate_final_dataset()

    df = prepare_categorical(df)  # train/test로 나뉘기 전에 category dtype 한 번만 확정

    train_set, test_set = split_data(df) # train/test 2분할

    if split_for_calibration:
        # calibrator 적합용 데이터를 train_set에서 별도로 떼어냄(model 학습에 안 쓰인 데이터여야
        # calibration이 model의 과적합까지 "잘 보정된 것"처럼 왜곡하지 않음).
        # apply_calibrator=False(버전 B)라도 이 분할 자체는 그대로 해야 A와 비교했을 때 "데이터 축소
        # 효과"만 순수하게 분리됨 -> val_calib은 만들어두되 calibrator 적합에만 안 씀.
        train_sub, val_calib = train_test_split(
            train_set,
            test_size=CALIBRATION_VAL_SIZE,
            random_state=42,
            stratify=train_set["is_phishing"],
        )
    else:
        # 데이터를 안 나누면 val_calib을 떼어낼 이유가 없음 -> train_set 전체를 그대로 학습에 씀
        # (calibration 적용 전 기존 동작과 완전히 동일 = 버전 A).
        train_sub, val_calib = train_set, None

    model_type, tuned_hyperparams = load_tuned_hyperparams()  # 없으면 (None, None) -> 기본값 사용
    model_type = model_type or "XGBoost"

    results = {}  # {시나리오명("B"/"C"): evaluate() 결과}
    for scenario_name, tracks in TRACK_SCENARIOS.items():
        # TRACK_SCENARIOS 개수만큼 반복. (Track별로 다른 모델 생성 및 평가 진행)
        cols = get_feature_columns(track_filter=tracks)

        model = train_model(train_sub, feature_cols=cols, hyperparams=tuned_hyperparams, model_type=model_type)

        calibrator = fit_calibrator(model, val_calib, feature_cols=cols, method="isotonic") if apply_calibrator else None

        # threshold는 train_sub에서 고르고, 점수는 test에 고정 적용 (낙관 편향 방지)
        # pr_auc_method="average_precision": Track B/C 최종 모델 비교이므로 직선 보간 편향이 없는
        # average_precision_score를 씀(sensitivity_analysis.py의 상대적 우열 비교와는 다른 기준).
        result = evaluate(
            model,
            test_set,
            feature_cols=cols,
            threshold_df=train_sub,
            pr_auc_method="average_precision",
            calibrator=calibrator,
        )

        if compute_permutation:
            # calibrator가 있으면(버전 C) calibrator.predict_proba(), 없으면(버전 A/B) model.predict_proba()
            # 기준으로 permutation importance를 계산 -> evaluate() 내부의 확률 산출 방식과 동일하게 맞춤.
            X_test = test_set[cols].copy()
            cat_cols = X_test.select_dtypes(include="object").columns
            X_test[cat_cols] = X_test[cat_cols].astype("category")
            y_test = test_set["is_phishing"]
            estimator = calibrator if calibrator is not None else model
            perm = permutation_importance(
                estimator, X_test, y_test, scoring="average_precision",
                n_repeats=PERMUTATION_N_REPEATS, random_state=42,
            )
            result["permutation_importance"] = {c: round(float(v), 6) for c, v in zip(cols, perm.importances_mean)}

        results[scenario_name] = result

    return results


_CALIBRATION_RESULTS_DIR = _PROJECT_ROOT / "calibration_results"

# evaluate() 반환값 중 JSON으로 그대로 저장 가능한 스칼라 지표만(confusion_matrix(ndarray)/
# classification_report(str)는 JSON에 못 넣거나 너무 장황해서 제외) -> sensitivity_analysis.py::
# summarize_fold_results()와 동일한 "스칼라만 저장" 컨벤션. feature_importance(gain)는 아래
# _version_summary()에서 별도로 붙임(diff 계산 대상은 아니라서 이 튜플에는 안 넣음).
_FINAL_SCALAR_METRIC_KEYS = SCALAR_METRIC_KEYS


def _scalar_metrics(result):
    return {k: result[k] for k in _FINAL_SCALAR_METRIC_KEYS}


def _version_summary(result):
    # 9개 스칼라 지표 + gain(feature_importance) + permutation_importance(compute_permutation=True로
    # run_final() 호출했을 때만 존재)를 함께 반환. 둘 다 diff 대상이 아니라 그대로 값만 저장:
    #  - gain: B/C는 같은 train_sub로 학습된 사실상 동일한 모델이라(calibration은 학습 후 확률만
    #    재정렬하는 후처리) 거의 동일하게 나오는 게 정상 -> 비교는 A vs B(학습 데이터가 줄어 트리가
    #    다르게 자랐는가)에서만 의미 있음.
    #  - permutation_importance: 반대로 B/C는 같은 model인데 estimator만 다름(model vs calibrator)
    #    이라, 비교는 B vs C(calibration의 확률 재정렬이 feature별 실질 기여도를 얼마나 바꾸는가)에서
    #    의미 있음.
    summary = _scalar_metrics(result)
    summary["gain_importance"] = {k: round(float(v), 6) for k, v in result["feature_importance"].items()}
    if "permutation_importance" in result:
        summary["permutation_importance"] = result["permutation_importance"]
    return summary


def run_final_calibration_compare():
    """run_final()을 A(원본)/B(데이터만 축소)/C(calibration 적용) 세 버전으로 실행해서 Track B/C별
    스칼라 지표를 비교. calibration_results/final_calibration_compare.json에 저장.

    A/B/C의 정확한 정의는 run_final()의 docstring 참고 — 요약하면:
      - A: train_set 100%로 학습, calibration 없음 (기존 방식)
      - B: train_sub(C와 동일 크기)로 학습, calibration은 적용 안 함
      - C: train_sub로 학습 + calibration 적용 (현재 운영 방식)

    off/on(A/C) 하나만 비교하면 "학습 데이터가 줄어든 효과"와 "calibration 자체 효과"가 뒤섞여서
    어느 쪽 때문에 지표가 달라졌는지 알 수 없음 -> 비교를 2단계로 나눔:
      - 비교1(B - A): calibration은 둘 다 없고 학습 데이터 크기(train_set 100% vs train_sub)만
        다름 -> 순수하게 "학습 데이터가 줄어든 효과"만 분리해서 봄.
      - 비교2(C - B): 학습 데이터(train_sub)는 완전히 동일하고 calibration 유무만 다름 ->
        순수하게 "calibration 자체 효과"만 분리해서 봄.
    (비교1 + 비교2 = A와 C를 그냥 통째로 비교했을 때의 전체 차이와 같음.)

    gain_importance/permutation_importance도 각 버전에 함께 저장됨(compute_permutation=True) ->
    해석 방향이 서로 반대이니 주의: gain은 A vs B에서, permutation_importance는 B vs C에서 비교하는
    게 의미 있음(각각 _version_summary()/모듈 상단 설명 참고)."""
    result_a = run_final(split_for_calibration=False, apply_calibrator=False, compute_permutation=True)
    result_b = run_final(split_for_calibration=True, apply_calibrator=False, compute_permutation=True)
    result_c = run_final(split_for_calibration=True, apply_calibrator=True, compute_permutation=True)

    tracks = {}
    for scenario_name in TRACK_SCENARIOS:
        a = _version_summary(result_a[scenario_name])
        b = _version_summary(result_b[scenario_name])
        c = _version_summary(result_c[scenario_name])
        diff1 = {k: round(b[k] - a[k], 4) for k in _FINAL_SCALAR_METRIC_KEYS}
        diff2 = {k: round(c[k] - b[k], 4) for k in _FINAL_SCALAR_METRIC_KEYS}
        tracks[scenario_name] = {
            "A_original(train_set 100%, calibration 없음)": a,
            "B_data_reduced_only(train_sub, calibration 없음)": b,
            "C_calibrated(train_sub, calibration 적용)": c,
            "비교1_diff(B-A, 학습 데이터 축소 효과)": diff1,
            "비교2_diff(C-B, calibration 자체 효과)": diff2,
        }

    payload = {
        "description": "run_final()을 A(원본)/B(데이터만 축소)/C(calibration 적용) 3버전으로 실행해 "
                        "'학습 데이터 축소 효과'와 'calibration 자체 효과'를 분리해서 비교",
        "calibration_val_size": CALIBRATION_VAL_SIZE,
        "설명": {
            "A": "split_for_calibration=False, apply_calibrator=False — train_set 100%로 학습, calibration 없음(기존 방식)",
            "B": "split_for_calibration=True, apply_calibrator=False — train_sub로 학습(C와 동일 크기), calibration은 적용 안 함",
            "C": "split_for_calibration=True, apply_calibrator=True — train_sub로 학습 + calibration 적용(현재 운영 방식)",
            "비교1(B-A)": "calibration은 둘 다 없고 학습 데이터 크기만 다름 -> 순수 '학습 데이터 축소' 효과",
            "비교2(C-B)": "학습 데이터(train_sub)는 동일하고 calibration 유무만 다름 -> 순수 'calibration 자체' 효과",
            "gain_importance 해석 주의": "B/C는 같은 train_sub로 학습된 사실상 동일한 모델(calibration은 "
                                        "학습 후 확률만 재정렬하는 후처리)이라 gain도 거의 동일하게 나옴 -> "
                                        "gain 비교는 A vs B(학습 데이터 축소가 트리 구조 자체를 바꾸는가)에서만 의미 있음.",
            "permutation_importance 해석 주의": "gain과 반대 방향 -> B는 model.predict_proba(), C는 "
                                                "calibrator.predict_proba() 기준으로 계산되므로, B vs C 비교가 "
                                                "'calibration의 확률 재정렬이 feature별 실질 기여도(순위 기반)를 "
                                                "얼마나 바꾸는가'를 보여줌. A vs B는 학습 데이터가 달라 모델 자체가 "
                                                "다르므로 이쪽은 참고용(순수 비교 아님).",
        },
        "tracks": tracks,
    }

    _CALIBRATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _CALIBRATION_RESULTS_DIR / "final_calibration_compare.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload


def run_sensitivity_mode():
    """민감도분석 모드: config.py 스윕 파라미터를 하나씩 바꿔가며 run_sensitivity() 반복 호출 -> 비교표 출력."""
    result = {}  # {param_name: run_sensitivity()의 반환값(summary 리스트)} 형태로 feature별 결과를 보관

    for param_name in SENSITIVITY_PARAMS:
        # run_sensitivity() 안에서 sensitivity_results/<param_name>.json으로 자동 저장됨.
        result[param_name] = run_sensitivity(
            param_name=param_name,
            candidate_values=["low", "mid", "high"],
            fixed_config=config,
            phishing_rate=config.CLASS_IMBALANCE,
            n=N,
            subgroup_ratio_key=SUBGROUP_RATIO_KEY,
        )

    result["subgroup_ratio_key"] = run_sensitivity_ratio_mode()

    return result


def run_sensitivity_ratio_mode():
    """지인:기관 하위집단 비중(RELATION_TYPE_RATIO, A~E)만 따로 민감도분석 진행.
    sophistication 기반이 아니라 build_dataset()의 subgroup_ratio_key 자체를 바꾸는 축이라
    SENSITIVITY_PARAMS(SOPH_* 스윕)와는 별개로 분리."""
    # run_sensitivity() 안에서 sensitivity_results/subgroup_ratio_key.json으로 자동 저장됨.
    summary = run_sensitivity(
        param_name="subgroup_ratio_key",
        candidate_values=list(config.RELATION_TYPE_RATIO.keys()),  # ["A", "B", "C", "D", "E"]
        fixed_config=config,
        sweep_target="subgroup_ratio_key",
        phishing_rate=config.CLASS_IMBALANCE,
        n=N,
    )
    return summary


def run_sensitivity_each_feature_mode(param_name, k=5):
    """각 feature에 대해 진행 가능하도록 하는 함수(run_sensitivity_mode()는 전체 feature에 대해서.)
    k: 기본 5(표준). 재검증 등으로 늘리고 싶으면(예: 10) 여기로 넘기면 됨 ->
       sensitivity_results/<param_name>_k<k>.json으로 기존 K=5 결과와 별도 저장됨."""
    # run_sensitivity() 안에서 sensitivity_results/<param_name>.json(또는 _k<k>.json)으로 자동 저장됨.
    summary = run_sensitivity(
        param_name=param_name,
        candidate_values=["low", "mid", "high"],
        fixed_config=config,
        phishing_rate=config.CLASS_IMBALANCE,
        n=N,
        subgroup_ratio_key=SUBGROUP_RATIO_KEY,
        k=k,
    )

    return summary


def run_stress_mode(scenario_key=None, k=None):
    """가정-파괴(stress) 모드: scenario_key 1개에 대해 run_stress_sensitivity() 호출.
    sophistication은 "mid" 고정, config.py의 STRESS_*_LEVELS에 정의된 level별로 확률/θ 절대값만 스윕."""
    if scenario_key is None:
        scenario_key = STRESS_SCENARIO_KEY
    if k is None:
        k = STRESS_K
    if scenario_key not in STRESS_SCENARIOS:
        raise ValueError(
            f"알 수 없는 stress scenario_key: {scenario_key!r}. STRESS_SCENARIOS 중 하나여야 함: {list(STRESS_SCENARIOS)}"
        )
    # run_stress_sensitivity() 안에서 sensitivity_results/<scenario_key>.json(또는 _k<k>.json)으로 자동 저장됨.
    summary = run_stress_sensitivity(
        scenario_key=scenario_key,
        level_overrides=STRESS_SCENARIOS[scenario_key],
        fixed_config=config,
        description=f"{scenario_key} 가정-파괴(stress) 검증",
        phishing_rate=config.CLASS_IMBALANCE,
        n=N,
        subgroup_ratio_key=SUBGROUP_RATIO_KEY,
        k=k,
    )
    return summary


def run_ablation_mode():
    """파생 feature 선정 파이프라인 모드.
    결과는 feature_selection_results/feature_selection_pipeline.json에 저장됨."""
    kwargs = dict(
        fixed_config=config,
        n=N,
        phishing_rate=config.CLASS_IMBALANCE,
        subgroup_ratio_key=SUBGROUP_RATIO_KEY,
        sophistication=FINAL_SOPHISTICATION,  # 실제 최종 dataset과 동일한 조건에서 검증되도록
    )

    # feature_ablation.py에 선언된 상수 값 이외의 값을 설정한 경우, 함께 넘김.
    if ABLATION_MULTICOLLINEARITY_THRESHOLD is not None:
        kwargs["multicollinearity_threshold"] = ABLATION_MULTICOLLINEARITY_THRESHOLD
    if ABLATION_LEAKAGE_THRESHOLD is not None:
        kwargs["leakage_threshold"] = ABLATION_LEAKAGE_THRESHOLD
    return run_feature_selection_pipeline(**kwargs)


def run_ablation_stability_mode():
    """파생 feature 안정성 재검증 모드: ABLATION_STABILITY_FEATURE_SETS에 담긴 feature 조합(들)이
    baseline 대비 보인 성능 차이가 "실제 효과"인지 "단일 실행의 우연(노이즈)"인지, StratifiedKFold
    (K회 반복)로 재검증. run_ablation_mode()와 달리 "무엇을 고를지"를 다시 정하지 않고, 이미 고른
    조합(들)의 성능만 같은 fold 분할 안에서 K번 비교함(feature_ablation.run_stability_check() 참고).
    주의: 조합에 넣는 feature는 SCHEMA에 아직 미등록(=baseline_cols에 없는) 것이어야 함 —
    이미 확정 등록된 feature를 넣으면 run_stability_check()가 컬럼 중복으로 ValueError를 냄.
    결과는 feature_stability_results/feature_stability_check.json에 저장됨."""
    return run_stability_check(
        fixed_config=config,
        feature_sets=ABLATION_STABILITY_FEATURE_SETS,
        k=ABLATION_STABILITY_K,
        n=N,
        phishing_rate=config.CLASS_IMBALANCE,
        subgroup_ratio_key=SUBGROUP_RATIO_KEY,
        sophistication=FINAL_SOPHISTICATION,  # 실제 최종 dataset과 동일한 조건에서 검증되도록
    )


def load_tuned_hyperparams(out_dir: Path = _TUNE_RESULTS_DIR):
    """optuna_results/best_params.json이 있으면 (model_type, tuned_hyperparams) 튜플을 반환,
    없으면 (None, None)(= train_model()이 기본 XGBoost 하이퍼파라미터를 그대로 씀).
    "winner" 필드(XGBoost/LightGBM 비교 결과)를 기준으로 이긴 쪽 하이퍼파라미터만 꺼내옴."""
    path = out_dir / "best_params.json"
    if not path.exists():
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    winner = payload.get("winner")
    if winner is None:
        return None, None
    return winner, payload[winner]["tuned_hyperparams"]


def run_tuning_mode():
    """Optuna 하이퍼파라미터 탐색 모드: get_or_generate_final_dataset()으로 dataset 확보
    -> train/test 분할(test는 안 건드림) -> train_set 안에서 다시 train_sub/val_sub 분할
    (feature_ablation.py와 동일한 방식) -> val_sub 기준으로 XGBoost/LightGBM 둘 다 Optuna 탐색
    후 더 나은 쪽(winner)을 선택.
    결과는 optuna_results/best_params.json에 저장되고, run_final()이 다음 실행부터
    이 파일을 자동으로 읽어서(winner 모델 + 그 하이퍼파라미터) 반영함."""
    df = get_or_generate_final_dataset()
    df = prepare_categorical(df)
    train_set, test_set = split_data(df)  # test_set은 여기서 전혀 안 씀

    train_sub, val_sub = train_test_split(
        train_set, test_size=TUNE_VAL_SIZE, random_state=42, stratify=train_set["is_phishing"]
    )

    feature_cols = get_feature_columns(track_filter=[Track.DEVICE, Track.DEVICE_STRUCTURAL])
    X_train, y_train = train_sub[feature_cols], train_sub["is_phishing"]
    X_val, y_val = val_sub[feature_cols], val_sub["is_phishing"]

    result = run_optuna_tuning_and_compare(X_train, y_train, X_val, y_val, n_trials=TUNE_N_TRIALS)

    _TUNE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_TUNE_RESULTS_DIR / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def main(mode: str, param_name=None, k=None, scenario_key=None):
    # param_name/k: mode="each_sen"일 때만 사용. scenario_key/k: mode="stress"일 때만 사용.
    # 둘 다 직접 Python으로 호출할 때만 넘기는 선택 인자이고, CLI(-m each_sen / -m stress)로는
    # 안 받음 -> 코드 상단의 EACH_SENSITIVITY_PARAM/EACH_SENSITIVITY_K, STRESS_SCENARIO_KEY/
    # STRESS_K를 바꿔서 확인할 feature(또는 시나리오)/K를 정함(안 넘기면 그 상수값을 그대로 씀).
    # mode="ablation"은 K-Fold가 아니라 train/test 고정 분할 기반이라 k를 받지 않음 ->
    # 임계값을 바꾸려면 ABLATION_MULTICOLLINEARITY_THRESHOLD/ABLATION_LEAKAGE_THRESHOLD를 수정.
    # mode="ablation_stability"는 반대로 K-Fold 기반(재검증 목적) -> K를 바꾸려면 ABLATION_STABILITY_K를 수정.
    if mode == "generate":
        result = generate_final_dataset()
    elif mode == "final":
        result = run_final()
    elif mode == "sen":
        result = run_sensitivity_mode()
    elif mode == "each_sen":
        if param_name is None:
            param_name = EACH_SENSITIVITY_PARAM
        if k is None:
            k = EACH_SENSITIVITY_K
        # param_name이 SENSITIVITY_PARAMS(유효한 SOPH_* 값 목록)에 있는지 방어적으로 한 번 더 확인.
        if param_name not in SENSITIVITY_PARAMS:
            raise ValueError(
                f"알 수 없는 param_name: {param_name!r}. SENSITIVITY_PARAMS 중 하나여야 함: {SENSITIVITY_PARAMS}"
            )
        result = run_sensitivity_each_feature_mode(param_name, k=k)
    elif mode == "ratio":
        result = run_sensitivity_ratio_mode()
    elif mode == "stress":
        result = run_stress_mode(scenario_key=scenario_key, k=k)
    elif mode == "ablation":
        result = run_ablation_mode()
    elif mode == "ablation_stability":
        result = run_ablation_stability_mode()
    elif mode == "tune":
        result = run_tuning_mode()
    elif mode == "calib_compare":
        result = run_final_calibration_compare()
    else:
        raise ValueError(f"알 수 없는 mode: {mode}")

    if mode == "generate":
        # result가 DataFrame(N=100000행)이라 그대로 print()하면 너무 장황함 -> 요약만 출력.
        print(f"dataset 생성 완료: shape={result.shape}, 저장 위치={_DATASET_DIR}")
    else:
        print(result)
    return result


# 실행 방법 (프로젝트 루트에서):
#   python -X utf8 -m Simulator.main              -> 기본값(mode=final)
#   python -X utf8 -m Simulator.main -m generate  -> dataset 생성 + DataSet/ 폴더 저장만(학습/평가 없음).
#                                                     final과 완전히 동일한 GEN_SEED/FINAL_SOPHISTICATION을
#                                                     쓰므로 나중에 -m final을 돌려도 100% 동일한 dataset이 나옴.
#   python -X utf8 -m Simulator.main -m final     -> 최종 dataset 생성 -> 학습 -> 평가 (Track B/C)
#   python -X utf8 -m Simulator.main -m sen       -> SENSITIVITY_PARAMS 전체 + subgroup_ratio_key 스윕
#                                                     (feature마다 sensitivity_results/<param_name>.json 자동 저장)
#   python -X utf8 -m Simulator.main -m each_sen  -> feature 1개만 스윕(위 EACH_SENSITIVITY_PARAM/EACH_SENSITIVITY_K 값 사용).
#                                                     확인할 feature나 K를 바꾸려면 CLI가 아니라 코드 상단의
#                                                     EACH_SENSITIVITY_PARAM/EACH_SENSITIVITY_K 줄을 직접 수정할 것.
#   python -X utf8 -m Simulator.main -m ratio     -> 지인:기관 하위집단 비중(RELATION_TYPE_RATIO, A~E)만 스윕
#   python -X utf8 -m Simulator.main -m stress    -> 가정-파괴(stress) 모드: sophistication은 "mid" 고정,
#                                                     STRESS_SCENARIO_KEY 시나리오 1개의 확률/θ 절대값만 스윕
#                                                     (위 STRESS_SCENARIO_KEY/STRESS_K 값 사용). 시나리오나 K를
#                                                     바꾸려면 CLI가 아니라 코드 상단의 두 줄을 직접 수정할 것.
#   python -X utf8 -m Simulator.main -m ablation  -> 파생 feature 선정 파이프라인: baseline(원본만) vs
#                                                     full(원본+파생 13개) 비교 -> train set 기준
#                                                     다중공선성 점검(+데이터 누수 플래그) -> SHAP 참고 +
#                                                     Permutation Importance 기반 가지치기 -> 최종 재학습/평가.
#                                                     feature_selection_results/feature_selection_pipeline.json에 저장.
#                                                     임계값을 바꾸려면 코드 상단의
#                                                     ABLATION_MULTICOLLINEARITY_THRESHOLD/
#                                                     ABLATION_LEAKAGE_THRESHOLD를 직접 수정할 것.
#   python -X utf8 -m Simulator.main -m ablation_stability
#                                                  -> 파생 feature 안정성 재검증: ABLATION_STABILITY_FEATURE_SETS에
#                                                     담긴 feature 조합(들)이 baseline 대비 보인 성능 차이가
#                                                     노이즈인지, 같은 fold 분할 안에서 StratifiedKFold(기본 K=10)로
#                                                     한 번에 재검증. "무엇을 고를지" 재결정이 아니라 이미 고른
#                                                     조합(들)의 성능만 K회 반복 비교(run_sensitivity()와 동일한
#                                                     K-Fold 패턴). 결과는 feature_stability_results/feature_stability_check.json에 저장.
#                                                     기본값은 아직 SCHEMA 미등록인 9개 후보(CANDIDATE_DERIVED_FEATURES)
#                                                     묶음 -> 이미 SCHEMA에 등록된(=baseline에 포함된) feature를
#                                                     조합에 넣으면 컬럼 중복으로 에러남. K나 비교할 조합을 바꾸려면
#                                                     코드 상단의 ABLATION_STABILITY_K/ABLATION_STABILITY_FEATURE_SETS를 직접 수정할 것.
#   python -X utf8 -m Simulator.main -m tune      -> Optuna로 XGBoost/LightGBM 둘 다 하이퍼파라미터
#                                                     탐색 후 val PR-AUC 더 높은 쪽(winner)을 선택.
#                                                     train_set 안에서 train_sub/val_sub로 나눠
#                                                     val_sub 기준으로 탐색(test_set은 안 건드림).
#                                                     결과는 optuna_results/best_params.json에 저장되고,
#                                                     이후 -m final을 돌리면 winner 모델+하이퍼파라미터가 자동 반영됨.
#                                                     trial 수를 바꾸려면 코드 상단의 TUNE_N_TRIALS 수정.
#   python -X utf8 -m Simulator.main -m calib_compare
#                                                  -> run_final()을 A(원본)/B(데이터만 축소)/C(calibration
#                                                     적용) 3버전으로 실행해서 Track B/C 스칼라 지표+gain+
#                                                     permutation_importance를 비교. calibration_results/
#                                                     final_calibration_compare.json에 저장. 비교1(B-A)=학습
#                                                     데이터 축소 효과, 비교2(C-B)=calibration 자체 효과로
#                                                     나눠서 봄(gain은 A vs B, permutation_importance는
#                                                     B vs C 비교가 의미 있음 -> run_final_calibration_compare() 참고).
#   -m은 --mode의 짧은 별칭.
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Voice Phishing Detection 시뮬레이터 파이프라인 실행")
    parser.add_argument(
        "-m", "--mode", default="final",
        choices=["generate", "final", "sen", "each_sen", "ratio", "stress", "ablation", "ablation_stability", "tune", "calib_compare"],
        help="실행 모드 (기본값: final)",
    )
    args = parser.parse_args()

    main(mode=args.mode)
