# 분할->학습->평가 모두 한 번에 처리하는 파일.
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    auc,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from Simulator.schema_utils import get_feature_columns, get_non_feature_columns


def prepare_categorical(df):
    # XGBoost 학습/평가용 dtype 준비: 문자열(범주형) 컬럼을 category dtype으로 변환.
    # train/val/test, K-Fold 등으로 나뉘기 "전" df 전체에 반드시 한 번만 호출해야 함.
    # 분할 후 조각마다 따로 astype("category")하면 조각끼리 카테고리 코드북이 달라질 수 있어서
    # (예: 특정 fold에 특정 범주가 우연히 하나도 없으면) XGBoost predict()에서
    # "category not in the training set" 에러가 남. 슬라이싱(.iloc[])은 카테고리 목록을
    # 새로 계산하지 않고 그대로 물려받으므로, 분할 전에 한 번만 하면 모든 조각이 동일한
    # 카테고리 코드북을 공유하게 됨.
    df = df.copy()
    df, _ = cast_object_columns(df)
    return df


# evaluate()가 돌려주는 스칼라 지표 키. 요약·비교 JSON은 이 순서 그대로 저장한다.
SCALAR_METRIC_KEYS = (
    "threshold", "accuracy", "precision", "recall", "f1",
    "PR-AUC", "Lift@Top5%", "Lift@Top10%", "Lift@Top20%",
)

def cast_object_columns(frame):
    """object dtype 컬럼만 category로 바꾼다. 이미 category면 대상이 없어 값이 그대로다.

    두 번째 반환값은 변환 전에 object였던 컬럼이다.
    train에서 본 코드북을 val에 맞출 때 이 목록을 쓴다.
    """
    object_cols = frame.select_dtypes(include="object").columns
    frame[object_cols] = frame[object_cols].astype("category")
    return frame, object_cols


def _resolve_feature_cols(feature_cols):
    # feature_cols 존재하는 경우: Track 시나리오(Track A,B,C)나 ablation용 커스텀 컬럼 목록이 넘어온 경우 -> 그대로 사용
    # feature_cols 존재하지 않는 경우: 민감도 분석 진행 -> 전체 feature 모두 사용
    # 방어적 필터: is_feature=False인 컬럼(phone_number, call_time 등)만 걸러냄("SCHEMA 등록
    # 여부"가 아니라 "is_feature=False 여부"만 봄) -> SCHEMA에 아직 없는 파생 feature 후보
    # (ablation 검증용)도 df에 실제 컬럼으로 존재하기만 하면 정상적으로 학습에 쓸 수 있음.
    banned_cols = set(get_non_feature_columns())
    cols = feature_cols or get_feature_columns()
    return [c for c in cols if c not in banned_cols]


def split_data(df):
    # dataframe을 train, test set으로 분할하는 함수.
    # train: 70%  test: 30% 으로 분할.
    # dataset 자체를 분할하기 때문에, get_feature_columns() 등의 schema_utils 함수는 호출하지 않음.
    train, test = train_test_split(
        df,
        test_size=0.3,  # 30%를 test set으로 분할.
        random_state=42,  # 재현성 확보를 위해 random_state 고정.(값이 중요한 게 아님. 동일 값을 사용하는 게 중요.)
        stratify=df["is_phishing"],
    )
    return train, test


def pr_auc(y, proba):
    # Precision-Recall AUC (사다리꼴 적분, precision_recall_curve 기반).
    # evaluate()의 기본값(pr_auc_method="trapezoidal")이 이 함수를 씀 -> 민감도 분석
    # (sensitivity_analysis.py)이 여기 해당. "상대적 우열"만 가리면 되는 비교라 사다리꼴 방식으로도 충분하다고 판단.
    p, r, _ = precision_recall_curve(y, proba)
    return float(auc(r, p))


def lift_at_top_k(y, proba, k=0.05):
    """
    상위 k% 표본 내 피싱 농축 배수(Lift) 계산
    - y: 실제 피싱 여부 라벨 (0 또는 1)
    - proba: 모델이 예측한 피싱 확률
    - k: 상위 표본 비율 (0.05=상위 5%, 0.1=상위 10%, 0.2=상위 20%)
    """
    y_arr = np.array(y)
    cutoff = max(int(np.ceil(len(y_arr) * k)), 1)

    # 예측 확률 기준 내림차순 정렬 후 상위 k% 인덱스 추출
    top_indices = np.argsort(proba)[::-1][:cutoff]

    # 상위 k% 내 피싱 적중률 / 전체 자연 피싱 발생률
    top_rate = np.mean(y_arr[top_indices])
    base_rate = np.mean(y_arr)

    lift = top_rate / base_rate if base_rate > 0 else 0.0
    return round(float(lift), 2)


def train_model(df, val=None, feature_cols=None, hyperparams=None, model_type="XGBoost"):
    # 학습을 위한 함수. model_type="XGBoost" 또는 "LightGBM".
    # val: 선택적 검증셋(조기종료용). 현재는 main.py/sensitivity_analysis.py 둘 다
    # val 없이 호출함(최종 모드도 train/test 2분할만 씀) -> 아래 val 분기는 지금 호출
    # 경로에서는 안 타지만, 나중에 조기종료가 다시 필요해지면 val을 넘기기만 하면 되도록 남겨둠.
    # feature_cols: 학습에 사용할 feature 컬럼 목록. Track 시나리오별로 다르게 넘어옴. 민감도 분석에서는 전체 feature 사용.
    # hyperparams: Optuna로 찾은 하이퍼파라미터 dict(optuna_apply.py::run_optuna_tuning()의
    # "tuned_hyperparams" 참고). 넘기면 아래 기본값 위에 덮어씀(n_estimators/scale_pos_weight
    # 포함 전부 override 가능). None이면 기존 기본값 그대로 사용(하위 호환).
    feature_cols = _resolve_feature_cols(feature_cols)
    X, categorical_cols = cast_object_columns(df[feature_cols].copy())
    y = df["is_phishing"]

    # 실제 category dtype 변환은 prepare_categorical()이 분할 전에 이미 끝내둠(fold 간
    # 카테고리 코드북을 통일하기 위함). 여기 있는 건 그걸 거치지 않고 바로 호출된 경우를 위한
    # 방어용 fallback일 뿐 -> object 컬럼이 남아있을 때만 동작(보통은 이미 없어서 no-op).
    # 주의: 이 fallback은 dtype이 아예 안 맞는 경우만 막아주고, prepare_categorical() 없이
    # train_fold/val_fold를 따로 변환하면 fold 간 카테고리 코드북이 어긋나는 문제(원래 버그)는 못 막음.

    # 클래스 불균형(피싱 1% vs 정상 99%) 보정.
    # scale_pos_weight = 음성 개수 / 양성 개수 -> 양성(피싱) 오분류에 더 큰 패널티를 줌.
    # config.CLASS_IMBALANCE(고정값)가 아니라 실제 df에서 계산 -> Track/fold마다 표본이 달라져도 항상 정확한 비율 반영.
    neg, pos = int((y == 0).sum()), int((y == 1).sum())
    scale_pos_weight = neg / pos if pos > 0 else 1

    if model_type == "XGBoost":
        params = dict(
            n_estimators=300,       # 트리 개수. 아래 학습률을 낮춘 만큼 넉넉하게 잡고 조기종료/고정 반복으로 제어.
            max_depth=4,            # 얕은 트리. 양성 표본이 적어(전체 1%) 트리가 깊으면 소수 양성 사례에 과적합하기 쉬움.
            learning_rate=0.05,     # 낮은 학습률 -> 한 트리가 과도하게 영향력을 갖지 않도록 완만하게 학습.
            scale_pos_weight=scale_pos_weight,  # 클래스 불균형 보정(위에서 계산). hyperparams가 덮어쓸 수도 있음.
            eval_metric="aucpr",    # PR-AUC. 극단적 불균형에서 accuracy/plain AUC보다 양성 클래스 성능을 잘 반영.
            tree_method="hist",     # 히스토그램 기반 분할 탐색 + 카테고리 dtype 지원에 필요.
            enable_categorical=True,  # category dtype 컬럼(number_type 등)을 원-핫 없이 직접 학습.
            random_state=42,        # 재현성 고정(다른 seed들과 마찬가지로 값 자체보다 고정 여부가 중요).
        )
        if hyperparams:
            params.update(hyperparams)  # Optuna 튜닝 결과로 기본값 override
        model = xgb.XGBClassifier(**params)
    elif model_type == "LightGBM":
        params = dict(
            objective="binary",
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=-1,          # LightGBM 자체 로그 억제(학습 결과와는 무관)
            # category dtype 컬럼(number_type 등)은 XGBoost와 달리 별도 플래그 없이
            # pandas category dtype을 자동 인식함.
        )
        if hyperparams:
            params.update(hyperparams)
        model = lgb.LGBMClassifier(**params)
    else:
        raise ValueError(f"알 수 없는 model_type: {model_type!r}. 'XGBoost' 또는 'LightGBM'이어야 함.")

    if val is not None:
        # val이 넘어온 경우(현재 호출 경로에서는 안 씀): 검증셋으로 조기종료 ->
        # 검증 성능이 20라운드 연속 개선 안 되면 멈춤(과적합 방지, 학습 시간 단축).
        # X_val에서 카테고리를 독립적으로 다시 계산하지 않고, X와 완전히 동일한
        # category dtype(코드북)을 그대로 강제 적용 -> val이 prepare_categorical()을
        # 거치지 않았거나 train과 독립적으로 변환돼도 fold 간 카테고리 불일치가 생기지 않음.
        X_val = val[feature_cols].copy()
        for col in categorical_cols:
            X_val[col] = X_val[col].astype(X[col].dtype)
        y_val = val["is_phishing"]

        if model_type == "XGBoost":
            model.set_params(early_stopping_rounds=20)
            model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
        else:  # LightGBM: set_params(early_stopping_rounds=...) 방식이 아니라 callbacks로 지정
            model.fit(
                X, y, eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
            )
    else:
        # 현재 기본 경로(최종 모드/K-Fold 모두): val 없이 n_estimators 고정 학습.
        # max_depth=4/learning_rate=0.05처럼 이미 보수적인 하이퍼파라미터로
        # 과적합을 억제하고 있어서, 조기종료 없이도 감당 가능하다고 판단.
        model.fit(X, y)

    return model


def evaluate(model, df, feature_cols=None, threshold_df=None, pr_auc_method="trapezoidal", calibrator=None):
    # 평가를 위한 함수.
    # accuracy / precision / recall / F1 / confusion_matrix / feature importance 반환.
    # feature_cols: train_model()과 동일한 컬럼 목록을 넘겨야 함(Track 시나리오 일치 필요).
    # threshold_df: threshold 탐색용 데이터(보통 train). None이면 df에서 탐색(낙관 편향 가능).
    # pr_auc_method: "trapezoidal"(기본, pr_auc()=precision_recall_curve+auc 사다리꼴 적분) 또는
    #                "average_precision"(average_precision_score, 직선 보간이 없어 모델/조합 간 비교에 더 적합).
    #                기본값은 sensitivity_analysis.py(상대적 우열 비교)를 그대로 둔 채, feature_ablation.py의
    #                파생 feature 비교와 main.py::run_final()의 최종 모델 비교에서만 "average_precision"을 명시.
    # calibrator: fit_calibrator()가 반환한 CalibratedClassifierCV(선택). 넘기면 model.predict_proba()
    #             대신 calibrator.predict_proba()로 확률을 뽑음(threshold 탐색·평가 양쪽 다 동일하게 적용).
    #             None(기본값)이면 기존과 완전히 동일하게 동작(하위 호환) -> calibration 적용 전/후를
    #             같은 함수로 그대로 비교 가능(main.py::run_final(use_calibration=...) 참고).
    if pr_auc_method not in ("trapezoidal", "average_precision"):
        raise ValueError(f"알 수 없는 pr_auc_method: {pr_auc_method!r}. 'trapezoidal' 또는 'average_precision'이어야 함.")
    feature_cols = _resolve_feature_cols(feature_cols)

    # train_model()과 동일하게, 실제 category dtype 변환은 prepare_categorical()이 분할 전에
    # 이미 끝내둠. 여기 있는 건 그걸 거치지 않고 바로 호출된 경우를 위한 방어용 fallback일 뿐
    # (보통은 이미 category라 no-op) -> fold 간 카테고리 코드북 불일치 문제는 못 막으니
    # 반드시 prepare_categorical()을 먼저 거쳐야 함.
    #   -> fallback은 _prepare_xy()로 옮겨 eval/threshold_df 양쪽에 공통 적용.
    # 목적: model.predict()의 기본 threshold(0.5)는 극단적 클래스 불균형(피싱 1% vs 정상 99%)
    # 데이터에서는 최적이 아닐 수 있음(양성 확률이 0.5를 잘 못 넘어서 recall이 과도하게 낮게 나올 수 있음).
    # 그래서 확률(proba)만 뽑아서, precision-recall curve 상에서 F1이 최대가 되는 threshold를 직접 탐색해 적용함.
    #   -> 탐색은 threshold_df(없으면 df), 적용은 평가셋 df. thresholds가 비면 0.5로 fallback.
    # 주의(한계): 지금은 별도의 검증셋이 없어서(run_final()도 train/test 2분할만 씀),
    # threshold를 "평가 대상 df 자기 자신"에서 찾는다
    #   -> threshold_df(보통 train)에서 탐색 후 평가셋에 고정 적용. main/sensitivity는 분리 호출.
    #   threshold_df=None이면 예전처럼 df에서 탐색해 낙관 편향 가능.
    # 진짜 편향 없는 평가를 원하면 threshold 탐색은 별도 검증셋에서, 최종 성능 측정은
    # test set에서 하도록 분리해야 함.
    #   -> train 탐색 / test·val 평가는 반영됨. 별도 val 3분할은 아직 없음.
    def _prepare_xy(frame):
        # prepare_categorical() 이후라면 보통 no-op. 직접 호출된 경우만 object→category.
        X_part, _ = cast_object_columns(frame[feature_cols].copy())
        return X_part, frame["is_phishing"]

    X, y = _prepare_xy(df)

    thr_frame = threshold_df if threshold_df is not None else df
    X_thr, y_thr = _prepare_xy(thr_frame)
    # calibrator가 있으면 보정된 확률을, 없으면 원본 model 확률을 씀(threshold 탐색·평가 동일 기준 적용).
    proba_thr = calibrator.predict_proba(X_thr)[:, 1] if calibrator is not None else model.predict_proba(X_thr)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_thr, proba_thr)
    # precision_recall_curve는 precision/recall을 thresholds보다 1개 더 많이 반환함
    # (마지막 지점은 threshold 없이 recall=0 지점) -> 그 마지막 지점은 탐색에서 제외.
    if len(thresholds) == 0:
        # 양성 표본이 없거나 curve가 비면 기본값 0.5 (argmax 크래시 방지)
        # 민감도 K-Fold에서 fold마다 양성이 매우 적을 때 위험
        best_threshold = 0.5
    else:
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
        best_threshold = float(thresholds[f1_scores.argmax()])

    proba = calibrator.predict_proba(X)[:, 1] if calibrator is not None else model.predict_proba(X)[:, 1]  # 평가셋 피싱(1) 확률
    pred = (proba >= best_threshold).astype(int)

    lift_5p = lift_at_top_k(y, proba, k=0.05)
    lift_10p = lift_at_top_k(y, proba, k=0.1)
    lift_20p = lift_at_top_k(y, proba, k=0.2)

    return {
        "threshold": best_threshold,  # 이번 평가에 실제로 적용된 threshold(참고/재현용)
        # --- fold 집계용 스칼라 지표 ---
        "accuracy": accuracy_score(y, pred),
        # zero_division=0: 한 클래스만 예측될 때 경고/에러 대신 0으로 처리
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y, pred),
        "classification_report": classification_report(y, pred, zero_division=0),
        # feature명 → 중요도. XGBoost 기본(gain 기반) importance
        "feature_importance": dict(zip(feature_cols, model.feature_importances_)),
        # PR-AUC: precision-recall auc로 베이스라인 대비 압도적으로 높다는 것을 보임()
        "PR-AUC": pr_auc(y, proba) if pr_auc_method == "trapezoidal" else float(average_precision_score(y, proba)),
        # Top-K : 상위층에서 흔들림 없이 잘 잡아주는지.
        "Lift@Top5%": lift_5p,
        "Lift@Top10%": lift_10p,
        "Lift@Top20%": lift_20p,
    }


def fit_calibrator(model, val_df, feature_cols, target_col="is_phishing", method="isotonic"):
    # scale_pos_weight 보정 때문에 왜곡된 model 확률을, 실제 발생 빈도에 가까운 확률로
    # 재정렬하는 보정기(CalibratedClassifierCV)를 적합. model은 이미 학습이 끝난 상태여야 하고,
    # val_df는 model 학습에 쓰이지 않은 별도 데이터여야 함(같은 데이터로 보정하면 model의
    # 과적합까지 "잘 보정된 것"처럼 왜곡됨).
    # method: "isotonic"(비모수, 표본이 충분할 때 권장) 또는 "sigmoid"(Platt scaling, 표본이 적을 때).
    feature_cols = _resolve_feature_cols(feature_cols)
    X_val, _ = cast_object_columns(val_df[feature_cols].copy())
    y_val = val_df[target_col]

    try:
        # scikit-learn 1.6+ : FrozenEstimator로 이미 학습된 model을 그대로 감싸서 재학습 없이 보정.
        from sklearn.frozen import FrozenEstimator
        calibrated_model = CalibratedClassifierCV(estimator=FrozenEstimator(model), method=method, cv=None)
    except ImportError:
        # scikit-learn <1.6(FrozenEstimator 없음): cv="prefit"이 "이미 학습된 model을 그대로 보정"
        # 이라는 뜻(1.6+에서 deprecated 되었을 뿐, 아직 동작은 함). 주의: 여기서 cv=None을 쓰면
        # "prefit 모델을 그대로 써라"가 아니라 "기본 K-fold로 model을 처음부터 다시 학습해라"는
        # 뜻이 되어버려(문서 기준) 완전히 다른(그리고 잘못된) 동작을 하게 됨 -> 반드시 "prefit"이어야 함.
        calibrated_model = CalibratedClassifierCV(estimator=model, method=method, cv="prefit")

    calibrated_model.fit(X_val, y_val)
    return calibrated_model
