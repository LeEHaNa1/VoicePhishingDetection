# 민감도 분석 진행을 위한 파일
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import contextmanager

import numpy as np
from sklearn.model_selection import StratifiedKFold

from Simulator.Generation.dataset_builder import build_dataset
from Simulator.schema_utils import get_feature_columns
from Simulator.Detection.train_eval import SCALAR_METRIC_KEYS, evaluate, prepare_categorical, train_model
from Simulator.Detection.derived_features import add_candidate_features

# 결과 저장 폴더: 실행 위치(cwd)와 무관하게 항상 프로젝트 루트 기준으로 고정.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_RESULTS_DIR = _PROJECT_ROOT / "sensitivity_results"

# run_sensitivity()의 k 기본값과 동일하게 맞춤. save_sensitivity_result()가 파일명을
# 분기할 때 "기본 K로 돌린 결과인지"를 판단하는 기준으로도 재사용.
DEFAULT_K = 5


def summarize_fold_results(fold_results: List[dict]) -> dict:
    """K-Fold 결과(fold_results, evaluate() 반환값 K개) 안의 스칼라 지표들을
    평균/표준편차/표준오차로 요약. confusion_matrix/classification_report/feature_importance처럼
    스칼라가 아닌 건 저장 파일을 깔끔하게 유지하기 위해 요약에서 제외.
    sem(표준오차, std/sqrt(K)): fold를 늘렸을 때(K가 클수록) 평균이 더 정밀해지는 걸
    반영하는 값 -> 후보값 간 차이가 "노이즈보다 큰지" 판단할 땐 std보다 이 값을 기준으로 봐야 함."""
    scalar_keys = SCALAR_METRIC_KEYS
    k = len(fold_results)
    summary = {}
    for key in scalar_keys:
        values = [fr[key] for fr in fold_results if key in fr]
        if not values:
            continue
        std = float(np.std(values))
        summary[key] = {
            "mean": round(float(np.mean(values)), 4),
            "std": round(std, 4),
            "sem": round(std / (k ** 0.5), 4) if k > 0 else None,
        }
    return summary


def save_sensitivity_result(
    param_name: str, summary: List[dict], out_dir: Path = _RESULTS_DIR, meta: Optional[dict] = None
) -> Path:
    """민감도 분석 결과(후보값별 요약 통계)를 sensitivity_results/<param_name>.json으로 저장.
    같은 feature를 같은 K로 다시 실행하면 새 파일을 추가로 만들지 않고 덮어씀(항상 최신 결과만 유지).
    K가 DEFAULT_K(5)가 아니면(예: 재검증용 K=10) <param_name>_k<K>.json으로 따로 저장해서
    기존 K=5 결과를 안 건드림.
    meta: stress 모드처럼 결과 해석에 필요한 부가 정보(kind/description/levels 등)가 있을 때만 채워서
    payload에 그대로 얹음. sophistication 기반 run_sensitivity()는 넘기지 않으므로 기존 파일 포맷은 그대로 유지됨."""
    k = len(summary[0]["fold_results"]) if summary else DEFAULT_K
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "param_name": param_name,
        "results": [
            {"value": entry["value"], "summary": summarize_fold_results(entry["fold_results"])}
            for entry in summary
        ],
    }
    if meta:
        payload["meta"] = meta
    filename = f"{param_name}.json" if k == DEFAULT_K else f"{param_name}_k{k}.json"
    path = out_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def run_sensitivity(
    param_name: str,                    # 스윕 대상인 feature 이름. config.py 변수명이 아니라 get_soph() 조회 key  ex) "url_rate"
    candidate_values: List[str],        # param_name에 하나씩 대입해볼 값 목록. sweep_target="sophistication"이면 [LOW, MID, HIGH], "subgroup_ratio_key"면 ["A".."E"]
    fixed_config,                       # config.py 모듈. param_name 외 나머지 값은 전부 여기서 그대로(고정) 가져다 씀. Detection/ 안에서 config.py import 하지 않기 위한 parameter.
    sweep_target: str = "sophistication",  # 무엇을 스윕할지: "sophistication"(get_soph() 기반 feature) | "subgroup_ratio_key"(RELATION_TYPE_RATIO, A~E)
    base_sophistication: Optional[str] = None,  # param_name을 제외한 다른 feature들에 공통 적용할 sophistication. 스윕 대상인 feature 외의 다른 feature들이 LOW/MID/HIGH 중 어떤 값을 쓸지 결정. (보통, MID)
    n: int = 10000,                     # 후보값마다 생성할 사건(row) 개수. 후보값끼리 동일해야 공정 비교 가능
    phishing_rate: Optional[float] = None,  # build_dataset()에 넘길 클래스 불균형 비율(보통 config.CLASS_IMBALANCE)
    subgroup_ratio_key: str = "B",      # 정상 이벤트 하위집단(지인/기관) 비중 키. sweep_target="sophistication"일 때만 유효(고정값으로 사용)
    gen_seed: int = 42,                 # build_dataset()용 random_state. 후보값 사이에 동일해야 "파라미터만 바뀌었다"고 말할 수 있음
    fold_seed: int = 7,                 # StratifiedKFold용 random_state. 마찬가지로 후보값 사이에 고정 필요
    k: int = DEFAULT_K,                 # K-Fold 분할 개수
):
    # 한 feature에 대한 민감도 분석 진행을 위한 함수.
    # K-Fold 학습/평가 -> 결과 비교표 반환.

    # 스윕 대상 외 feature에 쓸 sophistication. None이면 mid.
    base_soph = base_sophistication or "mid"
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=fold_seed)

    summary = []
    for value in candidate_values:
        # dataset 생성을 위한 build_dataset() 호출.
        # candidate_values가 [LOW, MID, HIGH]라면 각각의 sophistication 값에 대해 총 3번의 for문 반복.
        # sophistication에 {param_name: value} + __base__를 넘기면, 스윕 대상만 value,
        # 나머지 feature는 base_soph(get_soph의 __base__)로 고정됨.
        if sweep_target == "subgroup_ratio_key":
            # RELATION_TYPE_RATIO(A~E) 스윕: sophistication이 아니라 build_dataset()의
            # subgroup_ratio_key 인자 자체가 바뀜. sophistication은 지인/기관 비중과
            # 무관한 축이라 "mid"로 고정. (base_soph와 무관)
            df = build_dataset(
                n,
                phishing_rate=phishing_rate,
                subgroup_ratio_key=value,
                random_state=gen_seed,
                config=fixed_config,
                sophistication="mid",
            )
        else:
            # 기존 방식: get_soph() 조회 key(param_name)에 해당하는 feature의
            # sophistication만 candidate_values로 바꿔가며 스윕.
            df = build_dataset(
                n,
                phishing_rate=phishing_rate,
                subgroup_ratio_key=subgroup_ratio_key,
                random_state=gen_seed,
                config=fixed_config,
                sophistication={"__base__": base_soph, param_name: value},
            )

        # SCHEMA에 확정 등록된 파생 feature 4개가 get_feature_columns()에 포함되므로, df에도
        # 실제로 그 컬럼들이 있어야 함 -> K-Fold로 나뉘기 전에 미리 계산해서 추가.
        df = add_candidate_features(df)
        df = prepare_categorical(df)  # K-Fold로 나뉘기 전에 category dtype 한 번만 확정
        X = df[get_feature_columns()] # 학습에 사용할 feature columns 관련 데이터만 추출.
        y = df["is_phishing"] # label column.

        
        fold_results = [] # 이 후보값(value) 하나에 대한 K-Fold 결과를 모으는 곳.

        for train_idx, val_idx in skf.split(X, y):
            # train_idx : train fold로 쓸 행들의 위치(0부터 세는 positional 정수) 배열.
            # val_idx   : validation fold로 쓸 행들의 위치 배열.

            train_fold = df.iloc[train_idx] # train fold에 해당하는 data
            val_fold = df.iloc[val_idx] # validation fold에 해당하는 data

            # val_fold는 학습에 안 쓰고, 오직 평가에만 사용 -> 표준 K-Fold 방식.
            # threshold는 train_fold에서 고르고 val_fold에 고정 적용.
            model = train_model(train_fold)
            fold_results.append(evaluate(model, val_fold, threshold_df=train_fold))

        summary.append({
            "param_name": param_name,
            "value": value,
            "fold_results": fold_results,  # K개 fold의 evaluate() 결과 목록. 평균/표준편차 집계는 evaluate() 반환 형태 확정 후 추가.
        })

    # 호출 경로(main.py를 거치든 직접 호출하든)와 상관없이 항상 동일하게 저장:
    # 같은 param_name이면 sensitivity_results/<param_name>.json을 덮어쓰고, 없으면 새로 만듦.
    save_sensitivity_result(param_name, summary)

    return summary


@contextmanager
def _temporary_config_overrides(config_module, overrides: Dict[str, Any]):
    """config_module(보통 config.py 모듈)의 지정된 속성(overrides의 key)을 with 블록 동안만
    임시로 덮어쓰고, 블록을 벗어나면(예외 발생 여부와 무관하게) 원래 값으로 복구.
    run_stress_sensitivity()가 build_dataset() 호출 동안에만 config 값을 바꿔치기하기 위해 사용
    (다른 스레드/후속 호출에 영향이 남지 않도록 항상 원상복구를 보장)."""
    _missing = object()  # 원래 속성이 아예 없던 경우(삭제로 복구)와 None 값을 구분하기 위한 sentinel
    originals = {attr_name: getattr(config_module, attr_name, _missing) for attr_name in overrides}
    try:
        for attr_name, value in overrides.items():
            setattr(config_module, attr_name, value)
        yield
    finally:
        for attr_name, original in originals.items():
            if original is _missing:
                delattr(config_module, attr_name)
            else:
                setattr(config_module, attr_name, original)


def run_stress_sensitivity(
    scenario_key: str,                  # 저장 파일명 stem (예: "stress_num_in_msg"). config.py의 STRESS_*_KEY 상수와 일치해야 함.
    level_overrides: Dict[str, Dict[str, Any]],  # {level_name: {CONFIG_ATTR: value, ...}}. config.py의 STRESS_*_LEVELS 상수.
    fixed_config,                       # config.py 모듈. _temporary_config_overrides()로 잠깐 덮어썼다가 복구함.
    description: str = "",              # 결과 JSON의 meta.description에 그대로 저장되는 설명 문구.
    n: int = 10000,
    phishing_rate: Optional[float] = None,
    subgroup_ratio_key: str = "B",
    gen_seed: int = 42,
    fold_seed: int = 7,
    k: int = 5,
    feature_cols: Optional[List[str]] = None,  # None이면 기존처럼 get_feature_columns()(전체) 사용.
                                                # Track 등으로 스코프를 좁히고 싶을 때만 명시.
):
    """가정-파괴(stress) / θ·확률 재정의 민감도.

    run_sensitivity()가 sophistication(low/mid/high) 중 어느 게 나은지를 비교하는 것과 달리,
    이건 "순수가정"으로 정한 절대값 자체가 틀렸을 때 성능이 얼마나 흔들리는지를 보는 함수.
    그래서 sophistication은 전부 "mid"로 고정하고, level마다 config 속성(확률 밴드, theta_list 등)만
    직접 덮어써서 스윕함. get_soph()는 soph가 dict가 아니면 그대로 통과시키므로("mid" 문자열이면
    raw="mid"), 결국 매 level에서 실제로 쓰이는 건 override한 dict의 "mid" key 값 하나뿐임.
    (low/high key는 구조 유지를 위한 참고값으로만 남겨둠).

    - scenario_key: 저장 파일명 stem (예: stress_num_in_msg)
    - level_overrides: {level_name: {CONFIG_ATTR: value, ...}}
      예) {"baseline": {"NORMAL_IS_NUM_IN_MSG": {...}}, "extreme": {...}}
    - feature_cols: 특정 Track(예: Track C만)으로 스코프를 좁혀서 스윕하고 싶을 때 명시.
      (예: url_reliability_recheck.py가 "Track C 우위가 이 가정에 얼마나 취약한가"를 보기 위해
      get_feature_columns(track_filter=[Track.DEVICE, Track.DEVICE_STRUCTURAL])를 넘김)
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=fold_seed)
    summary = []

    for level_name, overrides in level_overrides.items():
        with _temporary_config_overrides(fixed_config, overrides):
            df = build_dataset(
                n,
                phishing_rate=phishing_rate,
                subgroup_ratio_key=subgroup_ratio_key,
                random_state=gen_seed,
                config=fixed_config,
                sophistication="mid",
            )

        # override는 build_dataset 동안에만 필요. 학습은 생성된 df만 사용.
        # SCHEMA에 확정 등록된 파생 feature 4개가 get_feature_columns()에 포함되므로, df에도
        # 실제로 그 컬럼들이 있어야 함 -> K-Fold로 나뉘기 전에 미리 계산해서 추가.
        df = add_candidate_features(df)
        df = prepare_categorical(df)
        cols = feature_cols or get_feature_columns()
        X = df[cols]
        y = df["is_phishing"]

        fold_results = []
        for train_idx, val_idx in skf.split(X, y):
            train_fold = df.iloc[train_idx]
            val_fold = df.iloc[val_idx]
            model = train_model(train_fold, feature_cols=cols)
            fold_results.append(evaluate(model, val_fold, feature_cols=cols, threshold_df=train_fold))

        summary.append({
            "param_name": scenario_key,
            "value": level_name,
            "fold_results": fold_results,
        })

    save_sensitivity_result(
        scenario_key,
        summary,
        meta={
            "kind": "stress",
            "description": description,
            "levels": list(level_overrides.keys()),
            "note": "sophistication=mid 고정; config 확률/θ override만 스윕",
        },
    )
    return summary
