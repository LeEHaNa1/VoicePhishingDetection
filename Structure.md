# Structure.md — 프로젝트 폴더 구조 안내

> "이 파일이 뭐 하는 파일인지, 어디에 뭘 만들어야 하는지" 헷갈리지 않도록 정리한 문서.
> 코드 구조가 바뀔 때마다 이 문서도 같이 업데이트할 것.

---

## 전체 구조

```
(프로젝트 루트)
├── README.md
├── Structure.md
├── DataSet/
└── Simulator/
    ├── main.py
    ├── schema.py
    ├── schema_columns.py
    ├── schema_utils.py
    ├── Generation/
    │   ├── config.py
    |   ├── generator_utils.py
    │   ├── normal_generator.py
    │   ├── phishing_generator.py
    │   └── dataset_builder.py
    └── Detection/
        ├── derived_features.py
        ├── feature_ablation.py
        ├── optuna_apply.py
        ├── train_eval.py
        ├── eval_confidence.py
        ├── plot_curve.py
        ├── sensitivity_analysis.py
        ├── borderline_recheck.py
        └── url_reliability_recheck.py
```

---

## Generation과 Detection 분리 목적

**"Generation-Detection Separation" 원칙**(AMLNet 3.3절 근거) 때문.

- `Generation/` = 가짜 데이터를 만드는 부분
- `Detection/` = 그 데이터로 모델을 학습 및 평가하는 부분

핵심 규칙: `Detection/` 안의 어떤 코드도 `Generation/config.py`를 import하면 안 됨.
데이터를 만들 때 쓴 파라미터(θ값, 지인:기관 비중 등)를 학습 코드가 알고 있으면, 그 정보가 실수로 feature처럼 새어 들어가서 "생성 규칙을 암기하는" 모델이 될 위험이 있음. 두 폴더를 코드 상으로도 분리해두면 이런 실수가 구조적으로 원천 차단되므로 두 폴더를 분리 생성함.

---

## 폴더/파일별 역할

### `Simulator/main.py`

역할: 전체 파이프라인의 진입점. Generation과 Detection을 순서대로 호출하는 최상위 스크립트.

하는 일:

- generate 모드: 확정한 값으로 최종 dataset만 생성해서 `DataSet/`에 저장(학습/평가 없음). 저장된 dataset이 있고 파라미터가 그대로면 재사용(캐싱), 다르면 재생성.
- 최종 모드: 팀이 확정한 값으로 데이터셋 확보(위 캐싱 로직 재사용) → 학습 → 최종 성능 출력
- 민감도분석 모드: `sensitivity_analysis.py`를 호출해서, config.py의 각 파라미터를 하나씩 바꿔가며 여러 데이터셋 생성 → 학습 → 결과 비교표 출력
- ablation 모드 : 파생 feature 검증 및 최종 후보 확정을 위한 pipeline 실행.
- ablation_stability 모드 : SCHEMA 미등록 잔여 후보(현재 9개, `ABLATION_STABILITY_FEATURE_SETS`)를 baseline에 더했을 때 성능 차이가 노이즈인지 K-Fold로 재검증. 확정 4개는 이미 baseline에 포함되어 있어 비교 대상에 넣지 않음.
- tune 모드 : Optuna로 XGBoost/LightGBM 둘 다 하이퍼파라미터 탐색 후 val PR-AUC 더 높은 쪽(winner)을 선택
- calib_compare 모드: `run_final()`을 α(원본)/β(데이터만 축소)/γ(calibration 적용) 3버전으로 실행해서 "학습 데이터 축소 효과"와 "calibration 자체 효과"를 분리해서 비교(왜 나눠야 하는지는 아래 "confidence calibration(α/β/γ 분리)" 참고).

주의: `main.py`는 `Generation/config.py`를 직접 import해도 됨(생성 단계를 총괄 지휘하는 역할이기 때문). 문제가 되는 건 `Detection/` 내부 코드가 config를 보는 것이므로 `main.py`에서는 고려하지 않아도 됨.

주요 함수:

- `_dataset_fingerprint()`: 현재 생성 파라미터+`DATASET_GEN_VERSION`을 해시해서, 저장된 dataset을 재사용해도 되는지 판단하는 지문 생성.
- `save_final_dataset(df, out_dir)`: `DataSet/`에 csv/parquet과 `.meta.json`(지문 포함) 저장.
- `generate_final_dataset()`: `build_dataset()` 호출해서 새로 dataset 생성.
- `get_or_generate_final_dataset()`: 저장된 dataset의 지문이 현재와 같으면 재사용, 다르면 `generate_final_dataset()` 재호출(캐싱 로직 본체).
- `run_final(split_for_calibration=None, apply_calibrator=None)`: dataset 확보 → `load_tuned_hyperparams()`로 tune 결과 있으면 반영 → `train_model()`/`evaluate(pr_auc_method="average_precision")` → 최종 성능 출력. Track B/C 모델 비교이므로 `average_precision_score` 사용. 두 파라미터는 각각 `None`이면 모듈 상단 `CALIBRATION_SPLIT_TRAIN`/`CALIBRATION_APPLY` 상수(기본 둘 다 `True`)를 따름 — 자세한 내용은 아래 "confidence calibration(α/β/γ 분리)" 참고.
- `run_final_calibration_compare()`: `run_final()`을 α/β/γ 세 버전으로 실행해서 Track B/C별 스칼라 지표를 비교하고 `calibration_results/final_calibration_compare.json`에 저장.
- `run_sensitivity_mode()`/`run_sensitivity_ratio_mode()`/`run_sensitivity_each_feature_mode()`/`run_stress_mode()`: 각각 `sensitivity_analysis.py`의 `run_sensitivity()`/`run_stress_sensitivity()`를 다른 파라미터 조합으로 호출.
- `run_ablation_mode()`: `feature_ablation.py::run_feature_selection_pipeline()` 호출
- `run_ablation_stability_mode()`: `feature_ablation.py::run_stability_check()` 호출(`ABLATION_STABILITY_K`, `ABLATION_STABILITY_FEATURE_SETS` 사용).
- `load_tuned_hyperparams(out_dir)`: `optuna_results/best_params.json`이 있으면 읽어서 `(model_type, hyperparams)` 반환, 없으면 `(None, None)`.
- `run_tuning_mode()`: `optuna_apply.py::run_optuna_tuning_and_compare()` 호출 후 결과 저장.

#### confidence calibration(α/β/γ 분리) — `run_final()`/`calib_compare` 모드

**배경**: `-m final`이 XGBoost 원본 확률(`scale_pos_weight`로 불균형 보정된 값)을 그대로 쓰던 것에, `CalibratedClassifierCV`(isotonic) 기반 confidence calibration을 추가함(`calibration 추가`/`기존 train_eval에 calibration을 추가`/`calibration confidence 평가지표` 3개 커밋). 처음엔 `train_eval.py`를 통째로 복사한 `train_eval_cali.py`라는 별도 파일로 구현되어 있었는데, `main.py`/`eval_confidence.py`만 이 fork를 쓰고 나머지 전부(`sensitivity_analysis.py`, `feature_ablation.py`, `borderline_recheck.py`, `url_reliability_recheck.py`, `optuna_apply.py`, `plot_curve.py`)는 원본 `train_eval.py`를 쓰는 상태였음 — 앞으로 `train_eval.py`에 fix가 들어가도(예: `pr_auc_method` 수정) fork에는 자동 반영 안 되는 구조적 위험이 있어, calibrator 관련 코드를 전부 `train_eval.py`에 통합하고 `train_eval_cali.py`는 삭제함(아래 "`Detection/train_eval.py`" 섹션 참고).

**왜 α/β/γ로 나눠서 봐야 하는가**: calibration을 켜려면 `train_set`(70%)을 다시 `train_sub`(모델 학습용)/`val_calib`(calibrator 적합용, `CALIBRATION_VAL_SIZE` 비율)로 나눠야 함 — 즉 calibration을 켜는 순간 "calibration이 적용된다"는 변화와 "모델이 더 적은 데이터로 학습된다"는 변화가 **동시에** 일어남. 그래서 calibration 적용 전/후를 그냥 통째로 비교하면 지표 차이가 어느 쪽 때문인지 알 수 없음 → 아래처럼 버전을 3개로 쪼개서, 비교를 2단계로 나눔:

| 버전                                | `split_for_calibration` | `apply_calibrator` | 학습 데이터                | calibrator |
| ----------------------------------- | ----------------------- | ------------------ | -------------------------- | ---------- |
| α(원본)                             | False                   | False              | `train_set` 100%           | 없음       |
| β(데이터만 축소)                    | True                    | False              | `train_sub`(C와 동일 크기) | 없음       |
| γ(calibration 적용, 현재 운영 방식) | True                    | True               | `train_sub`                | 있음       |

- **비교1 = β − α**: calibration은 둘 다 없고 학습 데이터 크기만 다름 → 순수 "학습 데이터가 줄어든 효과"만 분리.
- **비교2 = γ − β**: 학습 데이터(`train_sub`)는 완전히 동일하고 calibration 유무만 다름 → 순수 "calibration 자체 효과"만 분리.
- (`split_for_calibration=False`, `apply_calibrator=True` 조합은 calibrator를 적합할 `val_calib`이 없어 `ValueError`.)

**흐름**:

```
run_final(split_for_calibration, apply_calibrator)
│
├─ get_or_generate_final_dataset() → prepare_categorical() → split_data()   (α/β/γ 공통)
│
├─ [분기①] train_sub / val_calib 결정
│    split_for_calibration=False → train_sub=train_set(100%), val_calib=None        → α버전
│    split_for_calibration=True  → train_sub≈train_set×(1-CALIBRATION_VAL_SIZE),
│                                    val_calib=나머지                                → 버전 β, γ
│
├─ load_tuned_hyperparams()   (공통)
│
└─ TRACK_SCENARIOS(B, C) 각각:
     ├─ model = train_model(train_sub, ...)     ← 분기①의 train_sub를 그대로 씀
     ├─ [분기②] calibrator 결정
     │    apply_calibrator=False → calibrator=None                                  → 버전 α, β
     │    apply_calibrator=True  → calibrator=fit_calibrator(model, val_calib, ...)  → 버전 γ
     └─ evaluate(model, test_set, threshold_df=train_sub, calibrator=calibrator)
          └─ calibrator가 있으면 calibrator.predict_proba(), 없으면 model.predict_proba()로 확률 산출
             (threshold 탐색·최종 평가 둘 다 동일하게 적용)
```

---

### `Simulator/schema.py`

역할: Dataset의 column 정의를 위한 class 선언 파일.

---

### `Simulator/schema_columns.py`

역할: 실제 Dataset에 필요한 column 목록을 생성하는 파일. 컬럼 추가/수정 시 이 파일만 건드리면 됨.

- `SCHEMA: list[ColumnSchema]`: 실제 컬럼 27개 정의(원본 23개 + 확정 파생 feature 4개). `phone_number`/`call_time` 2개만 `is_feature=False`(학습 제외 — phone_number는 통째로 암기 위험, call_time은 hour_bucket과 중복 방지), 나머지 25개(원본 21개 + 파생 4개)가 실제 학습 feature.
- 파일 하단 주석: `derived_features.py`의 13개 파생 feature 후보 중 확정 4개는 위 SCHEMA에 등록 완료. 나머지 9개(검증 탈락 또는 대기 중)는 아직 미등록.

---

### `Simulator/schema_utils.py`

역할: 다른 file에서 schema 관련 함수를 사용할 수 있도록 관련 함수를 정의한 파일. 전부 `SCHEMA`를 조회만 함(값 변경 없음).

- `get_feature_columns(track_filter=None)`: `is_feature=True`인 컬럼명만 반환. `track_filter`로 특정 Track만 필터링 가능.
- `get_schema_column_names()`: `SCHEMA` 선언 순서 그대로 전체 컬럼명 반환.
- `get_non_feature_columns()`: `is_feature=False`인 컬럼명만 반환(phone_number, call_time). `train_model()`/`evaluate()`가 "SCHEMA 등록 여부"가 아니라 "is_feature=False만 차단"하도록 만들어서, 아직 SCHEMA 미등록인 파생 feature 후보도 학습에 쓸 수 있게 하는 용도.
- `get_nullable_columns()`/`get_non_nullable_columns()`: `nullable=True`/`False`인 컬럼명 반환.
- `validate_schema()`: SCHEMA 전체 자체 모순 검사(컬럼명 중복, 각 컬럼의 `validate()` 결과, `Track.ID` 컬럼이 정확히 1개인지, ID 컬럼이 `is_feature=True`로 잘못 설정되지 않았는지).

---

### `Generation/config.py`

역할: 마스터표에서 확정한 정상값, 피싱값 전체를 담는 설정 파일. 코드가 아닌 값들의 저장소 역할.

- 유형/하위집단 상수: `LOAN`/`INSTITUTION`/`ACQUAINTANCE`/`ETC`(피싱 4유형), `SUBGROUP_ACQUAINTANCE`/`SUBGROUP_INSTITUTION_PERSONAL`/`SUBGROUP_INSTITUTION_CORPORATE` 등(정상 하위집단).
- `SOPH_*` 상수 11개(`SOPH_NUMBER_TYPE_BAND`, `SOPH_FIRST_CONTACT`, `SOPH_REPEAT_GAP` 등): feature별로 sophistication(low/mid/high)을 개별 조회할 때 쓰는 key 이름들. `generator_utils.get_soph()`가 이 key로 딕셔너리를 찾음.
- `normalize(number)`/`classify_number_type(number)`: 전화번호 문자열을 정규화하고, 010/070/특번/02/00X(국제) 등 카테고리로 분류하는 함수.
- `WHITELIST_NUMS`/`WHITELIST_SET`/`WHITELIST_LIST`: 기관 정상 이벤트에서 우선 사용하는 실존 대표번호 화이트리스트.
- `CLASS_IMBALANCE = 0.01`: 전체 dataset의 피싱 비율(클래스 불균형).
- `TYPE_RATIO`: 피싱 4유형 간 비중. `RELATION_TYPE_RATIO`(A~E): 정상 이벤트 지인:기관 하위집단 비중 키(민감도분석 대상, 현재 D로 확정).
- `NORMAL_*`/`PHISHING_*` 쌍으로 된 다수의 확률·분포 dict(`*_IS_URL_IN_MSG`, `*_REPEAT_GAP`, `*_SMS_TO_CALL` 등): 각 feature마다 정상값/피싱값을 sophistication(low/mid/high)별로 담음. `normal_generator.py`/`phishing_generator.py`가 이 값들을 읽어서 이벤트를 합성.
- `STRESS_*_KEY`/`STRESS_*_LEVELS`(예: `STRESS_URL_RATE_KEY`/`STRESS_URL_RATE_LEVELS`): 스트레스테스트(`main.py -m stress`) 모드에서 특정 feature 값을 극단으로 밀어붙일 때 쓰는 레벨 정의.

---

### `Generation/normal_generator.py`

역할: 정상 이벤트 1건을 생성하는 함수를 담는 파일.

- `_group_key(subgroup, config)`: `기관_개인`/`기관_기업`을 `SUBGROUP_INSTITUTION`으로 묶는 내부 helper.
- `_generate_normal_phone(subgroup, config, soph)`: 지인이면 010 고정, 기관이면 화이트리스트 우선(85%) 또는 `NORMAL_NUMBER_TYPE` 비중으로 합성.
- `_calculate_repeat_gap(group_key, config, sophistication)`: `NORMAL_REPEAT_GAP` 혼합분포(지수분포, theta 랜덤 선택)로 재연락 간격 계산.
- `generate_normal_event(subgroup, config, sophistication="mid")`: 메인 함수. 위 helper들과 `config`의 각 `NORMAL_*` 값을 조합해 이벤트 dict 1건(23개 원본 컬럼 중 `is_phishing`/`incident_type` 제외 21개) 생성. 결측 규칙(규칙 1: 선행 조건 미충족 시 NaN)을 `if/else`로 직접 구현.

---

### `Generation/phishing_generator.py`

역할: 피싱 이벤트 1건을 생성하는 함수를 담는 파일.

- `_pick_from_call_band(config, soph)`: `PHISHING_CALL_NUMBER_TYPE` 비중으로 발신번호 카테고리를 뽑는 내부 helper.
- `_generate_phishing_phone(p_type, first_contact_type, sophistication, config)`: 문자 개시면 `PHISHING_SMS_NUMBER_TYPE`, 아니면(또는 유형에 문자 전용 분포가 없으면) `_pick_from_call_band()` 차용.
- `generate_phishing_event(p_type, sophistication, config)`: 메인 함수. `normal_generator.generate_normal_event()`와 동일한 구조로 피싱 유형별 `PHISHING_*` 값을 조합해 이벤트 dict 1건 생성. `hour_bucket`은 ETC 유형만 NaN(표본 부족), Track A(`is_carrier_altered`/`number_cluster`/`using_duration`/`unique_callees`)는 항상 NaN.

---

### `Generation/dataset_builder.py`

역할: `normal_generator`와 `phishing_generator`를 실제로 n번 호출해서, 완성된 데이터셋(표) 하나를 만드는 파일. Algorithm1 전체를 구현하는 곳.

- `build_dataset(n, phishing_rate, config, sophistication="mid", subgroup_ratio_key="B", random_state=None)`: Algorithm1을 1:1로 구현. 사건마다 `random.random() < phishing_rate`로 피싱/정상 독립 결정 → 피싱이면 `generate_phishing_event()`, 정상이면 `generate_normal_event()` 호출 → `is_phishing`/`incident_type` 컬럼 추가 → 전체를 DataFrame으로 만들고 컬럼 순서를 `get_schema_column_names()` 기준으로 고정 → `sample(frac=1.0)`으로 셔플 후 반환.
- `validate_check_dataset(df, preview_sample_size=10)`: 생성된 dataset의 결측 규칙 4가지(선행 조건 미충족 시 NaN, `get_non_nullable_columns()` 전체 결측 0건, Track A 컬럼 전부 NaN, `is_sequential_callers` 확정적 0/1)를 실제로 재검증하고 PASS/FAIL 리포트 출력. `schema.py`의 `depends_on`은 사람이 읽는 텍스트라 자동 검증이 안 되므로 여기서 조건을 직접 재현해서 검사.

---

### `Generation/generator_utils.py`

역할: `normal_generator.py`와 `phishing_generator.py`가 공유하는 함수 선언 파일.

- `AREA_CODES`: 지방 유선 국번 목록(번호 합성용). `SOPH_ALIAS`: sophistication 구 한글키(`짧게`/`중간`/`길게` 등) 호환용 별칭 dict.
- `get_soph(soph, key)`: sophistication 값이 문자열이면 그대로, dict이면 `key` 우선 → 없으면 `__base__` → 그래도 없으면 `mid`로 확정. 두 generator의 모든 확률 조회가 이 함수를 거침.
- `digits(n)`: n자리 임의 숫자 문자열 생성.
- `phone_from_category(category)`: `config.classify_number_type()`의 카테고리(010/특번/02/070/00X/기타)에 맞는 형식으로 전화번호 문자열 합성.

---

### `Detection/train_eval.py`

역할: 데이터셋 하나를 받아서, 분할→학습→평가까지 한 번에 처리하는 파일.

- `prepare_categorical(df)`: 범주형 컬럼을 pandas `category` dtype으로 변환(XGBoost `enable_categorical=True`/LightGBM 네이티브 범주형 처리에 필요).
- `_resolve_feature_cols(feature_cols)`: `feature_cols`가 없으면 `get_non_feature_columns()` 기준으로 전체 컬럼에서 비-feature만 제외하고 반환(SCHEMA 미등록 파생 feature도 학습에 쓸 수 있게 하기 위함).
- `split_data(df)`: train/val/test 분할.
- `pr_auc(y, proba)`: PR-AUC를 `precision_recall_curve()`+`auc()`(사다리꼴 적분)로 계산. `evaluate()`의 기본값(`pr_auc_method="trapezoidal"`)이 이 함수를 씀.
- `lift_at_top_k(y, proba, k=0.05)`: 상위 k% Lift 계산.
- `train_model(df, val=None, feature_cols=None, hyperparams=None, model_type="XGBoost")`: `model_type`에 따라 XGBoost/LightGBM 분기 학습. `hyperparams` 전달 시 Optuna tune 결과 반영.
- `evaluate(model, df, feature_cols=None, threshold_df=None, pr_auc_method="trapezoidal", calibrator=None)`: 학습된 모델을 평가. `threshold_df`를 별도로 받아 "임계값 결정에 test_set을 쓰지 않는" 낙관 편향 방지 구조. `pr_auc_method`(2026-09-19 추가)는 `"trapezoidal"`(기본, `pr_auc()`)과 `"average_precision"`(`average_precision_score()`, 직선 보간 편향이 없어 모델/조합 비교에 더 적합) 중 선택 — `sensitivity_analysis.py`는 기본값(사다리꼴, 상대적 우열 비교 목적)을 그대로 쓰고, `feature_ablation.py`와 `main.py::run_final()`은 `"average_precision"`을 명시적으로 넘김(아래 각 섹션 참고). `calibrator`(2026-09-23 추가)를 넘기면 threshold 탐색·최종 평가 모두 `model.predict_proba()` 대신 `calibrator.predict_proba()`로 확률을 뽑음 — `None`(기본값)이면 기존과 완전히 동일(하위 호환).
- `fit_calibrator(model, val_df, feature_cols, target_col="is_phishing", method="isotonic")`(2026-09-23 추가): 이미 학습된 `model`을 `scale_pos_weight`로 왜곡된 확률 대신 실제 발생 빈도에 가까운 확률로 재정렬하는 `CalibratedClassifierCV`를 적합해서 반환. `val_df`는 `model` 학습에 안 쓰인 데이터여야 함(같은 데이터로 보정하면 과적합까지 "잘 보정된 것"처럼 왜곡됨). scikit-learn 1.6+에서는 `FrozenEstimator`로 이미 학습된 `model`을 그대로 감싸서 보정(재학습 없음), 1.6 미만이라 `FrozenEstimator`가 없으면 `cv="prefit"`으로 fallback — **주의**: 이 fallback에 `cv=None`을 쓰면(과거 실제로 있었던 버그) "prefit 모델을 그대로 써라"가 아니라 "기본 K-fold로 model을 처음부터 다시 학습해라"는 뜻이 되어버려 조용히 다른(잘못된) 동작을 하므로 반드시 `"prefit"`이어야 함.

**연혁**: 이 calibrator 관련 코드는 원래 `train_eval.py`를 통째로 복사한 별도 파일(`train_eval_cali.py`)에 있었음. `main.py`/`eval_confidence.py`만 그 fork를 쓰고 나머지 전부는 원본 `train_eval.py`를 쓰는 상태라 두 파일이 서서히 어긋날 위험이 있었음 → 같은 날 `train_eval.py`에 통합하고 `train_eval_cali.py`는 삭제(`fit_calibrator`도 이 파일로 이동, `cv=None` fallback 버그도 이때 같이 고침).

**이 파일은 `Generation/config.py`를 import하지 않아야 함.**

---

### `Detection/eval_confidence.py`

역할: confidence calibration(isotonic)의 효과를 Track B/C 각각에 대해 정량 평가하는 스크립트. ECE만으로는 부족하다는 판단 하에(피싱 1% 데이터에서는 "항상 base rate만 예측"해도 ECE≈0이므로) Brier/LogLoss(확률의 질)와 PR-AUC(변별력)를 함께 산출함. `main.py`의 파이프라인 모드가 아니라 별도 실행 스크립트

CLI 모드가 아니라 `python -X utf8 -m Simulator.Detection.eval_confidence`로 실행.

- `_bin_stats(y_true, proba, n_bins, strategy)`: `strategy`가 `"quantile"`이면 동점 경계를 병합한 분위수 bin, `"uniform"`이면 등간격 bin으로 나눠 bin별 (평균 예측확률, 실제 양성 비율, 표본 수)를 반환. isotonic이 만드는 계단형 동점 값 때문에 quantile 경계가 중복되면 유효 bin 수가 줄어듦(Track C 보정 후는 1개까지 줄 수 있음).
- `_ece(mean_pred, frac_pos, counts)` / `_ece_quantile(y_true, proba, n_bins)`: bin 표본 수로 가중한 ECE 계산.
- `compute_calibration_metrics(y_true, proba, n_bins=10)`: 한 세트의 확률에 대해 `ece_quantile`/`ece_uniform`/`brier`/`log_loss`/`mean_predicted`/`base_rate`/`mean_gap_ratio`(평균 예측확률과 실제 비율의 격차, ±10% 이내면 "평균 정렬"로 판정)/`pr_auc`/`n_unique_proba`(서로 다른 확률값 개수, isotonic이라 작음)/`frac_near_zero_proba`·`positives_in_near_zero`(확률≈0인 표본 비율과 그 안의 실제 피싱 수, isotonic 0-plateau가 LogLoss를 얼마나 해치는지 진단)를 반환. 내부적으로 quantile bin 통계(`q_bins`)도 함께 반환하지만, 이 bin으로 그리는 3-panel reliability diagram은 삭제됨(아래 참고).
- `bins_to_rows(bins)`: bin 통계를 JSON 저장용 행 목록으로 변환. `confidence_metrics.json`의 `quantile_bins` 필드에 저장.
- `tail_bin_table(y_true, proba, edges=(0, 0.005, 0.01, 0.02, 0.05, 0.10, 1.0))`: 10-분위 bin은 피싱이 상위 10% bin 하나에 몰려 점이 2~3개(Track C 보정 후는 1개)뿐이라 정보가 거의 없으므로, 점수 내림차순 상위 0-0.5%/0.5-1%/1-2%/2-5%/5-10%/나머지 구간별 (표본 수, 양성 수, 평균 예측확률, 실제 양성 비율)을 별도 산출.
- `plot_tail_reliability(tail_by_track, out_dir)`: 위 상위 구간별 통계를 Track B/C 나란히 한 장(`calibration_results/tail_reliability_diagram.png`)에 그림 — 보고서 본문에 실제로 쓰인 그림.
- `_print_report(track_name, raw_m, cal_m)` / `_print_tail_table(raw_rows, cal_rows)`: 콘솔에 지표 표·상위 구간 표 출력.
- `run_confidence_analysis()`: 메인 함수. Track B/C 각각에 대해 `main.py::run_final()`의 버전 C(γ)와 동일한 분할(`CALIBRATION_VAL_SIZE`=0.3, `train_sub`/`val_calib`/`test_set`, seed 42)로 모델 학습 + `fit_calibrator()`로 isotonic 보정기 적합 → 위 함수들을 호출해 콘솔 출력 + `calibration_results/confidence_metrics.json`(Track별 `raw`/`calibrated`/`quantile_bins`/`tail_bins`) 저장.

**이 파일은 `Generation/config.py`를 import하지 않아야 함.**

---

### `Detection/plot_curve.py`

역할: Track B/C × Confidence Calibration 유무(Raw/Calibrated)로 나눈 Learning Curve와 Cumulative Lift Curve를 그리는 스크립트.

CLI 모드가 아니라 `python -X utf8 -m Simulator.Detection.plot_curve`로 실행.

- `_prepare_splits()`로 `main.py::run_final()`과 동일한 분할(`train_sub`/`val_calib`/`test_set`, seed 42)을 재사용 — 이 스크립트의 Raw PR-AUC(learning curve 마지막 지점)는 `final_calibration_compare.json`의 버전 B PR-AUC와 일치해야 함(재현성 확인용으로도 쓸 수 있음).
- `plot_learning_curve(...)`: `train_sub`를 10단계 크기로 잘라가며 매 크기마다 모델을 새로 학습해서 train subset과 `test_set`의 PR-AUC를 기록. `plot_results/learning_curve_track{B,C}_{raw,calibrated}.png`.
- `plot_cumulative_lift(...)`: `test_set` 예측 확률 상위 x%(1~100%) 누적 lift. `plot_results/cumulative_lift_track{B,C}_{raw,calibrated}.png`.
- `run_all_plots()`: Track당 model+calibrator를 한 번만 학습해 Lift Curve가 Raw/Calibrated 두 버전 모두 공유하고(Learning Curve는 크기별로 자체 재학습하므로 별도), Track B/C × Raw/Calibrated × (Learning+Lift) 총 8개 그래프를 `plot_results/`에 저장

** 보고서에 사용한 시각화는 Track C를 보정(calibration)한 curve만 사용.**
**이 파일은 `Generation/config.py`를 import하지 않아야 함.**

---

### `Detection/sensitivity_analysis.py`

역할: config.py의 스윕 파라미터(짧게/중간/길게, 지인기관비중 A~E 등)를 하나씩 바꿔가며 `dataset_builder` + `train_eval`을 반복 실행하고, 결과를 비교표로 정리하는 파일.

- `summarize_fold_results(fold_results)`: K번 반복 실행 결과(fold별 metric)를 평균/표준편차 등으로 요약.
- `save_sensitivity_result(...)`: 결과를 `sensitivity_results/`에 json으로 저장.
- `run_sensitivity(...)`: 파라미터 하나를 여러 값으로 바꿔가며 `build_dataset()`+`train_eval` 반복 실행(민감도분석 본체). `sen`/`each_sen`/`ratio` 모드가 공통으로 사용.
- `_temporary_config_overrides(config_module, overrides)`: `config.py`의 특정 값을 일시적으로 덮어썼다가 복원하는 context manager(스윕 중 원본 config 오염 방지).
- `run_stress_sensitivity(..., feature_cols=None)`: `STRESS_*_LEVELS`로 특정 feature 값을 극단으로 밀어붙이는 스트레스테스트(`stress` 모드) 실행. `feature_cols`(2026-09-23 추가)를 넘기면 그 컬럼만으로 학습/평가 — 기본값(None)이면 기존처럼 `get_feature_columns()`(전체)를 씀. `url_reliability_recheck.py`가 Track C(21개)만으로 재검증할 때 이 인자를 씀.

---

### `Detection/derived_features.py`

역할: 파생 feature 후보 column을 기존 dataset에 추가하는 함수 선언.

- `add_candidate_features(df)`: 원본 컬럼만으로 13개 파생 feature 후보를 전부 계산해서 `df`에 추가(확정 여부와 무관하게 항상 13개 전부 계산).
- 13개 중 4개는 ablation 검증 통과 후 `schema_columns.py`의 `SCHEMA`에 확정 등록됨(2026-09-19). `main.py::generate_final_dataset()`이 이 함수 호출 후 SCHEMA 등록 컬럼만 필터링해서 최종 dataset에는 4개만 반영. 나머지 9개는 여전히 미등록(검증 탈락 또는 대기 중, 최종 확정 목록은 아래 `feature_ablation.py` 섹션 참고).

---

### `Detection/feature_ablation.py`

역할: 파생 feature 검증 pipeline (선정 1회 + 사후 K-Fold 재검증)

이 파일은 목적이 다른 두 함수로 구성됨:

- `run_feature_selection_pipeline()`: 13개 후보 중 "무엇을 최종으로 고를지" 결정하는 선정 과정. 반복하면 결정 자체가 흔들릴 수 있어 dataset 1개, train/test 고정 분할 1회로 진행(K-Fold 미사용).
- `run_stability_check()`: 위에서 "이미 결정된" 최종 feature 조합의 성능 차이가 실제 효과인지 단일 실행의 우연(노이즈)인지 확인하는 사후 검증. "무엇을 고를지"를 다시 정하는 게 아니므로 `run_sensitivity()`와 동일한 StratifiedKFold(K-Fold) 방식을 써도 선정 과정의 1회 고정 분할 원칙과 모순되지 않음.

선언된 함수/상수:

- `CANDIDATE_DERIVED_FEATURES`: `derived_features.py`가 만드는 13개 후보 중 아직 SCHEMA 미등록인 9개(확정 4개 제외) 이름 목록. `run_stability_check()` 기본 검증 대상이기도 함.
- `FINAL_CANDIDATE_FEATURES`: `run_feature_selection_pipeline()` 실행(2026-09-19, N=100,000) 결과로 확정된 최종 4개(아래 표 참고). 참고용 상수 — SCHEMA에 이미 등록되어 `baseline_cols`에 포함되므로 `run_stability_check()`의 기본 검증 대상이 아님(기본은 잔여 9개).
- `DEFAULT_MULTICOLLINEARITY_THRESHOLD=0.8`/`DEFAULT_LEAKAGE_THRESHOLD=0.95`/`DEFAULT_VAL_SIZE=0.2`/`DEFAULT_PERM_N_REPEATS=10`: 선정 파이프라인 각 단계 기본 임계값·비율.
- `DEFAULT_STABILITY_K=10`: `run_stability_check()` 기본 K(민감도분석 기본값 5보다 크게 잡음 — 재검증 성격이라).
- `_metrics_summary(m)`: `evaluate()` 결과에서 threshold/accuracy/precision/recall/f1/PR-AUC/Lift@Top5~20%만 추려 요약(두 함수가 공유). 두 함수 모두 `evaluate()` 호출 시 `pr_auc_method="average_precision"`을 명시함(baseline/full/final, baseline/조합끼리 서로 비교하는 목적이라 — `train_eval.py` 섹션 참고).
- `_select_by_correlation(train_df, candidate_features, target_col, threshold)`: 2단계 다중공선성 제거(후보끼리 상관계수 threshold 초과 쌍 중 하나 drop).
- `_check_leakage(train_df, candidate_features, target_col, threshold)`: 2단계 데이터 누수 플래그(후보-target 상관계수만 검사, 자동 제거는 안 함).
- `_patch_shap_xgboost_base_score_bug()`: shap/xgboost 3.x `base_score` 문자열 파싱 버그 몽키패치.
- `_compute_shap_importance(model, X)`: 3단계 SHAP 중요도 계산(참고용).
- `_compute_permutation_importance(model, X_val, y_val, n_repeats, random_state)`: 3단계 Permutation Importance 계산(실제 가지치기 기준, `train_sub`/`val_sub` 분리해서 held-out 데이터로 검증). `scoring="average_precision"`을 처음부터 써서, 오늘 `evaluate()`의 `pr_auc_method` 추가와 무관하게 선정 기준 자체는 계속 동일했음.
- `run_feature_selection_pipeline(...)`: 위 helper들을 묶어 0~4단계 전체를 실행하는 선정 메인 함수.
- `_diff_summary(baseline_summary, combo_summary)`: `run_stability_check()`의 baseline/조합별 summary에서 지표별 평균(mean) 차이만 뽑아 요약(양수=해당 조합이 평균적으로 더 좋음).
- `run_stability_check(fixed_config, feature_sets=None, k=10, ...)`: dataset 1개만 생성 → `StratifiedKFold`로 K개 fold 분할 → baseline(25개, 확정 4개 포함) 모델과, `feature_sets`(`{조합명: [파생 feature 목록]}`, 기본값은 `{"remaining_candidates": CANDIDATE_DERIVED_FEATURES}`)에 담긴 조합별 모델들을 **같은 fold** 안에서 함께 반복 학습·평가 → `sensitivity_analysis.summarize_fold_results()`로 조합별 mean/std/sem 집계 → `_diff_summary()`로 baseline 대비 차이 요약. `feature_sets`에 넣는 feature가 `baseline_cols`와 겹치면 `ValueError`(컬럼 중복 방지, 2026-09-19 추가).

선정 파이프라인(`run_feature_selection_pipeline()`) 흐름:

- 0단계 : 기본 작업
- 1단계 : baseline vs full 비교
- 2단계 : 다중공선성 검증, 데이터 누수 플래그
- 3단계 : Embedded method + SHAP + Permutation Importance
- 4단계 : 최종 재학습

최종 산출물 : 검증 후 통과된 최종 파생 feature 목록으로 학습한 결과를 포함한 json 파일. (feature_selection_results/ 에 저장됨)

#### 파생 feature 최종 확정 목록 (2026-09-19, N=100,000 실행 기준)

`derived_features.py`가 만드는 13개 후보 중, 다중공선성(3개 제거) → permutation importance(6개 추가 제거)를 거쳐 **4개만 최종 생존**했음.

| column명                    | 정의                                         | 값 형태           | Track              | 결측 여부                                |
| --------------------------- | -------------------------------------------- | ----------------- | ------------------ | ---------------------------------------- |
| `structural_phishing_score` | 문자 구조적 위협 누적 지수(0~5점)            | 이산형(정수, 0~5) | 단말+구조적신호(C) | 결측 없음(하위 조건 미발생분은 0점 처리) |
| `is_sms_initiated_unreg`    | 미등록 발신자의 문자 개시 여부               | 이진(0/1)         | 단말(B)            | 결측 없음                                |
| `cold_contact`              | 완전 낯선 접촉 여부(미저장+이력 없음)        | 이진(0/1)         | 단말(B)            | 결측 없음                                |
| `repeat_pressure_intensity` | 재연락 압박 강도(재연락 간격 기반 연속 점수) | 연속형(점수)      | 단말(B)            | 결측 없음(재연락 없으면 0.0)             |

**성능 비교**(test_set 기준): baseline(원본 21개) f1=0.978, PR-AUC=0.992 vs final(원본+위 4개, 25개) f1=0.978(동일), PR-AUC=0.989(소폭 하락). 뚜렷한 성능 향상은 아니지만 손해도 거의 없는 수준.

#### `-m ablation` 재실행 시 결과가 달라지는 현상 (원인 확인 및 최종 결정, 2026-09-19)

동일한 `config.py`/코드로 `-m ablation`을 다시 실행하면 `final_candidate_features`가 4개가 아니라 5~6개로 나올 때가 있음(`unreg_sender_with_url`, `suspicious_unreg_number_combo`가 추가로 포함됨). config.py 변경이나 코드 변경 때문이 아니라 **`train_eval.py::train_model()`이 쓰는 `tree_method="hist"`(히스토그램 기반, 멀티스레드) 특성 때문**임을 직접 재현해서 확인함:

- 같은 Python 프로세스 안에서 동일 데이터로 두 번 학습 → 예측값 100% 완전 동일(결정론적).
- 완전히 새 프로세스로 두 번 연속 실행(동일 config, 동일 코드) → PR-AUC 등 지표가 소수점 4~5자리 수준에서 미세하게 달라짐. `random_state=42` 고정으로도 이 정도의 부동소수점 비결정성까지는 못 막음(멀티스레드 히스토그램 계산의 덧셈 순서가 스레드 스케줄링에 따라 실행마다 조금씩 달라질 수 있음 — 특정 터미널/셸 문제가 아니라 "프로세스를 새로 띄울 때마다" 나타나는 XGBoost 자체의 알려진 한계).
- 이 미세한 흔들림 자체는 무시할 수준이지만, `unreg_sender_with_url`/`suspicious_unreg_number_combo`처럼 permutation importance가 0에 거의 붙어있는 feature는 이 흔들림만으로도 0을 넘었다 안 넘었다 하며 매 실행마다 포함/제외가 뒤집힘.
- 반대로 위 표의 4개(`structural_phishing_score`/`is_sms_initiated_unreg`/`cold_contact`/`repeat_pressure_intensity`)는 여러 차례 재실행에서 한 번도 빠짐없이 생존 → 실행 노이즈에 흔들리지 않는 안정적인 신호로 판단.

**최종 결정**: `unreg_sender_with_url`/`suspicious_unreg_number_combo`는 노이즈 수준으로 판단해 제외하고, 위 표의 **4개를 최종 확정 목록으로 유지**함. `run_stability_check()`(`-m ablation_stability`)까지 돌리지 않고도, 반복 재실행 자체로 이미 "무엇이 안정적인 신호인지"가 충분히 드러났다고 판단.

#### `run_stability_check()` K=10 재검증 결과 (2026-09-19, N=100,000, `-m ablation_stability`)

baseline(21개) vs final(21+4개, 25개)을 같은 fold 안에서 10회 반복 비교. threshold/accuracy 제외, diff는 "final − baseline"(mean 기준):

| 지표        | baseline mean(sem) | final mean(sem) | diff    | 판정        |
| ----------- | ------------------ | --------------- | ------- | ----------- |
| precision   | 0.9809(0.0041)     | 0.9839(0.0043)  | +0.0030 | 노이즈 범위 |
| recall      | 0.9718(0.0053)     | 0.9728(0.0061)  | +0.0010 | 노이즈 범위 |
| f1          | 0.9762(0.0032)     | 0.9782(0.0038)  | +0.0020 | 노이즈 범위 |
| PR-AUC      | 0.9906(0.0028)     | 0.9897(0.0029)  | −0.0009 | 노이즈 범위 |
| Lift@Top5%  | 19.899(0.0592)     | 19.859(0.0577)  | −0.040  | 노이즈 범위 |
| Lift@Top10% | 9.970(0.0202)      | 9.960(0.0210)   | −0.010  | 노이즈 범위 |
| Lift@Top20% | 4.990(0.0095)      | 4.990(0.0095)   | 0.000   | 차이 없음   |

전 지표에서 diff가 baseline/final 결합 표준오차(√(sem²+sem²))보다 작음 → **통계적으로 baseline과 구별 안 됨**. 지난 단일 실행에서 관찰됐던 PR-AUC/Lift@Top5% 하락은 실제 효과가 아니라 XGBoost 프로세스 간 비결정성 노이즈의 한 표본이었음이 이 재검증으로 확인됨. 4개 유지 결정을 뒷받침하는 근거.

**참고**: `run_stability_check()`는 이후 `feature_sets` 인자로 여러 조합(`ABLATION_STABILITY_FEATURE_SETS`의 `all4`/`strong_pair`=`cold_contact`+`structural_phishing_score`/`weak_pair`=`is_sms_initiated_unreg`+`repeat_pressure_intensity`)을 한 번에 비교할 수 있도록 확장됨. 이 세분화 버전도 K=10으로 실행 완료(2026-09-19) — `strong_pair`만으로 `all4`의 긍정적 움직임(f1/precision 상승)을 거의 그대로 재현했고, `weak_pair`는 f1/PR-AUC diff가 정확히 0으로 사실상 기여가 없었음. 다만 세 조합 모두 baseline과 통계적으로 구별되지 않는 노이즈 범위였으므로, **4개를 그대로 유지하기로 최종 결정**(줄일 근거는 있지만 줄여야 할 이유도 없음).

**실제 파이프라인 반영 완료**:

- ① `schema_columns.py`의 `SCHEMA`에 위 4개 `ColumnSchema` 등록(`ValueType.CONTINUOUS_SCORE`를 `schema.py`에 신규 추가해서 `repeat_pressure_intensity`에 사용). SCHEMA 총 23→27개, `get_feature_columns()` 21→25개.
- ② `main.py::generate_final_dataset()`에서 `build_dataset()` 직후 `add_candidate_features()` 호출 → `get_schema_column_names()`(+`is_phishing`/`incident_type`) 기준으로 필터링해서 미확정 9개 후보는 자동 제외하고 저장.
- ③ `main.py::DATASET_GEN_VERSION`을 1→2로 올려 기존 캐시 무효화.
- ④ `-m generate`/`-m final` 실제 실행 완료 — `DataSet/final_dataset.*` shape=(100000, 29)(원본 21+파생 4+비feature 2+라벨 2), 4개 파생 feature 전부 결측 0건으로 확인.
- **부수 수정**: `get_feature_columns()`가 이제 파생 4개를 포함하므로, `build_dataset()` 결과만으로 바로 `get_feature_columns()`를 쓰던 `sensitivity_analysis.py::run_sensitivity()`/`run_stress_sensitivity()`에도 `add_candidate_features()` 호출을 추가(안 하면 파생 컬럼이 없어 KeyError). `feature_ablation.py::CANDIDATE_DERIVED_FEATURES`에서도 확정된 4개를 제거(9개만 남음) — 안 그러면 이제 `get_feature_columns()`(=`baseline_cols`)에 이미 포함된 4개가 `full_cols`에서 중복 선택됨.
- (선택) `-m ablation_stability`는 baseline 대비 성능 차이 자체가 노이즈인지 K-Fold로 재검증하는 별도 도구로 계속 사용 가능(위에서 이미 실행 완료).
- **뒤늦게 발견/수정한 회귀 버그**: 위 SCHEMA 등록 이후 `baseline_cols`(=`get_feature_columns()`)가 이미 파생 4개를 포함하게 되면서, `run_stability_check()`의 기존 기본값(`FINAL_CANDIDATE_FEATURES`, 구 4개)과 `main.py::ABLATION_STABILITY_FEATURE_SETS`의 `all4`/`strong_pair`/`weak_pair` 조합이 전부 "이미 baseline에 있는 feature를 또 더하는" 상태가 되어 컬럼 중복으로 크래시하는 문제가 있었음(알아보기 힘든 XGBoost 내부 `AttributeError`로 발생). 다음과 같이 수정함:
  - `run_stability_check()`에 `feature_sets`의 각 조합이 `baseline_cols`와 겹치면 명확한 `ValueError`를 내는 방어 로직 추가.
  - `run_stability_check()` 기본값을 `{"remaining_candidates": CANDIDATE_DERIVED_FEATURES}`(아직 SCHEMA 미등록인 9개 묶음)로 변경.
  - `main.py::ABLATION_STABILITY_FEATURE_SETS`도 동일하게 `{"remaining_candidates": CANDIDATE_DERIVED_FEATURES}`로 교체(9개 중 일부가 단일 실행의 우연 때문에 탈락한 건 아닌지 재검증하는 용도로 의미 변경).

#### PR-AUC 계산 방식 분리 및 재검증

- `evaluate(pr_auc_method="average_precision")`을 `feature_ablation.py`(baseline/full/final, baseline/조합 비교 전부)와 `main.py::run_final()`(Track B/C 비교)에 명시 — 직선 보간 편향이 없는 `average_precision_score`가 모델 간 비교에 더 적합하다는 판단.
- `optuna_apply.py`는 애초에 `train_eval.py`를 쓰지 않고 독립적으로 `average_precision_score`를 써서 변경 불필요.

**중요**: feature 선정을 실제로 결정하는 `_compute_permutation_importance()`는 처음부터 `scoring="average_precision"`을 내부적으로 써 왔기 때문에, 이번 `pr_auc_method` 정리는 "어떤 파생 feature가 최종 선정되는가"라는 핵심 결정에는 영향을 주지 않음 — 영향을 받는 건 `metrics_comparison`/`step1`/`step4`/K=10 재검증 표에 **보고용으로 찍히는 PR-AUC 숫자**뿐.
이 정리 직후 `-m ablation`을 다시 실행해서 직접 확인함: 이미 탈락했던 후보들이 더 강해진 baseline 대비로도 여전히 기여가 없음을 재확인한 것이라, **4개 확정 결정은 그대로 유지**.

---

### `Detection/optuna_apply.py`

역할: XGBoost/LightGBM 하이퍼파라미터를 Optuna로 탐색하고, 두 모델을 비교하는 파일.

하는 일:

- `create_objective()`: 모델 종류(XGBoost/LightGBM)별 Optuna 목적함수 생성. train_sub로 학습, val_sub의 PR-AUC(average_precision)를 최적화 대상으로 삼음.
- `run_optuna_tuning()`: 모델 종류 1개에 대해 `study.optimize()` 실행 → 최적 하이퍼파라미터 + 확정 n_estimators 반환.
- `run_optuna_tuning_and_compare()`: XGBoost/LightGBM 둘 다 탐색해서 val PR-AUC가 더 높은 쪽을 winner로 선택.

최종 산출물: winner 모델 종류 + 하이퍼파라미터가 담긴 json 파일(`optuna_results/best_params.json`). `main.py::run_final()`이 다음 실행부터 이 파일을 자동으로 읽어서 반영함.

**이 파일은 `Generation/config.py`를 import하지 않아야 함.**

---

### `Detection/borderline_recheck.py`

역할: `-m final` 단일 실행에서 경계선으로 남은 해석을, 같은 dataset/split/하이퍼파라미터로 재확인하는 스크립트. CLI 모드가 아니라 `python -X utf8 -m Simulator.Detection.borderline_recheck`로 실행.

하는 일:

- 약한 파생 2개(`is_sms_initiated_unreg`, `repeat_pressure_intensity`)를 빼고 Track B/C를 재학습해 F1/PR-AUC diff를 비교.
- `is_global` 빈도 및 `number_type == "00X(국제)"`와의 일치율 확인.
- Track B/C 최종 모델에 대해 test_set permutation importance(`scoring="average_precision"`, 10회)를 계산해 gain과 대조.

최종 산출물: `borderline_recheck_results/borderline_recheck.json`.

**이 파일은 `Generation/config.py`를 import하지 않아야 함.** (`main.py`의 dataset 캐시·하이퍼파라미터 로더만 사용)

---

### `Detection/url_reliability_recheck.py`

역할: `-m final`에서 확인된 "Track C가 Track B보다 우위"라는 결론이 `is_reliable_url`(URL 신뢰도) 가정에 얼마나 의존하는지 확인하는 스크립트. `is_reliable_url` stress 테스트를 **Track C(21개 feature)로 스코프를 좁혀서** 재실행하고, 그 과정에서 `is_reliable_url`이 무너질 때 다른 feature가 그 빈자리를 부분적으로 메우는지("부분 복구")를 **2단계로 나눠** 검사함. 원래 실행분(전체 25개 feature 기준, 아래 "① 성능" 표)과는 별개로 **"Track C 단독 버전" 결과 하나만** 만듦 — 전체 25개 버전을 이 스크립트가 다시 만드는 게 아님. Track C로 좁혀서 다시 도는 이유는 두 가지지만 결과물은 하나임:

1. 원래 질문이 "Track C가 Track B를 앞서는 우위가 이 가정에 얼마나 취약한가"였는데, 전체 모델 기준 결과로는 정확히 답할 수 없었음.
2. gain만으로 내린 "credit이 다른 feature로 옮겨간다"는 해석이 맞는지, `sms_to_call`/`sms_to_call_gap` 사례처럼 gain의 착시가 아닌지 permutation importance로 재확인해야 했음.

CLI 모드가 아니라 `python -X utf8 -m Simulator.Detection.url_reliability_recheck`로 실행.

하는 일 (2단계 구성 — "얼마나 흔들리는가"와 "왜 흔들리는가"를 나눠서 봄):

- `NORMAL_IS_RELIABLE_URL`(1.0→0.5)/`PHISHING_IS_RELIABLE_URL`(0.0→0.5)을 5단계(baseline/mild/moderate/strong/extreme)로 스윕(원래 stress 테스트와 동일한 레벨). Track C(`Track.DEVICE`+`Track.DEVICE_STRUCTURAL`, 21개 — 확정 파생 4개 포함)로 스코프 고정.
- **[1단계 — 얼마나]** `sensitivity_analysis.run_stress_sensitivity(..., feature_cols=Track C 21개)`로 Track C만의 K-Fold 성능(F1/PR-AUC 등 ± SEM)을 재산출. 각 fold의 `confusion_matrix`에서 TN/FP/FN/TP 평균도 직접 뽑아 붙임(`summarize_fold_results()`는 스칼라만 남기고 버리므로 원본 `fold_results`에서 따로 추출). 이 K-Fold 결과는 `sensitivity_results/stress_is_reliable_url_trackC.json`에도 자동 저장되지만, 그 파일엔 FN이 없음(9개 스칼라만 — 프로젝트 공통 저장 포맷).
- **[2단계 — 왜]** 레벨별로 단일 70/30 split 학습 후, `is_reliable_url` + gain이 올랐던 "대체 후보" 5개(`msg_number_official_match`/`has_appinstall_link`/`cold_contact`/`structural_phishing_score`/`is_sequential_callers`)가 `is_reliable_url`의 빈자리를 진짜로 부분 복구하는지 gain과 permutation importance를 나란히 기록해 비교. `run_stress_sensitivity()`는 fold별 성능 요약만 반환하고 개별 feature의 permutation importance는 계산/보관하지 않으므로, 1단계와 별개로 `train_model()`→`evaluate()`를 직접 호출한 뒤 `permutation_importance()`를 추가로 구함.
- 1단계·2단계는 서로 데이터를 주고받진 않지만(코드상 독립) 같은 모델을 두 각도(성능/feature 기여도)에서 보는 것으로 **함께 해석**되므로, 하이퍼파라미터(둘 다 기본값)·PR-AUC 계산 방식(둘 다 사다리꼴, `train_eval.py::pr_auc()`)을 동일하게 맞춤(2026-09-23) — 기준이 다른 두 모델의 결과를 하나의 이야기로 엮어 읽는 걸 방지.

최종 산출물: `url_reliability_recheck_results/url_reliability_recheck.json`(1단계 K-Fold 성능+FN, 2단계 gain/permutation 전부 포함).

**이 파일은 `Generation/config.py`를 import하지 않아야 함.** (`main.py`의 dataset 캐시·하이퍼파라미터 로더만 사용, config 값 오버라이드는 `sensitivity_analysis._temporary_config_overrides()` 재사용)

---

### `is_reliable_url` stress 재검증 결과

**배경**: Track C의 우위가 `is_reliable_url`(정상/피싱 URL 신뢰도) 가정에 얼마나 의존하는지 확인하기 위해, `NORMAL_IS_RELIABLE_URL`/`PHISHING_IS_RELIABLE_URL`을 확정값(1.0/0.0)에서 점점 흐리는 stress 테스트를 새로 만들어 실행함(기존 `url_rate` stress는 "URL 등장 빈도"만 흔들어서 이 질문에 답할 수 없었음).

아래는 서로 다른 두 번의 실행 결과임(한 스크립트/파일에서 둘 다 나온 게 아님 — 혼동하기 쉬운 부분이라 명시):

- **① 전체 25개 feature 기준**: 처음 실행한 stress 테스트 결과. 스크립트로 저장된 적 없이 아래 표로만 기록돼 있음.
- **② Track C 21개 기준**: `url_reliability_recheck.py`(2026-09-23 추가)로 ①을 "Track C만 놓고 보면 어떤가"로 다시 실행한 결과. `url_reliability_recheck_results/url_reliability_recheck.json` + `sensitivity_results/stress_is_reliable_url_trackC.json`에 저장됨.

**① 성능(K-fold, 전체 25개 feature 기준)**:

| level    | F1          | PR-AUC      | Recall | FN(평균) |
| -------- | ----------- | ----------- | ------ | -------- |
| baseline | 0.957±0.003 | 0.977±0.001 | 0.940  | 11.8     |
| mild     | 0.928±0.005 | 0.962±0.002 | 0.905  | 18.8     |
| moderate | 0.916±0.004 | 0.954±0.004 | 0.880  | 23.6     |
| strong   | 0.911±0.007 | 0.949±0.005 | 0.877  | 24.2     |
| extreme  | 0.905±0.007 | 0.945±0.005 | 0.872  | 25.2     |

- SEM 대비 baseline→mild 차이가 F1은 5배, PR-AUC는 6.7배 커서 노이즈가 아닌 실제 효과로 판단.
- 구간별 하락폭(F1 −0.029/−0.012/−0.005/−0.006)을 보면 전체 하락의 절반 이상이 **가장 약한 단계(mild, 5%만 흐림)**에서 이미 발생 — "극단값에서만 위험하다"가 아니라 "mild(baseline보다 현실적인 가정)에서 이미 우위를 상당히 잃는다"로 해석해야 함.
- FN이 11.8→25.2건(약 2.1배)으로 늘어나는 게, F1 하락폭(~5pt)보다 이 프로젝트의 평가 기준(FN 비용 > FP 비용)에 더 부합하는 요약임.

**② 성능(K-fold, Track C 21개 기준, `url_reliability_recheck.py` 1단계, 2026-09-23 하이퍼파라미터·PR-AUC 방식 통일 후 재실행)**:

| level    | F1          | PR-AUC      | Recall | FN(평균) |
| -------- | ----------- | ----------- | ------ | -------- |
| baseline | 0.957±0.002 | 0.976±0.001 | 0.940  | 11.8     |
| mild     | 0.927±0.004 | 0.962±0.003 | 0.912  | 17.4     |
| moderate | 0.919±0.005 | 0.955±0.004 | 0.885  | 22.8     |
| strong   | 0.911±0.007 | 0.949±0.005 | 0.879  | 23.8     |
| extreme  | 0.909±0.008 | 0.946±0.006 | 0.876  | 24.4     |

①(전체 25개)과 방향·패턴이 거의 동일함(baseline→mild 구간에서 하락 대부분 발생, FN이 baseline 대비 약 2배로 증가) — **스코프를 Track C로 좁혀도 결론이 바뀌지 않음**을 확인. (이전 기록치와 소수점 둘째~셋째 자리 수준 차이가 있는데, 이 표는 애초부터 기본 하이퍼파라미터+사다리꼴만 썼던 단계라 방법론 변경 때문이 아니라 `tree_method="hist"`의 실행 간 비결정성 때문임 — Track A/B/C 비교 한계에서 이미 확인된 현상과 동일)

**② permutation importance 재검증(`url_reliability_recheck.py` 2단계, Track C 21개 기준, 2026-09-23 재실행 — 1단계와 동일하게 기본 하이퍼파라미터+사다리꼴 PR-AUC로 통일)**:

| feature                     | baseline | mild   | moderate | strong | extreme | 판정                                                |
| --------------------------- | -------- | ------ | -------- | ------ | ------- | --------------------------------------------------- |
| `is_reliable_url`           | 0.2916   | 0.0543 | 0.0162   | 0.0010 | 0.0003  | (기준) 확실히 붕괴                                  |
| `msg_number_official_match` | 0.0395   | 0.1047 | 0.1391   | 0.1244 | 0.1176  | **진짜 대체 확인**(baseline 대비 약 3배)            |
| `is_sequential_callers`     | 0.0198   | 0.0297 | 0.0360   | 0.0391 | 0.0405  | **진짜 대체 확인**(baseline 대비 약 2배, 단조 증가) |
| `has_appinstall_link`       | 0.00001  | 0.0051 | 0.0062   | 0.0071 | 0.0105  | 오르나 절대값 미미(≤0.011)                          |
| `structural_phishing_score` | 0.0090   | 0.0106 | 0.0143   | 0.0220 | 0.0051  | 추세 불안정(strong에서 peak 후 extreme서 급락)      |
| `cold_contact`              | 0.0011   | 0.0067 | 0.0034   | 0.0060 | 0.0035  | 값 작고 추세 불안정 — 유의미한 대체 아님            |

튜닝된 하이퍼파라미터로 실행했던 이전 버전과 비교하면(구체 수치는 본문 원본 대신 이 표로 대체됨):

- **핵심 결론(2/5만 진짜 대체)은 하이퍼파라미터를 바꿔도 그대로 유지됨** — `msg_number_official_match`/`is_sequential_callers`만 baseline 대비 2~3배로 뚜렷하게 오르고, 나머지 3개는 여전히 절대값이 작거나(`has_appinstall_link`) 추세가 불안정함(`structural_phishing_score`, `cold_contact`). 두 하이퍼파라미터 설정에서 같은 결론이 나온다는 것 자체가 이 결론이 하이퍼파라미터 선택의 우연한 산물이 아님을 보여줌.
- 다만 `is_reliable_url` 자체의 baseline permutation importance는 튜닝된 하이퍼파라미터(0.153)보다 기본 하이퍼파라미터(0.292)에서 거의 2배 더 높게 나옴 — 튜닝된 모델은 여러 feature에 신호를 더 고르게 분산시키는 반면, 기본 하이퍼파라미터 모델은 `is_reliable_url` 하나에 더 강하게 의존한다는 뜻. 두 경우 모두 baseline에서 `is_reliable_url`이 Track C 21개 중 가장 중요한 단일 feature라는 방향성은 동일함.
- `cold_contact`는 이전 실행에서 "extreme이 baseline보다 낮게 끝남"이라고 기록했으나, 이번 재실행에서는 반대 방향(extreme 0.0035 > baseline 0.0011)이 나옴 — 값 자체가 노이즈 수준(0.001~0.007)이라 방향을 특정하는 서술은 부정확했고, "작고 불안정해서 유의미한 대체로 보기 어렵다" 정도로만 해석하는 게 맞음.

**결론**: gain만 보면 5개 후보 전부 credit이 옮겨간 것처럼 보이지만, permutation으로 재확인하면 **`msg_number_official_match`/`is_sequential_callers` 2개만 실제 대체가 확인**되고 나머지 3개는 gain의 착시 — 이 결론은 하이퍼파라미터를 기본값으로 바꿔도 재현됨. 또한 baseline에서 gain(0.36)과 permutation(0.29) 모두 `is_reliable_url`을 Track C 21개 중 최상위로 꼽아 "Track C의 우위가 이 가정 하나에 상당히 의존한다"는 그림도 재확인됨. Track C 21개 기준 F1은 baseline 0.9466→extreme 0.8961(약 −5.3%p)로, 전체 feature set 기준(①) 결과와 "mild 단계에서 이미 하락 대부분 발생 후 정체"라는 패턴이 일치함(스코프·하이퍼파라미터 차이로 인한 결론 변화 없음).

---

### 최종 학습 결과 (`-m final`)

조건: N=100,000, train/test 70/30, 모델 XGBoost. Optuna val PR-AUC는 XGBoost 0.9941 > LightGBM 0.9935. `evaluate(pr_auc_method="average_precision")`. `is_global`(number_type과 100% 중복)/`sms_to_call`(permutation importance≈0) 제외 후 Track B 12개 / Track C 19개로 재학습.

| Track               | precision | recall | f1     | PR-AUC | Lift@Top5% | Lift@Top10% | FN (미탐) |
| ------------------- | --------- | ------ | ------ | ------ | ---------- | ----------- | --------- |
| B (단말만)          | 0.9856    | 0.9161 | 0.9496 | 0.9695 | 19.66      | 9.87        | 25/298    |
| C (단말+구조적신호) | 1.0000    | 0.9698 | 0.9847 | 0.9857 | 19.80      | 9.90        | 9/298     |

**결정**: 메인 보고 모델은 Track C.

- `is_global`은 `number_type`과 100% 중복, `sms_to_call`은 `sms_to_call_gap`의 조건부 파생이라 둘 다 gain·permutation 모두 0 — 최종 재학습에서 제외.
- 피처 중요도 보고 기준은 permutation(PR-AUC). Track B 상위는 `number_type` > `sms_to_call_gap` > `hour_bucket`. Track C 상위는 `is_reliable_url` > `number_type` > `sms_to_call_gap`(구조적 신호가 1위로 올라섬).

---

#### confidence calibration의 적용 여부

α(원본)만으로는 calibration의 효과를 알 수 없으므로, 학습 데이터가 동일하고 calibration 적용 여부만 다른 β(calibration 적용 전)와 γ(calibration 적용 후)를 비교한다(설계는 위 "confidence calibration(α/β/γ 분리)" 참고, 산출: `-m calib_compare` → `calibration_results/final_calibration_compare.json`). 아래 차이표의 모든 값은 **"이후 − 이전"**(γ−β, β−α, γ−α) 방향으로 계산했다.

| Track | 버전 | threshold | accuracy | precision | recall | f1     | PR-AUC | Lift@Top5% | Lift@Top10% |
| ----- | ---- | --------- | -------- | --------- | ------ | ------ | ------ | ---------- | ----------- |
| B     | α    | 0.9497    | 0.9991   | 0.9891    | 0.9161 | 0.9512 | 0.9727 | 19.66      | 9.87        |
| B     | β    | 0.9403    | 0.9990   | 0.9856    | 0.9161 | 0.9496 | 0.9743 | 19.66      | 9.87        |
| B     | γ    | 0.3937    | 0.9990   | 0.9856    | 0.9161 | 0.9496 | 0.9695 | 19.66      | 9.87        |
| C     | α    | 0.9061    | 0.9996   | 0.9931    | 0.9664 | 0.9796 | 0.9911 | 19.93      | 10.00       |
| C     | β    | 0.9493    | 0.9996   | 1.0000    | 0.9564 | 0.9777 | 0.9899 | 19.87      | 10.00       |
| C     | γ    | 0.6643    | 0.9997   | 1.0000    | 0.9698 | 0.9847 | 0.9857 | 19.80      | 9.90        |

| Track | 비교                    | threshold | accuracy | precision | recall  | f1      | PR-AUC  | Lift@Top5% | Lift@Top10% |
| ----- | ----------------------- | --------- | -------- | --------- | ------- | ------- | ------- | ---------- | ----------- |
| B     | β−α(데이터 축소 효과)   | −0.0094   | 0.0000   | −0.0036   | 0.0000  | −0.0017 | +0.0016 | 0.00       | 0.00        |
| B     | γ−β(calibration 효과)   | −0.5466   | 0.0000   | 0.0000    | 0.0000  | 0.0000  | −0.0048 | 0.00       | 0.00        |
| B     | γ−α(원본과 보정 총차이) | −0.5560   | 0.0000   | −0.0036   | 0.0000  | −0.0017 | −0.0032 | 0.00       | 0.00        |
| C     | β−α(데이터 축소 효과)   | +0.0432   | 0.0000   | +0.0069   | −0.0101 | −0.0019 | −0.0012 | −0.06      | 0.00        |
| C     | γ−β(calibration 효과)   | −0.2850   | +0.0001  | 0.0000    | +0.0134 | +0.0070 | −0.0042 | −0.07      | −0.10       |
| C     | γ−α(원본과 보정 총차이) | −0.2418   | +0.0001  | +0.0069   | +0.0034 | +0.0051 | −0.0054 | −0.13      | −0.10       |

**해석**: γ−β(순수 calibration 효과)는 두 Track 모두 PR-AUC를 0.005 이내로만 낮추고 precision/Lift는 거의 그대로다(Track C recall/F1만 소폭 상승, test 피싱 298건 중 4건 차이). threshold만 큰 폭으로 이동(B: 0.94→0.39, C: 0.95→0.66)하는데, 이는 판별 능력 변화가 아니라 `scale_pos_weight`로 부풀려져 있던 확률 척도가 실제 발생 빈도에 맞게 재조정된 결과다(확률 자체의 정직성 확인은 `eval_confidence.py`의 ECE/Brier 참고 — 같은 β/γ를 비교해 Track B는 ECE 0.0262→0.0002, Brier 0.0068→0.0008, Track C는 ECE 0.0126→0.0001, Brier 0.0024→0.0004로 감소). β−α(데이터 축소 효과)는 두 Track 모두 지표 변화가 ±0.01 이내로 작아, calibration을 켜기 위해 학습 데이터를 줄이는 것 자체의 손해는 미미하다. γ−α(총효과)는 두 효과가 합쳐진 값으로, calibration 파이프라인 전체를 원본과 비교할 때 사용한다.

---

## 결과물 저장 폴더

코드를 실행하면 자동으로 생성되는 폴더들. 소스 코드가 아니라 실행 결과물임.

| 폴더                               | 저장 내용                                                                                                                                                                          | git 추적 여부                                                |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `DataSet/`                         | 최종 dataset(`final_dataset.csv`/`.parquet`/`.meta.json`). `-m generate`/`-m final` 실행 시 생성.                                                                                  | 추적됨(커밋 대상)                                            |
| `sensitivity_results/`             | 민감도분석 모드(`sen`/`each_sen`/`ratio`/`stress`) 결과 json.                                                                                                                      | `.gitignore` 처리(재실행하면 다시 생성되므로 커밋 대상 아님) |
| `feature_selection_results/`       | ablation 모드 결과 json(`feature_selection_pipeline.json`).                                                                                                                        | `.gitignore` 처리                                            |
| `feature_stability_results/`       | ablation_stability 모드 결과 json(`feature_stability_check.json` — baseline summary + 조합별(`combos`) summary/diff). `feature_selection_results/`와 성격이 달라 별도 폴더로 분리. | `.gitignore` 처리                                            |
| `optuna_results/`                  | tune 모드 결과 json(`best_params.json` — winner 모델+하이퍼파라미터).                                                                                                              | `.gitignore` 처리                                            |
| `borderline_recheck_results/`      | `-m final` 경계선 재확인 json(`borderline_recheck.json` — 약한 파생 제거 비교, `is_global` 빈도, permutation importance).                                                          | `.gitignore` 처리                                            |
| `url_reliability_recheck_results/` | `is_reliable_url` stress 재검증 json(`url_reliability_recheck.json` — Track C 21개 기준, 레벨별 gain/permutation importance).                                                      | `.gitignore` 처리                                            |
| `calibration_results/`             | confidence calibration 관련 산출물. `main.py`: `final_calibration_compare.json`(`calib_compare`). `eval_confidence.py`: `tail_reliability_diagram.png`, `confidence_metrics.json`. | `.gitignore` 처리                                            |
| `plot_results/`                    | `plot_curve.py` 산출물 — `learning_curve_track{B,C}_{raw,calibrated}.png`, `cumulative_lift_track{B,C}_{raw,calibrated}.png`(총 8개, 보고서에는 이 중 Track C calibrated만 사용).  | `.gitignore` 처리                                            |

---

## Algorithm1과 파일의 대응 관계 (참고용)

| Algorithm1 줄                             | 담당 파일                                                          |
| ----------------------------------------- | ------------------------------------------------------------------ |
| Require (n, phishing_rate, config)        | `main.py`에서 값을 정해서 `dataset_builder.build_dataset()`에 전달 |
| 1~4, 8, 11, 12~17 (전체 루프·분기·라벨링) | `dataset_builder.py`                                               |
| 5~7 (피싱 이벤트 생성)                    | `phishing_generator.py`                                            |
| 9~10 (정상 이벤트 생성)                   | `normal_generator.py`                                              |
| 18 (반환)                                 | `dataset_builder.py`의 리턴값                                      |

---
