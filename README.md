# Infant fMRI imputation

Infant movie fMRI의 불규칙한 시간 결측을 STSR attention과 INR로 복원하는 독립 연구 코드입니다.
원래 TR 격자를 유지하고, 정상 관측 중 가린 값만 학습 정답으로 사용합니다.
주 실험은 **raw NIfTI 기반 처리 + STSR 무작위 초기화**입니다. 사전학습 가중치를
사용하지 않습니다. [Raw 파이프라인과 검증 조건](docs/raw_pipeline.md)을 먼저 확인하세요.

2026-11-08: 영화·gaze 정렬을 요구하지 않는 [탐색적 raw BOLD QC 코호트](docs/raw_qc_cohort.md)를
추가했습니다. 실행 설정은 `configs/infant_raw_qc_imputation.yaml`입니다.
이 코호트는 raw 촬영 시간과 새 motion/QC 규칙을 사용하며, movie decoding용으로
시간 정렬이 검증된 자료와 구분합니다. Gaze 상태는 미확정으로 보존합니다.

`fmri_stsr`의 `feat/infant-irregular-imputation` 브랜치,
커밋 `05a332bd68bfd263c6e9d039691b1bf66ea00a8b`에서 필요한 코드만 추출했습니다.
새 Git 이력을 사용하며 원본 checkout에 연결하지 않습니다. `stsr` 패키지 이름은
checkpoint 호환을 위해 유지합니다. [파일별 출처](docs/source_provenance.json)를 기록했습니다.

## 설치와 테스트

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,raw]'
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m pytest -q
```

이미 필요한 라이브러리를 가진 HPC 환경에서는 그 Python으로
`python -m venv --system-site-packages .venv`를 만든 뒤
`python -m pip install --no-deps --no-build-isolation -e .`로 설치할 수 있습니다.
이 경우 라이브러리는 기존 환경을 공유하고 프로젝트 코드는 이 저장소에서 읽습니다.

## 데이터와 실행

원본 데이터는 `/pscratch/sd/s/suincho/data/infant_movies_dryad_jm63xsjm3`에서 읽습니다.
데이터·체크포인트·실행 산출물은 Git에 넣지 않습니다.

```bash
infant-audit-raw \
  --data-root /pscratch/sd/s/suincho/data/infant_movies_dryad_jm63xsjm3 \
  --split outputs/mask_audit/cohort_split.json --out outputs/new_raw_audit

# 검증된 raw 시간 대응, motion/gaze annotation, anatomical ROI가 필요합니다.
infant-prepare \
  --selection verified_raw_selection.json \
  --data-root /pscratch/sd/s/suincho/data/infant_movies_dryad_jm63xsjm3 \
  --out outputs/new_raw_prepared

infant-impute train \
  --manifest outputs/new_raw_prepared/manifest.json \
  --config configs/infant_raw_imputation.yaml \
  --device cuda --out outputs/new_raw_training
```

현재 raw 감사 결과는 영아 94기록/98파일이며, burn-in 파일과 raw 시간·annotation
대응이 확인되지 않아 영화·gaze 조건의 코호트는 준비되지 않았습니다. 감사 도구의
`selection_unverified.json`은 학습에 바로 사용할 수 없습니다.

기존 공개 전처리 자료의 EDA와 feasibility pilot은 별도 명령으로 유지합니다.
기존 adapter의 새 이름은 `infant-prepare-preprocessed-pilot`입니다.

```bash
infant-audit-masks \
  --data-root /pscratch/sd/s/suincho/data/infant_movies_dryad_jm63xsjm3 \
  --eye-shift-tr 0 --guard-tr 0 --out outputs/new_mask_audit

infant-eda \
  --data-root /pscratch/sd/s/suincho/data/infant_movies_dryad_jm63xsjm3 \
  --audit-dir outputs/mask_audit --signal-samples-per-movie 1 \
  --out outputs/new_eda

infant-impute train \
  --manifest outputs/infant_prepared/manifest.json \
  --config configs/infant_irregular_imputation.yaml \
  --device cuda --out outputs/infant_pilot
```

현재 로컬 저장소에는 기존 마스크 분석을 `outputs/mask_audit/`로 복사했습니다.
[산출물 해시](docs/inherited_audit.json)를 보존했습니다. 이 폴더는 Git에 포함하지
않으므로 새 clone에서는 audit 명령으로 생성하고 EDA의 `--audit-dir`에 지정합니다.
출력 디렉터리는 새 경로를 사용합니다.

학습 전 해부학적 ROI와 가족 단위 train/val/test 선택이 필요합니다. `Child_Play`는
휴식 구간을 삭제해 이어 붙인 데이터라 원래 clip 경계를 확인하기 전까지 연속 시간
imputation에서 제외합니다. EDA의 고정 공간 probe는 해부학적 ROI가 아닙니다.

현재 구현·테스트·EDA 단계이며, 본학습을 통한 infant BOLD 복원이나 movie decoding
성능 향상은 아직 검증하지 않았습니다. 공개 전처리 신호에 남은 보간·시간 필터링
효과는 raw 데이터 평가와 구분해야 합니다.

[전체 방법과 기존 분석](docs/infant_irregular_imputation.md) · [초기 EDA 결과](docs/eda_initial.md)

[구현 타당성 점검 및 남은 과제](docs/implementation_review_2026-11-07.md):
평가 마스크 재현성, test 재분류 차단, 분석 범위 밖 tail의 모델 계산 제외를 수정했습니다.
평가 protocol v2는 별도 `evaluation_seed`를 사용합니다. 5-fold 교차검증과
최종 재학습은 아직 구현되지 않았습니다. Raw 경로에서는 고정 reference·annotation
조건에서 프레임 간 입력 불변성을 테스트했으며, 실제 코호트 전체의 QC와 시간
대응 검증은 남아 있습니다.
