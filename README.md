# Infant fMRI imputation

Infant movie fMRI의 불규칙한 시간 결측을 STSR attention과 INR로 복원하는 독립 연구 코드입니다.
원래 TR 격자를 유지하고, 정상 관측 중 가린 값만 학습 정답으로 사용합니다.

`fmri_stsr`의 `feat/infant-irregular-imputation` 브랜치,
커밋 `05a332bd68bfd263c6e9d039691b1bf66ea00a8b`에서 필요한 코드만 추출했습니다.
새 Git 이력을 사용하며 원본 checkout에 연결하지 않습니다. `stsr` 패키지 이름은
checkpoint 호환을 위해 유지합니다. [파일별 출처](docs/source_provenance.json)를 기록했습니다.

## 설치와 테스트

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
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
