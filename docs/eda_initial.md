# 초기 데이터 EDA

로컬 Dryad 자료의 5개 영화에서 infant metadata 전체를 분석하고, 훈련 분할에서
선택한 5개 run의 전처리 신호를 고정 공간 probe로 점검했습니다. 기존 가족 분할과
마스크 분석 결과를 재사용했으며 모델 학습이나 검증·테스트 분할 변경은 하지 않았습니다.

## 데이터 구성

| 항목 | 결과 |
|---|---:|
| Movie 기록 | 94개 |
| 독립 참가자 | 36명 |
| 가족 | 35가족 |
| 세션 ID | 75개 |
| 반복 세션이 있는 참가자 | 20명 |
| 참가자별 가장 어린 보고 연령 중앙값 | 7.55개월 |
| 전체 기록의 연령 범위 | 3.3–32.0개월 |

24개월을 넘는 기록 2개는 모두 `Child_Play`입니다. 참가자별 최초 보고 연령과 모든
기록을 중복 포함한 연령 분포를 구분해야 합니다. 같은 세션의 영화 간 연령·성별
metadata 충돌은 발견하지 못했습니다. 성인 비교 자료는 81개 movie 기록이 있으며
이번 infant 연령 통계에 섞지 않았습니다.

| 영화 | 기록 | 참가자 | 유효 프레임 비율 중앙값¹ |
|---|---:|---:|---:|
| Aeronaut | 44 | 27 | 90.0% |
| Catepillar | 10 | 10 | 92.5% |
| Child_Play | 22 | 19 | 95.6% |
| Meerkats | 10 | 10 | 87.8% |
| Mouseforsale | 8 | 8 | 97.2% |

¹ Gaze metadata가 있는 기록에서 motion·gaze confound로 제외되지 않은 프레임의 비율입니다.
영화 간 같은 참가자가 겹치므로 참가자 열을 합산해 전체 참가자 수로 쓰면 안 됩니다.
`clean`은 confound상 유효하다는 뜻이며 raw signal에 모든 artifact가 없다는 보장은 아닙니다.

## 유효 관측과 분할

CSV `num_TR` 안의 10,929개 프레임은 motion/gaze 공동 상태 기준으로 clean 9,399,
motion-only 476, gaze-only 535, 둘 다 제외 225, gaze 정보 미상 294개입니다.
시선 metadata가 없는 기록은 `Aeronaut:s6607_1_2`, `Child_Play:s5477_1_2`입니다.

`Child_Play`는 휴식이 삭제된 네 clip의 연결 자료라 현재 연속 시간 실험에서 제외합니다.
남은 적격 데이터는 train 46기록/19가족, validation 6기록/2가족, test 19기록/5가족입니다.
검증 가족 수가 작아 후속 성능 비교에는 가족 단위 반복 분할·불확실성 평가가 필요합니다.

정상 프레임만으로 연속 16 TR(32초)을 가리고도 최소 두 관측을 남길 수 있는 구간은
적격 71개 기록 중 66개에서 존재했습니다. 시작에서 가능한 기록은 34개, 끝에서는
27개였습니다. 이것은 **완전히 관측된 인위적 holdout의 가용성**이며, 실제 오염 구간의
복원 정확도가 아닙니다. 기존 스트레스 평가는 일부 정답만 있는 구간도 별도 분모로 평가합니다.

## 시간축과 신호 QC

90 TR보다 짧은 연속 영화 기록은 Aeronaut `s7057_1_1` 88 TR,
Catepillar `s0057_1_3` 85 TR, Mouseforsale `s2057_1_5` 58 TR입니다.

각 연속 영화에서 ID 순으로 첫 훈련 run, 가장 짧은 run, motion이 가장 많은 run을
선택한 후 중복을 제거해 5개를 점검했습니다. 각 run에서 영상 중앙 범위의 고정 격자
최대 216개 voxel을 읽었습니다. 비영점 관측이 있는 probe는 run당 39–50개였습니다.
해부학적 visual ROI나 전체 뇌 QC를 의미하지 않습니다.

- 검사한 probe에서는 NaN/Inf를 발견하지 못했습니다.
- Catepillar의 해당 run은 frame 85–92, Mouseforsale은 58–92에서 probe가 모두 0이었습니다.
- 나머지 세 run에서는 CSV `num_TR=90` 뒤의 frame 90–92에도 신호가 남아 있었습니다.
- 따라서 CSV 뒤의 구간을 일괄적으로 실제 무촬영/zero padding으로 해석하면 안 됩니다.
  기존 `padding` 필드는 현재 분석에서 제외하는 tail을 뜻하며, raw-to-movie 정렬을 확인하기
  전까지 보수적으로 입력·정답·복원 대상에서 제외합니다. 이번 EDA로 모델 입력 정책을 바꾸지는 않았습니다.

전처리된 `_Z` 신호에서 관측 표본의 표준편차 중앙값은 약 0.94–1.00이었습니다.
정규화된 데이터이므로 tSNR을 품질 지표로 해석하지 않았습니다. 그림의 adjacent-difference
RMS도 sparse probe의 변화량이며 표준화된 whole-brain DVARS가 아닙니다.

## 산출물과 재현

로컬 산출물은 `outputs/eda_initial/`에 있습니다:

- `summary.json`, `records.csv`, `records.json`: 참가자·영화·분할별 통계와 분모
- `cohort_overview.png`, `.pdf`: 데이터 구성·연령·motion/gaze·유효 시간
- `signal_qc.json`, `signal_qc.png`, `.pdf`: 선택한 훈련 run의 공간 probe QC
- `completion.json`: 완료 여부와 출력 파일 해시

기존 마스크 audit와 참가자 CSV의 해시를 비교하고, NIfTI의 시간 길이/TR도 대조했습니다.
원본 ZIP은 변경하지 않았습니다. 실행 명령은 README의 `infant-eda` 항목을 사용합니다.
고정 probe를 학습 ROI로 사용하거나 전처리 후 holdout 평가를 raw ground truth 평가로
해석하지 않습니다. 다음 단계는 해부학적 ROI 선정과 원래 시간축·전처리 누출 검증입니다.
