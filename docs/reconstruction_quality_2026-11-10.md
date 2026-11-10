# 실제 결측 복원과 decoding 없는 품질 평가 — 2026-11-10

Fold 0의 1,000-step checkpoint로 개발 촬영 39개에 inference를 적용했다. 그중 실제 복원 가능한 결측이 있는 32개 촬영에서 **172,454 voxel×시점 값**을 예측했다. 최소 한 voxel이 예측된 시점은 촬영별 합계 679개다. 원래 유효 관측 972,288개는 정확히 보존했고, 복원 대상으로 인정되지 않은 11,354개 값은 NaN으로 남겼다. 첫 분석 TR 제외와 공간 coverage 부족이 포함된다.

훈련 촬영 32개와 validation 촬영 7개의 출력은 membership으로 구분했다. 훈련 촬영의 출력은 일반화 성능 증거가 아니다. Reserved test 14개에는 inference하지 않았다. 현재 출력은 촬영마다 최대 256 TR, occipital ROI 256 voxel의 시계열이며 whole-brain NIfTI 복원이 아니다.

## 실제 출력

위치: `/pscratch/sd/s/suincho/data/infant_movies_dryad_jm63xsjm3/derived/raw_qc_v2_fold0_imputed_20261110`

각 `records/<movie>/<subject>/imputed.npz`에는 `values`(관측+STSR 복원), `valid`(원래 관측), `imputed`(실제 예측 위치), `unresolved`(미해결), `raw_frame_indices`, `voxel_ijk`, `coords_mm`, `times_s`, 원래 motion/signal/slice QC flags, checkpoint/input SHA256을 저장했다. `linear_values`와 `observed_mean_values`는 같은 위치에 적용한 비교 방법이다. 양쪽/한쪽 관측 여부와 관측까지의 시간 거리도 저장했다.

Validation 촬영 7개에는 `real_gap_plot.png`가 있다. voxel 0/128/255를 사전 지정해 보여 주며, 좋은 예측만 골라서 제시하지 않았다. 검은색은 관측, 보라색은 STSR, 주황색은 선형 보간, 청록색은 voxel별 관측 평균이다. 보라색 음영은 실제 결측 복원 위치다. **이 그림에 실제 결측의 깨끗한 정답은 없다.**

## 정답이 있는 검증

기존 validation 7개 촬영 / 4개 가족의 동일한 인위적 마스크 70개를 재구성했다. 기존 선형 보간의 target 수·MSE·정규화 MSE를 재계산해 일치함을 확인한 후 관측 평균 기준선을 추가했다. 평균과 표준편차는 가린 값을 제외한 관측만으로 계산했다.

| 방법 | 가족 평균 정규화 MSE ↓ | 가족 평균 voxel r ↑ |
|---|---:|---:|
| STSR | 1.2811 | 0.3262 |
| 선형 보간 | 1.5147 | 0.2972 |
| 양옆 유효값 평균 | 1.5059 | 0.2853 |
| voxel별 관측 평균 | 1.5109 | 상수 시계열이므로 정의 안 됨 |

마스크 → 촬영 → 가족 순서로 평균하여 가족에 동일 비중을 주었다. voxel r은 가린 시점에서 계산했으며, 서로 떨어진 시점이 포함될 수 있다. 이는 연속 구간의 spectrum/FC 보존 평가가 아니다. STSR은 평균 오차에서 우세하지만 모든 가족에서 모든 기준선보다 좋지는 않다. 이 checkpoint는 같은 validation으로 선택되었고 나머지 fold/seed 재현성과 독립 test 성능은 아직 확인하지 않았다.

## decoding 없이 평가할 항목

1. **주요 정확도:** QC 통과 관측을 가린 뒤 nMSE/MAE, voxel별 temporal correlation을 계산한다. 평균값·선형 보간과 동일 target을 사용한다. 결측 길이, leading/interior/trailing, 주변 관측 거리별로 분해한다. 현재 nMSE/MAE/r 및 길이·위치별 MSE가 구현돼 있다.
2. **시간 변화 보존:** 충분히 긴 연속 정답 구간을 확보하면 일차 차분 오차, autocorrelation 및 power-spectrum 차이를 비교한다. 불규칙한 target들을 이어 붙여 FFT를 계산하면 안 된다. PSD/ACF/차분 오차는 이번 출력에서 새로 계산하지 않았다.
3. **공간 관계 보존:** 같은 정답 구간에서 ROI voxel 간 correlation/covariance 행렬의 변화를 비교할 수 있다. 이것은 decoding 없이 공간 관계를 보는 보조 지표다. 현재 ROI는 whole-brain network가 아니며, 관측 그대로인 부분이 전체 점수를 지배하지 않도록 같은 마스크 비율에서 비교해야 한다. 이 분석은 아직 실행하지 않았다.
4. **재현성·불확실성:** 가족별 5-fold 및 여러 seed에서 paired 개선량을 확인하고, 실제 결측에 대해 모델 간 disagreement와 gap/context 조건을 함께 보고한다. 단일 모델의 보간과의 차이는 보정된 불확실성 구간이 아니다. 현재 결과는 한 fold/seed다.

## 실제 오염 구간에서 가능한 점검과 한계

깨끗한 정답이 없으므로 실제 오염 구간의 MSE/r은 계산하지 않았다. 관측값 보존, 예측의 finite 여부, 비정상 amplitude, 경계 jump, 관측까지의 거리, 선형 보간과의 차이를 진단값으로 저장했다. 이 값은 정확도 점수나 성공 판정이 아니다.

**DVARS 감소·tSNR 증가·매끄러운 파형만으로 성공이라고 결론 내리지 않는다.** [MRIQC 정의](https://mriqc.readthedocs.io/en/22.0.6/iqms/bold.html)에 따르면 DVARS는 시간 차분 크기를, tSNR은 평균/시간 표준편차를 사용한다. 따라서 신호를 평평하게 만드는 것만으로 이 지표가 좋아질 수 있다는 것이 수식에서 따르는 한계다. 신경 정보가 보존됐다는 증거와 구분해야 한다.

현재 결론은 '같은 인위적 결측에서 STSR의 관측값 예측 오차가 두 보간 방법 및 관측 평균보다 낮다'이다. 실제 motion 구간의 신경 신호를 정확히 회복했다는 결론은 아니다.
