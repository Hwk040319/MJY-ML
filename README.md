# 배터리 열폭주 이미지 분류 미니프로젝트

동력공학 발명 동아리 MJY · 2026

열화상 이미지 한 장을 보고 열폭주 진행 단계를 **초기(0) / 중기(1) / 후기(2)** 중 하나로
분류하는 모델을 만듭니다. ImageNet 사전학습 ResNet18을 전이학습하는 것이 공통 출발점입니다.

> 이 프로젝트의 3개 클래스는 학습용으로 묶은 교육용 라벨입니다.
> 실제 배터리 안전 판정 기준이 아닙니다.

---

## 5분 안에 시작하기 (Google Colab)

노트북으로 바로 시작하려면 [00_quickstart_colab.ipynb](00_quickstart_colab.ipynb) 를
Colab에서 열어 위에서부터 실행하세요. 아래는 같은 내용을 요약한 것입니다.

```python
import torch; print(torch.cuda.is_available())   # False 면 런타임 -> T4 GPU

!git clone https://github.com/Hwk040319/MJY-ML.git
%cd MJY-ML
!pip install -q -r requirements.txt

# 데이터는 Google Drive 공유 링크로 배포됩니다 (운영진 공지 확인)
!pip install -q gdown
FILE_ID = "13DWY5tg_L4SYxujkdVQ89qEQZC08lr7L"   # 1회차 강의 실습용 (22MB)
# FILE_ID = "1PJNyDDdYd47wXD83DiW9PqFzbe7TLl0n"   # 회차 사이 팀 실험용 (11GB)

!gdown "https://drive.google.com/uc?id=$FILE_ID" -O data.tar
!mkdir -p data && tar -xf data.tar -C data

!python check_data.py --data-root data
!python train_baseline.py --data-root data --output-dir outputs/baseline
```

| 시점 | 받을 파일 |
|---|---|
| 1회차 강의 실습 | `battery_lecture_sample.tar` (약 700장 · 22MB) |
| 회차 사이 팀 실험 | `battery_train_val.tar` (약 11GB) |

`private_test` 는 배포되지 않습니다. 상세 설정과 문제 해결은 [docs/SETUP.md](docs/SETUP.md) 를 보세요.

---

## 파일 구성

| 파일 | 역할 |
|---|---|
| `battery_dataset.py` | 이미지와 라벨을 짝지어 텐서로 변환 |
| `common.py` | 시드 고정, 전처리, 평가 지표 계산 |
| `check_data.py` | 학습 전 데이터·라벨·분할 누수 검사 |
| `train_baseline.py` | ResNet18 학습, 최고 체크포인트 저장 |

> `predict_test.py` 는 이 저장소에 없습니다. **비공개 Test 이미지는 배포하지 않으며**,
> 채점은 운영진이 제출받은 체크포인트로 일괄 수행합니다. 아래 "최종 제출" 참고.

---

## 개선 실험

Baseline 대비 **한 번에 하나의 옵션만** 바꿉니다. 두 개를 동시에 바꾸면
점수가 왜 변했는지 설명할 수 없고, 발표 점수에서 손해를 봅니다.

```bash
# A. 데이터 증강
python train_baseline.py --data-root data --augment --output-dir outputs/exp_aug

# B. 클래스 가중치 (적은 클래스의 오답에 큰 손실)
python train_baseline.py --data-root data --use-class-weights --output-dir outputs/exp_weight

# C. 전체 미세조정 (반드시 작은 learning rate 와 함께)
python train_baseline.py --data-root data --unfreeze --lr 1e-4 --output-dir outputs/exp_ft
```

전체 옵션은 `python train_baseline.py --help`

**`--output-dir` 을 매번 다르게 지정하세요.** 같은 경로를 쓰면 이전 결과가 덮어써집니다.

실험은 `templates/experiment_log.csv` 를 복사해 한 줄씩 기록합니다.
실패한 실험도 지우지 마세요. 발표에서 근거가 됩니다.

---

## 학습 결과 파일

`--output-dir` 안에 네 개가 생깁니다.

| 파일 | 내용 |
|---|---|
| `best_model.pt` | Public Validation Macro F1 최고 시점의 가중치 |
| `history.csv` | epoch별 train/val loss, accuracy, macro F1 |
| `validation_report.json` | 최고 시점의 점수, Class F1, 혼동행렬 |
| `run_config.json` | 실행 옵션 전체 (재현성 근거) |

`history.csv` 에서 train loss는 계속 내려가는데 val macro F1이 떨어지면 과적합입니다.

---

## 최종 제출

**비공개 Test 이미지는 여러분에게 전달되지 않습니다.** 직접 예측을 만들지 않고,
`outputs/<가장 좋았던 실험 폴더>/best_model.pt` 파일 **하나만** 제출합니다.
운영진이 모든 팀의 체크포인트를 모아 동일한 조건으로 채점합니다.

```python
# Colab 에서 다운로드
from google.colab import files
files.download('outputs/exp_aug/best_model.pt')
```

제출 방법·마감은 [docs/SUBMISSION.md](docs/SUBMISSION.md), 금지 사항은 [docs/RULES.md](docs/RULES.md) 를 보세요.

---

## 평가

```
성능점수 = 70 × (팀 Macro F1 ÷ 1위 팀 Macro F1)
발표점수 = 30 (문제 이해 / 실험 설계 / 결과 해석 / 한계 인식 각 7.5점)
총점     = 성능점수 + 발표점수
```

동점 시 Macro F1 → Accuracy → 제출 시각 순으로 순위를 가립니다.
