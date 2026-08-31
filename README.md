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

`train_baseline.py`도 시작할 때 같은 검사를 한 번 더 실행합니다. 검사에서
`experiment_id` 또는 `image_name`이 train/public_val 사이에 겹치면 학습을 중단합니다.

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
| `check_data.py` | 학습 전 데이터·라벨·experiment_id·image_name 누수 검사 |
| `train_baseline.py` | ResNet18 학습, 최고 체크포인트 저장 |
| `predict_one.py` | 학습한 체크포인트로 이미지 한 장 예측 |
| `00_quickstart_colab.ipynb` | 위 파일들을 순서대로 실행하는 메인 실습 노트북 |
| `02_cnn_theory_mnist.ipynb` | (선택) CNN 원리·학습 파라미터를 직접 실험해보는 이론 보충 노트북 |

> `predict_test.py` 는 이 저장소에 없습니다. **비공개 Test 이미지는 배포하지 않으며**,
> 채점은 운영진이 제출받은 체크포인트로 일괄 수행합니다. 아래 "최종 제출" 참고.

---

## (선택) CNN 원리를 직접 실습해보고 싶다면

`train_baseline.py`는 이미 학습된 ResNet18을 **빌려 쓰는** 전이학습이라, CNN 내부가 실제로
어떻게 학습되는지는 이 코드만으로 보기 어렵습니다. [02_cnn_theory_mnist.ipynb](02_cnn_theory_mnist.ipynb)
는 MNIST 손글씨 숫자로 아주 작은 CNN을 처음부터 학습시키며 다음을 직접 확인하는 보충 실습입니다.

- CNN 구조(Conv → Pool → Dense)가 텐서 모양을 어떻게 바꾸는지
- epoch 수 / optimizer(sgd·adam·adamw) / activation(relu·leaky_relu·tanh·sigmoid) 을
  바꾸면 학습 곡선과 최종 정확도가 어떻게 달라지는지
- Confusion Matrix로 결과를 해석하는 법, 이미지 한 장을 직접 예측해보는 법

채점·제출과는 무관하며, MNIST를 쓰는 이유는 다운로드가 즉시 끝나고 한 epoch이 몇 초 안에
돌아 구조와 파라미터 자체에 집중할 수 있기 때문입니다.

---

## 개선 실험

Baseline 대비 **한 번에 하나의 옵션만** 바꿉니다. 두 개를 동시에 바꾸면
점수가 왜 변했는지 설명할 수 없고, 발표 점수에서 손해를 봅니다.

```bash
# A. 데이터 증강
python train_baseline.py --data-root data --augment --epochs 5 --output-dir outputs/exp_aug

# B. 클래스 가중치 (적은 클래스의 오답에 큰 손실)
python train_baseline.py --data-root data --use-class-weights --epochs 5 --output-dir outputs/exp_weight

# C1. 미세조정 대조군: backbone은 고정하고 learning rate만 낮춤
python train_baseline.py --data-root data --lr 1e-4 --epochs 5 --output-dir outputs/exp_lr_control

# C2. 전체 미세조정: C1과 같은 learning rate에서 unfreeze만 추가
python train_baseline.py --data-root data --unfreeze --lr 1e-4 --epochs 5 --output-dir outputs/exp_ft

# D. 옵티마이저 변경 (같은 lr·epoch에서 optimizer만 변경)
python train_baseline.py --data-root data --optimizer sgd --epochs 5 --output-dir outputs/exp_sgd
```

전체 옵션은 `python train_baseline.py --help`

> `--optimizer` 는 backbone을 고정한 채 head(Linear 하나)만 학습하는 기본 설정에서는
> 효과가 크지 않을 수 있습니다. optimizer 차이를 뚜렷하게 보고 싶다면
> `02_cnn_theory_mnist.ipynb` 에서 먼저 확인해보세요. 이 프로젝트에 `activation` 옵션이
> 없는 이유도 같습니다 — ResNet18 head는 활성화 함수 없는 `nn.Linear` 하나뿐이라
> 바꿀 대상이 없습니다. 정규화 강도를 바꾸고 싶다면 `--weight-decay`를 별도 실험으로
> 기록하세요. optimizer 비교의 기본값은 세 optimizer 모두 `weight_decay=0`으로 같습니다.
> 같은 `lr=1e-3`에서의 비교이므로 optimizer별 최고 성능 순위가 아니라 공통 조건의
> 수렴 차이를 보는 실험입니다. 미세조정은 C1과 C2를 비교해야 `unfreeze` 효과만 해석할 수 있습니다.

### 이미지 한 장 예측

학습이 끝난 뒤 Public Validation 이미지 하나를 직접 확인할 수 있습니다.

```bash
python predict_one.py \
  --image data/public_val/images/이미지파일명.png \
  --checkpoint outputs/baseline/best_model.pt
```

출력에는 예측 클래스와 초기·중기·후기별 확률이 함께 표시됩니다. 이 결과는 학습 과정을
이해하기 위한 예시이며, 비공개 Test 채점용 파일을 만드는 단계는 아닙니다.

**`--output-dir` 을 매번 다르게 지정하세요.** 같은 경로를 쓰면 이전 결과가 덮어써집니다.

실험은 `templates/experiment_log.csv` 를 복사해 한 줄씩 기록합니다.
실패한 실험도 지우지 마세요. 발표에서 근거가 됩니다.

> 운영진 Drive의 `ARCHIVE_grouped_results_2026-08-07` 폴더는 누수 없는 Grouped baseline을
> 실행했던 **비공개 보관 기록**입니다. 학생 배포 자료가 아닙니다.
> 당시 실행물과 현재 코드의 `history.csv` 열 구성·저장 파일이 다를 수 있으므로,
> 새 실험의 재현 근거는 항상 현재 코드가 생성한 `run_config.json`과 결과 파일을 사용하세요.

---

## 학습 결과 파일

`--output-dir` 안에 다음 여섯 개가 생깁니다.

| 파일 | 내용 |
|---|---|
| `best_model.pt` | Public Validation Macro F1 최고 시점의 가중치 |
| `history.csv` | epoch별 train loss, val loss, val accuracy, val macro F1 |
| `validation_report.json` | 최고 시점의 점수, Class F1, 혼동행렬 |
| `run_config.json` | 실행 옵션·장치·라이브러리·커밋·로컬 수정 diff·소스 해시 (재현성 근거) |
| `learning_curves.png` | epoch별 loss와 validation 점수 변화 그래프 |
| `confusion_matrix.png` | 최고 checkpoint의 Public Validation 혼동행렬 |

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
