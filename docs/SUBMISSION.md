# 제출

## 왜 submission.csv 를 직접 안 만드나요

비공개 Test(private_test) 이미지는 채점 무결성을 위해 참가자에게 배포하지 않습니다.
따라서 각 팀은 예측 파일이 아니라 **모델 자체(체크포인트)** 를 제출합니다.
운영진이 모든 팀의 체크포인트를 동일한 조건(같은 private_test, 같은 스크립트)으로
일괄 실행해 순위를 계산합니다.

## 1. 모델 잠그기

`outputs/*/validation_report.json` 들의 `macro_f1` 을 비교해 가장 높은 것을 고릅니다.
고른 뒤에는 더 이상 학습하지 않습니다.

## 2. 제출 파일

**채점에 필요한 파일은 `best_model.pt` 하나뿐입니다.**

| 파일 | 경로 | 구분 |
|---|---|---|
| `best_model.pt` | 선택한 실험 폴더 안 | **필수 · 이것만 있으면 채점됩니다** |
| `experiment_log.csv` | 팀 실험 기록 | 발표 준비용 (제출은 선택) |
| `run_config.json` | 같은 폴더 안 (자동 생성) | 참고용 (제출 불필요) |
| 발표 자료 | 정해진 템플릿 | 2회차 당일 발표 시 사용 |

`experiment_log.csv` 와 `run_config.json` 을 내지 않아도 성능 점수는 그대로 채점됩니다.
다만 실험 기록이 없으면 발표에서 근거를 대기 어려워 발표 점수에서 불리합니다.

`best_model.pt` 안에는 이미 학습 시 사용한 `image_size` 등 설정이 함께 저장되어 있어
운영진 채점 스크립트가 자동으로 맞춰 읽습니다. 파일명을 바꾸지 마세요.

## 3. 제출 방법

Colab에서:
```python
from google.colab import files
files.download('outputs/exp_aug/best_model.pt')
```

내려받은 뒤 `[팀명]_best_model.pt` 로 이름을 바꿔
**[제출 링크/폼: 운영진 공지 확인]** 에 업로드합니다.

## 4. 마감

**2회차 전날 14:00** 까지 제출해야 합니다. 운영진은 그 이후 채점을 진행하므로
14:00 이후 제출은 채점되지 않습니다.

- 마감 전 24시간(전날 14:00 기준 이틀 전 14:00 이후)에는 신규 학습을 권장하지 않습니다 (`docs/RULES.md` 참고).
- 파일이 손상되었거나 로드되지 않으면 0점 처리됩니다. 제출 전
  아래 명령으로 로드가 되는지 스스로 확인하세요.

```python
import torch
ckpt = torch.load('outputs/exp_aug/best_model.pt', map_location='cpu', weights_only=False)
print(ckpt['macro_f1'], ckpt['args'])   # 오류 없이 출력되면 정상
```

## 채점 방식

```
성능점수 = 70 × (팀 Macro F1 ÷ 1위 팀 Macro F1)
발표점수 = 30 (문제 이해 / 실험 설계 / 결과 해석 / 한계 인식 각 7.5점)
총점     = 성능점수 + 발표점수
```

동점 시 Macro F1 → Accuracy → 제출 시각 순으로 순위를 가립니다.
