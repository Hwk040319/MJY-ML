# 환경 설정 (Google Colab)

이 프로젝트는 **Google Colab에서 실행**합니다. 로컬 PC(PowerShell)는 코드를 읽거나
`git clone` 하는 보조 용도로만 쓰고, GPU가 필요한 학습은 반드시 Colab에서 하세요.

---

## 1. GPU 확인

```python
import torch
print(torch.cuda.is_available())
```

`False` 가 나오면 상단 메뉴 **런타임 → 런타임 유형 변경 → T4 GPU** 선택 후 이 셀을 다시 실행하세요.

---

## 2. 코드 내려받기

```python
!git clone https://github.com/Hwk040319/MJY-ML.git
%cd MJY-ML
!pip install -q -r requirements.txt
```

---

## 3. 데이터 내려받기

데이터는 Google Drive로 배포됩니다. 링크는 운영진 공지를 확인하세요.

| 파일 | 내용 | 용량 |
|---|---|---|
| `battery_lecture_sample.tar` | 1회차 강의 실습용 (약 300장) | 작음 · 링크 별도 공지 |
| `battery_train_val.tar` | 팀별 실험용 전체 train + public_val | 약 11GB · FILE_ID `1PJNyDDdYd47wXD83DiW9PqFzbe7TLl0n` |

> `private_test` 는 배포되지 않습니다. 채점은 제출된 체크포인트로 운영진이 수행합니다.

### 방법 A · 공유 링크에서 직접 받기 (권장)

```python
!pip install -q gdown
# FILE_ID 는 공유 링크의 /d/ 와 /view 사이 문자열입니다.
!gdown "https://drive.google.com/uc?id=1PJNyDDdYd47wXD83DiW9PqFzbe7TLl0n" -O data.tar
!mkdir -p data && tar -xf data.tar -C data
!ls data
```

### 방법 B · 내 드라이브에 바로가기 추가 후 마운트

1. 공유 링크를 열고 **내 드라이브에 바로가기 추가**
2. Colab에서:

```python
from google.colab import drive
drive.mount('/content/drive')

# 대용량 파일은 드라이브에서 직접 풀지 말고 먼저 로컬로 복사하세요.
!cp "/content/drive/MyDrive/battery_train_val.tar" /content/data.tar
!mkdir -p data && tar -xf /content/data.tar -C data
```

**드라이브에서 tar 를 직접 `-xf` 하면 큰 파일에서 연결이 끊기는 경우가 있습니다.**
`Transport endpoint is not connected` 오류가 나면 위처럼 로컬 복사 후 압축을 푸세요.

---

## 4. 실행

```python
!python check_data.py --data-root data
!python train_baseline.py --data-root data --output-dir outputs/baseline
```

---

## 세션이 끊기면 데이터가 사라집니다

Colab의 `/content/` 는 런타임을 재시작하면 초기화됩니다. `/content/drive/` 는 유지됩니다.
매번 다시 받는 것이 번거로우면 압축 파일을 자기 드라이브에 한 번 복사해두고 재사용하세요.

---

## 문제 해결

| 증상 | 원인 | 해결 |
|---|---|---|
| `cuda.is_available()` 가 False | 런타임이 CPU | 런타임 유형을 T4 GPU로 변경 |
| `Transport endpoint is not connected` | 드라이브 마운트 끊김 | `drive.mount('/content/drive', force_remount=True)` |
| `No such file or directory` (압축 푼 폴더) | 세션 재시작으로 `/content/` 초기화됨 | 압축을 다시 푸세요 |
| `Download quota exceeded` | 같은 파일에 동시 접근 과다 | 24시간 후 재시도, 또는 운영진에게 새 링크 요청 |
| `FileNotFoundError: data/train/images` | 압축 해제 경로 불일치 | `!ls data` 로 실제 구조 확인 |
| 학습이 매우 느림 | GPU 미할당 | 1번 항목 다시 확인 |

---

## 운영진 메모 · 드라이브 다운로드 잠금 대응

같은 파일을 많은 사람이 동시에 받으면 구글이 최대 24시간 다운로드를 차단할 수 있습니다.
이때는 드라이브에서 해당 파일의 **사본을 만들고 새 링크로 재공유**하면 즉시 풀립니다.
사본은 별개 파일이라 할당량이 새로 시작됩니다.
