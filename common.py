"""공통 유틸리티: 시드 고정, 이미지 전처리, 평가 지표.

이 파일은 참가자가 직접 수정할 필요가 없습니다.
train_baseline.py 와 predict_test.py 가 이 파일의 함수를 가져다 씁니다.
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torchvision import transforms

# 교육용 3개 클래스. 원본 6단계를 두 단계씩 묶은 것입니다.
CLASS_NAMES = ["초기", "중기", "후기"]
NUM_CLASSES = 3

# ImageNet 사전학습 모델이 기대하는 입력 통계값
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int = 42) -> None:
    """난수를 고정해 실험을 재현 가능하게 만듭니다.

    시드를 고정하지 않으면 같은 명령어를 두 번 돌려도 점수가 달라져서
    '내가 바꾼 옵션 때문에 오른 것'인지 '그냥 운'인지 구분할 수 없습니다.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_transforms(image_size: int = 224, train: bool = False, augment: bool = False):
    """이미지를 모델 입력 텐서로 바꾸는 전처리 파이프라인을 만듭니다.

    augment=True 일 때만 학습용 증강이 추가됩니다.
    검증/테스트에는 절대 증강을 넣지 않습니다. 평가 조건이 달라지기 때문입니다.

    [수정 규칙]
      아래 `train and augment` 블록은 자유롭게 바꿔도 됩니다. 학습에만 쓰입니다.

      그 아래 기본 블록은 검증과 최종 채점에 쓰입니다. 채점은 운영진 환경의
      기본 블록으로 실행되므로, 이 부분을 바꾸면 여러분의 모델이 학습 때와
      다른 조건으로 채점되어 점수가 부당하게 낮아집니다. 수정하지 마세요.
      입력 크기를 바꾸려면 코드가 아니라 --image-size 옵션을 사용하세요.
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if train and augment:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ])

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])


@torch.no_grad()
def evaluate(model, loader, device, criterion=None) -> dict:
    """모델을 평가 모드로 돌려 Accuracy / Macro F1 / Class F1 / 혼동행렬을 계산합니다."""
    model.eval()
    all_preds, all_targets = [], []
    total_loss, total_n = 0.0, 0

    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(images)
            if criterion is not None:
                loss = criterion(logits, targets)
                total_loss += loss.item() * images.size(0)

        total_n += images.size(0)
        all_preds.append(logits.argmax(dim=1).cpu())
        all_targets.append(targets.cpu())

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    return {
        "loss": total_loss / total_n if criterion is not None else None,
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "class_f1": [
            float(v) for v in f1_score(targets, preds, average=None,
                                       labels=list(range(NUM_CLASSES)), zero_division=0)
        ],
        "confusion_matrix": confusion_matrix(
            targets, preds, labels=list(range(NUM_CLASSES))
        ).tolist(),
    }


def format_confusion_matrix(cm) -> str:
    """혼동행렬을 터미널에서 읽기 쉽게 문자열로 만듭니다. 행=실제, 열=예측."""
    header = "            " + "".join(f"{'예측 ' + n:>10}" for n in CLASS_NAMES)
    lines = [header]
    for i, row in enumerate(cm):
        lines.append(f"{'실제 ' + CLASS_NAMES[i]:>10}  " + "".join(f"{v:>10}" for v in row))
    return "\n".join(lines)


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("[경고] GPU를 찾지 못했습니다. CPU로 실행하면 매우 느립니다.")
    return torch.device("cpu")
