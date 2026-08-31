"""실습 02 — ResNet18 전이학습 Baseline 학습.

모든 팀은 옵션 없이 이 파일을 한 번 돌려 같은 출발점을 만듭니다.
그 다음 개선 실험에서는 '한 번에 하나의 옵션만' 바꿉니다.

기본 실행 (Baseline):
    python train_baseline.py --data-root data --output-dir outputs/baseline

개선 실험 예시:
    python train_baseline.py --data-root data --augment           --output-dir outputs/exp_aug
    python train_baseline.py --data-root data --use-class-weights --output-dir outputs/exp_weight
    python train_baseline.py --data-root data --unfreeze --lr 1e-4 --output-dir outputs/exp_ft
    python train_baseline.py --data-root data --optimizer sgd      --output-dir outputs/exp_sgd
"""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from battery_dataset import BatteryDataset
from common import (CLASS_NAMES, NUM_CLASSES, evaluate, format_confusion_matrix,
                    get_transforms, resolve_device, save_json, set_seed)


def parse_args():
    ap = argparse.ArgumentParser(description="배터리 열화상 3-class 분류 Baseline")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--output-dir", default="outputs/baseline")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42,
                    help="공정한 비교를 위해 42 고정을 권장합니다")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--augment", action="store_true",
                    help="학습 데이터에 crop/flip/rotation/밝기 변화를 추가")
    ap.add_argument("--use-class-weights", action="store_true",
                    help="적은 클래스의 오답에 더 큰 손실을 부여")
    ap.add_argument("--unfreeze", action="store_true",
                    help="backbone 까지 전체 미세조정 (--lr 1e-4 정도를 함께 쓰세요)")
    ap.add_argument("--optimizer", choices=["adamw", "adam", "sgd"], default="adamw",
                    help="가중치를 어떤 규칙으로 업데이트할지 선택 (기본 adamw)")
    return ap.parse_args()


def build_model(unfreeze: bool, device):
    """ImageNet 사전학습 ResNet18 을 불러와 마지막 분류층만 3-class 로 교체합니다."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if not unfreeze:
        # backbone 고정: 사전학습된 특징 추출기를 그대로 쓰고 새 head 만 학습합니다.
        for param in model.parameters():
            param.requires_grad = False

    # fc 를 새로 만들면 requires_grad=True 인 새 파라미터가 됩니다.
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(device)


def build_optimizer(name: str, params, lr: float):
    """--optimizer 옵션에 따라 옵티마이저를 만듭니다.

    이 프로젝트는 backbone을 고정한 채 head(Linear 하나)만 학습하는 경우가 기본이라
    옵티마이저 차이가 CNN을 처음부터 학습할 때만큼 극적이지는 않습니다. 그래도 수렴
    속도와 최종 점수가 조금씩 달라지므로 개선 실험의 한 축으로 다뤄볼 수 있습니다.
    """
    if name == "sgd":
        # SGD는 관성(momentum)을 줘야 Adam 계열과 비슷한 속도로 수렴합니다.
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    return torch.optim.AdamW(params, lr=lr)  # 기본값: weight decay가 분리된 Adam


def freeze_bn(module):
    """BatchNorm 레이어를 eval 모드로 고정합니다.

    requires_grad=False 는 가중치 업데이트만 막을 뿐, model.train() 상태에서
    BatchNorm 의 running_mean/running_var 통계는 계속 갱신됩니다 (PyTorch 기본 동작).
    "backbone 고정"이 이름 그대로 동작하려면 BatchNorm 도 eval 모드로 묶어야 합니다.
    """
    if isinstance(module, nn.BatchNorm2d):
        module.eval()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.unfreeze and args.lr > 5e-4:
        print(f"[주의] --unfreeze 에 lr={args.lr} 는 큽니다. 1e-4 정도를 권장합니다.")

    if args.optimizer == "sgd" and args.lr <= 5e-4:
        print(f"[주의] --optimizer sgd 에 lr={args.lr} 는 작습니다. "
              f"SGD는 보통 adam 계열보다 큰 lr(예: 1e-2)이 필요합니다.")

    # ---------- 데이터 ----------
    train_set = BatteryDataset(
        args.data_root, "train",
        transform=get_transforms(args.image_size, train=True, augment=args.augment))
    val_set = BatteryDataset(
        args.data_root, "public_val",
        transform=get_transforms(args.image_size, train=False))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    counts = train_set.class_counts()
    print(f"Training {len(train_set)}장  {dict(zip(CLASS_NAMES, counts))}")
    print(f"Public Validation {len(val_set)}장")

    # ---------- 모델 / 손실 / 최적화 ----------
    model = build_model(args.unfreeze, device)

    if args.use_class_weights:
        # 적은 클래스일수록 큰 가중치. 전체수 / (클래스수 * 해당클래스수)
        total = sum(counts)
        weights = torch.tensor(
            [total / (NUM_CLASSES * c) if c > 0 else 0.0 for c in counts],
            dtype=torch.float32, device=device)
        print(f"class weights = {[round(w, 3) for w in weights.tolist()]}")
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(args.optimizer, trainable, args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    n_train = sum(p.numel() for p in trainable)
    print(f"학습 대상 파라미터 {n_train:,}개 "
          f"({'전체 미세조정' if args.unfreeze else 'head 만 학습'})  "
          f"optimizer={args.optimizer}\n")

    # ---------- 학습 루프 ----------
    history_path = out_dir / "history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1"])

    best_f1, best_report = -1.0, None

    for epoch in range(1, args.epochs + 1):
        model.train()
        if not args.unfreeze:
            model.apply(freeze_bn)  # backbone 고정 시 BatchNorm 통계도 함께 고정
        running, seen, t0 = 0.0, 0, time.time()

        for images, targets, _ in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(images)          # (1) 예측: 클래스별 점수 3개
                loss = criterion(logits, targets)  # (2) 정답과의 차이 측정

            scaler.scale(loss).backward()       # (3) 역전파: 각 가중치의 책임 계산
            scaler.step(optimizer)              # (4) 업데이트: 가중치 수정
            scaler.update()

            running += loss.item() * images.size(0)
            seen += images.size(0)

        train_loss = running / seen
        report = evaluate(model, val_loader, device, criterion)

        print(f"epoch {epoch}/{args.epochs}  "
              f"train_loss {train_loss:.4f}  val_loss {report['loss']:.4f}  "
              f"val_acc {report['accuracy']:.4f}  val_macroF1 {report['macro_f1']:.4f}  "
              f"({time.time() - t0:.0f}s)")

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, round(train_loss, 6),
                                    round(report["loss"], 6),
                                    round(report["accuracy"], 6),
                                    round(report["macro_f1"], 6)])

        # 최고 Macro F1 시점만 저장합니다. 마지막 epoch 가 최고라는 보장은 없습니다.
        if report["macro_f1"] > best_f1:
            best_f1 = report["macro_f1"]
            best_report = {**report, "best_epoch": epoch}
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "macro_f1": best_f1,
                "args": vars(args),
            }, out_dir / "best_model.pt")

    # ---------- 결과 정리 ----------
    save_json(best_report, out_dir / "validation_report.json")
    save_json(vars(args), out_dir / "run_config.json")

    print("\n" + "=" * 60)
    print(f"최고 Public Validation Macro F1: {best_f1:.4f} (epoch {best_report['best_epoch']})")
    print("Class F1: " + "  ".join(
        f"{n} {v:.4f}" for n, v in zip(CLASS_NAMES, best_report["class_f1"])))
    print("\n혼동행렬 (행=실제, 열=예측)")
    print(format_confusion_matrix(best_report["confusion_matrix"]))
    print("=" * 60)
    print(f"저장 위치: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
