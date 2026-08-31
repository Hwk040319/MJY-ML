"""학습한 체크포인트로 열화상 이미지 한 장을 예측합니다.

사용법:
    python predict_one.py --image data/public_val/images/example.png \
        --checkpoint outputs/baseline/best_model.pt
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from common import CLASS_NAMES, get_transforms, resolve_device
from train_baseline import build_model


def parse_args():
    ap = argparse.ArgumentParser(description="배터리 열화상 이미지 한 장 예측")
    ap.add_argument("--image", required=True, help="예측할 PNG 이미지 경로")
    ap.add_argument("--checkpoint", required=True, help="best_model.pt 경로")
    return ap.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    image_path = Path(args.image)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_args = checkpoint.get("args", {})
    device = resolve_device()

    model = build_model(
        bool(saved_args.get("unfreeze", False)), device, pretrained=False
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    image_size = int(saved_args.get("image_size", 224))
    transform = get_transforms(image_size, train=False)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    predicted = int(probabilities.argmax().item())

    result = {
        "image": str(image_path),
        "predicted_class": predicted,
        "predicted_label": CLASS_NAMES[predicted],
        "probabilities": {
            name: round(float(probabilities[i]), 6)
            for i, name in enumerate(CLASS_NAMES)
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
