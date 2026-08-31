"""배터리 열화상 이미지 Dataset.

강의에서 함께 읽는 파일입니다. 핵심은 __getitem__ 하나입니다.
'이미지 파일 한 장'이 '모델이 계산할 수 있는 숫자 텐서'로 바뀌는 지점입니다.
"""

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class BatteryDataset(Dataset):
    """data_root/split 폴더 하나를 읽는 Dataset.

    폴더 구조
        data/
          train/       images/*.png + labels.csv
          public_val/  images/*.png + labels.csv
          private_test/images/*.png          <- labels.csv 없음 (정답 비공개)

    labels.csv 열
        image_name, target, original_stage, experiment_id
    """

    def __init__(self, data_root, split, transform=None, require_labels=True):
        self.split_dir = Path(data_root) / split
        self.image_dir = self.split_dir / "images"
        self.transform = transform

        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"이미지 폴더를 찾을 수 없습니다: {self.image_dir}\n"
                f"--data-root 값이 맞는지 확인하세요."
            )

        label_path = self.split_dir / "labels.csv"

        if label_path.is_file():
            df = pd.read_csv(label_path)
            self.image_names = df["image_name"].tolist()
            self.targets = df["target"].astype(int).tolist()
            self.experiment_ids = (
                df["experiment_id"].tolist() if "experiment_id" in df.columns else None
            )
        else:
            if require_labels:
                raise FileNotFoundError(f"labels.csv 가 없습니다: {label_path}")
            # private_test: 정답이 없으므로 target 자리에 -1 을 채웁니다.
            self.image_names = sorted(p.name for p in self.image_dir.glob("*.png"))
            self.targets = [-1] * len(self.image_names)
            self.experiment_ids = None

        if len(self.image_names) == 0:
            raise RuntimeError(f"이미지가 0장입니다: {self.image_dir}")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # (1) 파일명을 실제 경로로 바꿔 이미지를 엽니다.
        name = self.image_names[idx]
        image = Image.open(self.image_dir / name).convert("RGB")

        # (2) transform 이 resize -> ToTensor -> normalize 를 수행합니다.
        #     이 시점에서 이미지는 [3, H, W] 실수 텐서가 됩니다.
        if self.transform is not None:
            image = self.transform(image)

        # (3) 모델 입력, 정답, 파일명을 함께 돌려줍니다.
        #     파일명은 예측 결과를 확인할 때 어떤 이미지인지 표시하는 데 씁니다.
        return image, self.targets[idx], name

    def class_counts(self, num_classes=3):
        """클래스별 이미지 수. class weight 계산과 불균형 확인에 씁니다."""
        counts = [0] * num_classes
        for t in self.targets:
            if 0 <= t < num_classes:
                counts[t] += 1
        return counts
