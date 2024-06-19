from anomalib.data.image.folder import Folder
from anomalib.models import EfficientAd
from anomalib.models.image.efficient_ad.lightning_model import EfficientAdModelSize
from anomalib.data.base.dataset import TaskType
from anomalib.data.utils import TestSplitMode, ValSplitMode
from anomalib.engine import Engine
from cog import Path
import io

def train(param: str) -> Path:
    datamodule = Folder(name="hazelnut_toy", normal_dir="good", root="datasets/hazelnut_toy", abnormal_dir=None, normal_test_dir=None, mask_dir=None, normal_split_ratio=0.2, extensions=None, train_batch_size=1, eval_batch_size=32, num_workers=8, task=TaskType.SEGMENTATION, image_size=(1080, 1920), transform=None, train_transform=None, eval_transform=None, test_split_mode=TestSplitMode.SYNTHETIC, test_split_ratio=0.2, val_split_mode=ValSplitMode.FROM_TEST, val_split_ratio=0.5, seed=None)
    datamodule.setup()
    model = EfficientAd(imagenet_dir="datasets/hazelnut_toy", teacher_out_channels=384, model_size=EfficientAdModelSize.S, lr=0.0001, weight_decay=1e-05, padding=False, pad_maps=True)
    engine = Engine()
    engine.train(datamodule=datamodule, model=model)
    checkpoint_path = "results/EfficientAd/hazelnut_toy/latest/weights/lightning/model.ckpt"
    return checkpoint_path


if __name__ == "__main__":
    result = train("train")
    print(result)