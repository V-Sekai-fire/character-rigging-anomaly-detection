from anomalib.data.image.folder import Folder
from anomalib.models import EfficientAd
from anomalib.models.image.efficient_ad.lightning_model import EfficientAdModelSize
from anomalib.data.base.dataset import TaskType
from anomalib.data.utils import TestSplitMode, ValSplitMode
from anomalib.engine import Engine
from cog import Path, BaseModel, Input
import io
import shutil

class TrainingOutput(BaseModel):
    weights: Path
    dataset_root: Path
    pretrained: Path
    
def train(normal_dir: list[Path] = Input(description="A file containing training normal data"),) -> TrainingOutput:
    _normal_dir = Path("normal")
    _dataset_dir = Path("dataset")
    _dataset_normal_dir = _dataset_dir / _normal_dir
    _dataset_normal_dir.mkdir(parents=True, exist_ok=True)
    for dir_path in normal_dir:
        for file_path in dir_path.iterdir():
            if file_path.is_file():
                shutil.copy(file_path, str()) 
    datamodule = Folder(name="hazelnut_toy", normal_dir=str(_normal_dir), root=str(_dataset_dir), abnormal_dir=None, normal_test_dir=None, mask_dir=None, normal_split_ratio=0.2, extensions=None, train_batch_size=1, eval_batch_size=32, num_workers=8, task=TaskType.SEGMENTATION, image_size=None, transform=None, train_transform=None, eval_transform=None, test_split_mode=TestSplitMode.SYNTHETIC, test_split_ratio=0.2, val_split_mode=ValSplitMode.FROM_TEST, val_split_ratio=0.5, seed=None)
    datamodule.setup()
    model = EfficientAd(imagenet_dir=_dataset_dir, teacher_out_channels=384, model_size=EfficientAdModelSize.S, lr=0.0001, weight_decay=1e-05, padding=False, pad_maps=True)
    engine = Engine()
    engine.train(datamodule=datamodule, model=model)
    weights_file = "results/EfficientAd/dataset/latest/weights/lightning/model.ckpt"
    return TrainingOutput(weights=Path(weights_file), dataset_root=Path(normal_dir), pretrained=Path("pre_trained"))

# anomalib predict --return_predictions false --ckpt_path results/EfficientAd/avatar_rigging/latest/weights/lightning/model.ckpt --config results/EfficientAd/avatar_rigging/latest/config.yaml