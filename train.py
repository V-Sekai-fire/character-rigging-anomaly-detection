from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode
from anomalib.models import EfficientAd
from anomalib.engine import Engine
from cog import Path
import io

def train(param: str) -> Path:
    datamodule = Folder(
        name="hazelnut_toy",
        root="datasets/hazelnut_toy",
        normal_dir="good",
        test_split_mode=TestSplitMode.SYNTHETIC,
    )
    datamodule.setup()
    model = EfficientAd()
    engine = Engine()
    engine.fit(datamodule=datamodule, model=model)
    return io.StringIO("hello " + param)