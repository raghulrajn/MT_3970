import sys
from pathlib import Path
import yaml
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import torch
from src.models.model import Model_ContactAware as Model_ContactAware

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

cfg = load_yaml("/home/RUS_CIP/st189432/MT-3970/configs/model/model.yaml")
model_instance = Model_ContactAware(config = cfg)

ckpt = torch.load("/home/RUS_CIP/st189432/master-thesis-template-master/mlruns/1/0c909d46edf04e6dae13f16d2c56a0bd/artifacts/checkpoints/epoch=epoch=16-val_loss=val/loss=0.0000.ckpt", weights_only=False)
model_instance.load_state_dict(ckpt["state_dict"])
scripted = model_instance.to_torchscript(method="script")
scripted.save("model_eager.pt")
