import torch.nn as nn
import torchvision.models as tvmodels
from torch_fidelity.helpers import vassert

from torch_fidelity.feature_extractor_base import FeatureExtractorBase


class FeatureExtractorResNetBase(FeatureExtractorBase):
    raise NotImplementedError("ResNet is not yet implemented")