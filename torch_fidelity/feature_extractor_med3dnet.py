# Portions of source code adapted from the following sources:
#   https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/inception.py
#   Distributed under Apache License 2.0: https://github.com/mseitzer/pytorch-fid/blob/master/LICENSE

import os
import sys
import zipfile
from contextlib import redirect_stdout

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from torch_fidelity.feature_extractor_base import FeatureExtractorBase
from torch_fidelity.helpers import vassert, text_to_dtype, get_kwarg, vprint
from torch_fidelity.interpolate_compat_tensorflow import interpolate_bilinear_2d_like_tensorflow1x


MEDICALNET_GDRIVE_ID = "13tnSvXY7oDIEloNFiGTsjUIYfS3g3BfG"
MEDICALNET_DEPTHS = ("10", "18", "34", "50", "101", "152", "200")


def _medicalnet_cache_dir():
    root = os.path.join(torch.hub._get_torch_home(), "checkpoints", "medicalnet")
    os.makedirs(root, exist_ok=True)
    return root


def _download_medicalnet_weights(depth, verbose=True):
    """
    Ensure the MedicalNet pretrained weights for the given ResNet depth exist locally.

    The MedicalNet release is a single archive containing all depths, so the first call
    downloads and extracts everything; subsequent calls hit the cache. Files land in
    ``$TORCH_HOME/checkpoints/medicalnet/resnet_<depth>.pth``.
    """
    cache_dir = _medicalnet_cache_dir()
    target = os.path.join(cache_dir, f"resnet_{depth}.pth")
    if os.path.exists(target):
        return target

    try:
        import gdown
    except ImportError as e:
        raise RuntimeError(
            "Automatic download of MedicalNet weights requires the 'gdown' package "
            "(Google Drive's confirmation flow can't be handled by torch.hub). "
            "Install it with: pip install gdown — or pass --feature-extractor-weights-path "
            "to point at a local resnet_<depth>.pth checkpoint."
        ) from e

    archive_path = os.path.join(cache_dir, "MedicalNet_pretrain.zip")
    vprint(verbose, f"Downloading MedicalNet pretrained weights from Google Drive into {cache_dir}")
    gdown.download(id=MEDICALNET_GDRIVE_ID, output=archive_path, quiet=not verbose)

    vprint(verbose, "Extracting MedicalNet archive")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(cache_dir)
    try:
        os.remove(archive_path)
    except OSError:
        pass

    # Hoist any resnet_*.pth nested inside subdirectories (e.g. pretrain/) up to cache_dir.
    for r, _, ff in os.walk(cache_dir):
        for f in ff:
            if f.startswith("resnet_") and f.endswith(".pth"):
                src = os.path.join(r, f)
                dst = os.path.join(cache_dir, f)
                if os.path.abspath(src) != os.path.abspath(dst):
                    os.replace(src, dst)

    if not os.path.exists(target):
        raise FileNotFoundError(
            f"resnet_{depth}.pth not found after extracting the MedicalNet archive into {cache_dir}. "
            f"Inspect the archive contents and set --feature-extractor-weights-path manually."
        )
    return target


class FeatureExtractorMed3dNetBase(FeatureExtractorBase):

    def __init__(
        self,
        name,
        features_list,
        block,
        layers,
        feature_extractor_weights_path=None,
        feature_extractor_internal_dtype=None,
        shortcut_type='B',
        **kwargs,
    ):
        """
        Med3dNet feature extractor for 3D Grayscale 24bit images.

        Args:

            name (str): Unique name of the feature extractor, must be the same as used in
                :func:`register_feature_extractor`.

            features_list (list): A list of the requested feature names, which will be produced for each input. This
                feature extractor provides the following features:

                - '256'
                - '512'
                - '1024'
                - '2048'

            feature_extractor_weights_path (str): Path to the pretrained Med3dNet model weights in PyTorch format.
                Refer to `util_convert_inception_weights` for making your own. Downloads from internet if `None`.

            feature_extractor_internal_dtype (str): dtype to use inside the feature extractor. Specifying it may improve
                numerical precision in some cases. Supported values are 'float32' (default), and 'float64'.
        """
        super(FeatureExtractorMed3dNetBase, self).__init__(name, features_list)
        vassert(
            feature_extractor_internal_dtype in ("float32", "float64", None),
            "Only 32 and 64 bit floats are supported for internal dtype of this feature extractor",
        )
        self.feature_extractor_internal_dtype = text_to_dtype(feature_extractor_internal_dtype, "float32")

        self.inplanes = 64
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4)

        self.conv_seg = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), Flatten())

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



        if feature_extractor_weights_path is None:
            depth = name.rsplit("-", 1)[-1]
            vassert(
                depth in MEDICALNET_DEPTHS,
                f"Cannot infer MedicalNet depth from extractor name '{name}'. "
                f"Expected one of {MEDICALNET_DEPTHS}, or pass feature_extractor_weights_path explicitly.",
            )
            feature_extractor_weights_path = _download_medicalnet_weights(
                depth, verbose=get_kwarg("verbose", kwargs)
            )

        state_dict = torch.load(feature_extractor_weights_path)["state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        self.load_state_dict(state_dict)

        self.to(self.feature_extractor_internal_dtype)
        self.requires_grad_(False)
        self.eval()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        vassert(torch.is_tensor(x) and x.dtype == torch.float64, f"Expecting image as torch.Tensor with dtype=torch.uint8 has {x.dtype}")
        vassert(x.dim() == 5 and x.shape[1] == 1, f"Input is not Bx1xDxHxW: {x.shape}")
        features = {}
        remaining_features = self.features_list.copy()

        x = x.to(self.feature_extractor_internal_dtype)
        # N x 1 x ? x ?

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        if "256" in remaining_features:
            features["256"] = self.conv_seg(x)
            remaining_features.remove("256")
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)
        x = self.layer2(x)
        if "512" in remaining_features:
            features["512"] = self.conv_seg(x)
            remaining_features.remove("512")
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)

        x = self.layer3(x)
        if "1024" in remaining_features:
            features["1024"] = self.conv_seg(x)
            remaining_features.remove("1024")
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)

        x = self.layer4(x)
        if "2048" in remaining_features:
            features["2048"] = self.conv_seg(x)
            remaining_features.remove("2048")
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)
        return tuple(features[a] for a in self.features_list)

    @staticmethod
    def get_provided_features_list():
        return "256", "512", "1024", "2048"

    @staticmethod
    def get_default_feature_layer_for_metric(metric):
        return {
            "isc": None,
            "fid": "2048",
            "kid": "2048",
            "prc": "2048",
        }[metric]

    @staticmethod
    def can_be_compiled():
        return True

    @staticmethod
    def get_dummy_input_for_compile():
        return (torch.rand([1, 3, 4, 4]) * 255).to(torch.uint8)


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Flatten(torch.nn.Module):
    def forward(self, inp):
        return inp.view(inp.size(0), -1)



class FeatureExtractorMed3dNet10(FeatureExtractorMed3dNetBase):
    """Constructs a ResNet-18 model.
    """
    def __init__(self, *args, **kwargs): 
        super(FeatureExtractorMed3dNet10, self).__init__(*args, BasicBlock, [1, 1, 1, 1], **kwargs)        

class FeatureExtractorMed3dNet18(FeatureExtractorMed3dNetBase):
    """Constructs a ResNet-18 model.
    """
    def __init__(self, *args, **kwargs): 
        super(FeatureExtractorMed3dNet18, self).__init__(*args, BasicBlock, [2, 2, 2, 2], **kwargs)  

class FeatureExtractorMed3dNet34(FeatureExtractorMed3dNetBase):
    """Constructs a ResNet-18 model.
    """
    def __init__(self, *args, **kwargs): 
        super(FeatureExtractorMed3dNet34, self).__init__(*args, BasicBlock, [3, 4, 6, 3], **kwargs)  


class FeatureExtractorMed3dNet50(FeatureExtractorMed3dNetBase):
    """Constructs a ResNet-18 model.
    """
    def __init__(self, *args, **kwargs): 
        super(FeatureExtractorMed3dNet50, self).__init__(*args, Bottleneck, [3, 4, 6, 3], **kwargs)        

class FeatureExtractorMed3dNet101(FeatureExtractorMed3dNetBase):
    """Constructs a ResNet-18 model.
    """
    def __init__(self, *args, **kwargs): 
        super(FeatureExtractorMed3dNet101, self).__init__(*args, Bottleneck, [3, 4, 23, 3], **kwargs)  

class FeatureExtractorMed3dNet152(FeatureExtractorMed3dNetBase):
    """Constructs a ResNet-18 model.
    """
    def __init__(self, *args, **kwargs): 
        super(FeatureExtractorMed3dNet152, self).__init__(*args, Bottleneck, [3, 8, 36, 3], **kwargs)  

class FeatureExtractorMed3dNet200(FeatureExtractorMed3dNetBase):
    """Constructs a ResNet-18 model.
    """
    def __init__(self, *args, **kwargs): 
        super(FeatureExtractorMed3dNet200, self).__init__(*args, Bottleneck, [3, 24, 36, 3], **kwargs)  

