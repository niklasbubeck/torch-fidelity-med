import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_fidelity.feature_extractor_base import FeatureExtractorBase
from torch_fidelity.helpers import vassert, text_to_dtype, get_kwarg


# Caffe-style ImageNet means in BGR order, on [0, 1] input scale.
# Matches MONAI generative's `subtract_mean` used by RadImageNetPerceptualSimilarity.
RADIMAGENET_BGR_MEAN = (0.406, 0.456, 0.485)

# Hub-entrypoint name. RadImageNet only ships a ResNet50 from Warvito/radimagenet-models.
RADIMAGENET_HUB_REPO = "Warvito/radimagenet-models"
RADIMAGENET_HUB_ENTRYPOINT = "radimagenet_resnet50"


class FeatureExtractorRadImageNet(FeatureExtractorBase):
    INPUT_IMAGE_SIZE = 224

    def __init__(
        self,
        name,
        features_list,
        feature_extractor_weights_path=None,
        feature_extractor_internal_dtype=None,
        **kwargs,
    ):
        """
        RadImageNet ResNet50 feature extractor for 2D radiology images.

        Loads the pretrained ResNet50 from the Warvito/radimagenet-models torch.hub repo
        (the same backbone used by MONAI generative's RadImageNetPerceptualSimilarity).
        Penultimate-layer feature maps are global-average-pooled to a 1D vector for use
        in FID, KID, PRC, ISC.

        Args:

            name (str): Unique name of the feature extractor, must match the registered
                name (e.g. ``radimagenet-resnet50``).

            features_list (list): Provided features; only ``feature_vec`` is supported.

            feature_extractor_weights_path (str): Path to a local pickled model object
                (e.g. ``torch.save(model, ...)`` of the hub model). When ``None`` the
                model is fetched via ``torch.hub`` from ``Warvito/radimagenet-models``.

            feature_extractor_internal_dtype (str): ``float32`` or ``float64``.
        """
        super().__init__(name, features_list)
        vassert(
            feature_extractor_internal_dtype in ("float32", "float64", None),
            "Only 32 and 64 bit floats are supported for internal dtype of this feature extractor",
        )
        self.feature_extractor_internal_dtype = text_to_dtype(feature_extractor_internal_dtype, "float32")

        verbose = get_kwarg("verbose", kwargs)

        if feature_extractor_weights_path is None:
            self.model = torch.hub.load(
                RADIMAGENET_HUB_REPO,
                model=RADIMAGENET_HUB_ENTRYPOINT,
                verbose=verbose,
            )
        else:
            self.model = torch.load(feature_extractor_weights_path)

        self.register_buffer(
            "bgr_mean",
            torch.tensor(RADIMAGENET_BGR_MEAN).view(1, 3, 1, 1),
        )

        self.to(self.feature_extractor_internal_dtype)
        self.requires_grad_(False)
        self.eval()

    def forward(self, x):
        vassert(
            torch.is_tensor(x) and x.dtype == torch.uint8,
            f"Expecting image as torch.Tensor with dtype=torch.uint8, got {x.dtype}",
        )
        vassert(x.dim() == 4 and x.shape[1] == 3, f"Input is not Bx3xHxW: {x.shape}")

        x = x.to(self.feature_extractor_internal_dtype) / 255.0
        # B x 3 x ? x ?  in [0, 1], RGB

        x = F.interpolate(
            x,
            size=(self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        # RGB -> BGR
        x = x[:, [2, 1, 0], ...]

        # Caffe-style mean subtraction (no std), matching RadImageNet's training preprocessing.
        x = x - self.bgr_mean.to(x.dtype)

        out = self.model(x)
        # Hub model returns spatial feature maps [B, C, H, W]; pool to a flat vector.
        if out.dim() == 4:
            out = F.adaptive_avg_pool2d(out, 1).flatten(1)

        out = out.to(torch.float32)
        return tuple(out for _ in self.features_list)

    @staticmethod
    def get_provided_features_list():
        return ("feature_vec",)

    @staticmethod
    def get_default_feature_layer_for_metric(metric):
        return {
            "isc": "feature_vec",
            "fid": "feature_vec",
            "kid": "feature_vec",
            "prc": "feature_vec",
        }[metric]

    @staticmethod
    def can_be_compiled():
        return True

    @staticmethod
    def get_dummy_input_for_compile():
        return (torch.rand([1, 3, 4, 4]) * 255).to(torch.uint8)
