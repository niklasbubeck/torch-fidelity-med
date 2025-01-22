# Functions fid_features_to_statistics and fid_statistics_to_metric are adapted from
#   https://github.com/bioinf-jku/TTUR/blob/master/fid.py commit id d4baae8
#   Distributed under Apache License 2.0: https://github.com/bioinf-jku/TTUR/blob/master/LICENSE

import numpy as np
from torch.utils.data import DataLoader
from generative.metrics import MultiScaleSSIMMetric
from tqdm import tqdm

from torch_fidelity.helpers import get_kwarg, vprint
from torch_fidelity.utils import (
    prepare_input_from_id,
)

KEY_METRIC_MS_SSIM = "multi_scale_structural_similarity"

ms_ssim = MultiScaleSSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=7)

def calculate_ms_ssim(input_id1, input_id2, **kwargs):
    input_desc1 = prepare_input_from_id(input_id1, **kwargs)
    input_desc2 = prepare_input_from_id(input_id2, **kwargs)
    bs = get_kwarg("batch_size", kwargs)

    dl1 = DataLoader(input_desc1, batch_size=bs)
    dl2 = DataLoader(input_desc2, batch_size=bs)

    cuda = get_kwarg("cuda", kwargs)
    ms_ssim_list = []
    for batch1 in tqdm(dl1):
        for batch2 in tqdm(dl2):
            if cuda:
                batch1 = batch1.cuda()
                batch2 = batch2.cuda()
            print(batch1.shape, batch2.shape, batch1.device, batch2.device)
            ms_ssim_list.append(ms_ssim(batch1, batch2).mean().item())
    ms_ssim_list = np.array(ms_ssim_list)
    out = {KEY_METRIC_MS_SSIM: ms_ssim_list.mean()}
    return out
