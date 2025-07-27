# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import logging
import os
import re
from collections import OrderedDict

import torch
import yaml


def load_checkpoint(model: torch.nn.Module, model_pth: str) -> dict:
    checkpoint = torch.load(model_pth, map_location='cpu')
    checkpoint = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    # Handle position encoding size mismatch
    model_state = model.state_dict()
    for key in list(checkpoint.keys()):
        if key in model_state:
            if checkpoint[key].shape != model_state[key].shape:
                if 'pos_enc.pe' in key:
                    # Skip position encoding parameters with size mismatch
                    print(f"Skipping {key} due to size mismatch: {checkpoint[key].shape} != {model_state[key].shape}")
                    del checkpoint[key]
                    continue
    
    model.load_state_dict(checkpoint, strict=False)
    info_path = re.sub('.pth$', '.yaml', model_pth)
    configs = {}
    if os.path.exists(info_path):
        with open(info_path, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
    return configs
