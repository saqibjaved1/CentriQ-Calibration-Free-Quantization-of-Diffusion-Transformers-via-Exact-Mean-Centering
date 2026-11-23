# Vendored verbatim from sinq/utils.py @ commit 91dc9fb
# (https://github.com/huawei-csl/SINQ). Do not edit in place;
# changes go upstream then re-vendor.
'''
written by SINQ authors
'''

import math
import gc
import torch


def is_divisible(val1: int, val2: int) -> bool:
    return int(val2 * math.ceil(val1 / val2)) == val1

def cleanup():
    torch.cuda.empty_cache()
    gc.collect()
