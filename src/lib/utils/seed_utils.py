from typing import Optional

import os
import random
import sys
import time
import logging

import numpy as np


log = logging.getLogger(__name__)


def _torch_rank_if_distributed() -> int:
    """Return torch.distributed rank if initialized, else 0."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    return 0


def set_seed(seed: Optional[int] = None, deterministic: bool = False) -> int:
    # generate a random seed
    if seed is None:
        try:
            seed = int.from_bytes(os.urandom(4), byteorder=sys.byteorder)
        except NotImplementedError:
            seed = int(time.time() * 1000)
        seed %= 2**31
    seed = int(seed)

    # distributed rank offset
    seed += _torch_rank_if_distributed()

    log.info("Seed: %d", seed)

    # numpy
    random.seed(seed)
    np.random.seed(seed)

    # torch
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass

            log.warning("PyTorch/cuDNN deterministic algorithms are enabled. This may affect performance")
    except ImportError:
        pass
    except Exception as e:
        log.warning("PyTorch seeding error: %s", e)

    return seed