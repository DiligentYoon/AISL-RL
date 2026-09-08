from __future__ import annotations

import json
import os

import gymnasium
import torch

from collections import deque
from typing import Any, Optional, Union, Tuple, List, Deque, Dict

from lib.buffer.buffer import Buffer


class GeometricReplayBuffer(Buffer):
    pass