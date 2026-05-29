from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter
import os

old_logdir = "C:/Users/kuty1/AISL_RL/logs/g1_recovery/ppo/45"
new_logdir = "C:/Users/kuty1/AISL_RL/logs/g1_recovery/ppo/45/new"

ea = EventAccumulator(old_logdir)
ea.Reload()

writer = SummaryWriter(new_logdir)

for tag in ea.Tags()["scalars"]:
    events = ea.Scalars(tag)

    for e in events:
        step = e.step
        value = e.value

        # 예: 특정 tag만 보정
        if tag in ["Reward / Instantaneous reward (max)",
                   "Reward / Instantaneous reward (min)",
                   "Reward / Instantaneous reward (mean)",
                   "Reward / Total reward (max)",
                   "Reward / Total reward (min)",
                   "Reward / Total reward (mean)"
                   ]:
            value = value * 0.5  # 원하는 보정식으로 교체

        writer.add_scalar(tag, value, step)

writer.close()