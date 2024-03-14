import os
# import wandb
import pytz
from datetime import datetime
import pandas as pd


class WandBLogger:
    """WandB logger."""

    def __init__(self, args, system_prompt):
        pass

    def finish(self):
        self.print_final_summary_stats()
        self.logger.finish()

    def print_summary_stats(self, iter):
        pass

    def print_final_summary_stats(self):
        pass

