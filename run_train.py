import logging
import os

logging.basicConfig(level=logging.INFO)
from Config.TrainConfig import TrainConfig
from Trainer.Trainer import Trainer
from Utils.LoggerUtil import LoggerUtil

if __name__ == "__main__":
    conf = TrainConfig()
    # 指定配置文件即可
    conf.load("./train_configs/conf.yml")
    LoggerUtil.init_logger("YesNoEvidence", os.path.join(conf.output_dir, "logs.txt"))
    trainer = Trainer(conf)
    trainer.train()
