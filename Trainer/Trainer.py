import json
from typing import Tuple, Dict
from torch.optim import Optimizer
from torch.utils.data import DataLoader, RandomSampler

from Config.TrainConfig import TrainConfig
from InfoLogger.InfoLogger import InfoLogger
from LossModel.LossModel import LossModel
from ModelSaver.GeneralModelSaver import GeneralModelSaver
import numpy as np
import random
from DataLoader.GPDataSet import collect_fn, GPDataSet
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import os
from Model.GPModel import GPModel
from Utils.LoggerUtil import LoggerUtil
from os.path import join
from Evaluator.Evaluator import Evaluator

logger = LoggerUtil.get_logger()


class Trainer():
    def __init__(self, conf: TrainConfig):
        self.conf = conf
        self.seed_everything(seed=conf.seed)
        # data loader
        self.train_data_loader = self.get_data_loader()
        # model
        self.model = self.get_model()
        # evaluator
        self.evaluator = self.get_evaluator()
        # loss model
        self.loss_model = self.get_loss_model()
        # model saver
        self.model_saver = self.get_model_saver()
        # device
        self.device = self.get_device()
        # optimizer
        self.optimizer = self.get_optimizer(model=self.model)

        # step info
        self.num_epoch, self.epoch_steps = self.get_step_info()

        #  scheuler
        self.lr_scheduler = self.get_lr_scheduler(optimizer=self.optimizer)

        # info logger
        self.info_logger = self.get_info_logger()

    def seed_everything(self, seed: int = 1029):
        '''
        设置整个开发环境的seed
        :param seed:
        :param device:
        :return:
        '''
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    def get_data_loader(self) -> DataLoader:
        self.tokenizer = BertTokenizer.from_pretrained(self.conf.pretrained_model_dir)
        # train data
        train_dataset = GPDataSet(
            conf=self.conf, data_path=join(self.conf.data_dir, "train.json"), tokenizer=self.tokenizer)
        sampler = RandomSampler(data_source=train_dataset)
        train_data_loader = DataLoader(dataset=train_dataset, batch_size=self.conf.batch_size, sampler=sampler,
                                       num_workers=2, collate_fn=collect_fn, pin_memory=True, drop_last=True)

        return train_data_loader

    def get_model(self):
        model = GPModel(conf_or_model_dir=self.conf)
        return model

    def get_evaluator(self) -> Evaluator:
        return Evaluator(conf=self.conf, data_path=join(self.conf.data_dir, "dev.json"))

    def get_loss_model(self) -> LossModel:
        return LossModel(conf=self.conf)

    def get_model_saver(self) -> GeneralModelSaver:
        return GeneralModelSaver(conf=self.conf)

    def get_optimizer(self, model: GPModel) -> Optimizer:
        """
        因为是BERT，就默认用adam了
        :param model:
        :return:
        """
        no_decay = ["bias", "LayerNorm.weight"]
        paras = dict(model.named_parameters())
        optimizer_grouped_parameters = [{
            "params": [p for n, p in paras.items() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
            {"params": [p for n, p in paras.items() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.conf.lr)
        return optimizer

    def get_step_info(self) -> Tuple[int, int]:
        return self.conf.num_epoch, len(self.train_data_loader.dataset) // self.conf.batch_size

    def get_lr_scheduler(self, optimizer: Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        total_steps = self.num_epoch * self.epoch_steps
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(self.conf.warmup_proportion * total_steps),
                                                    num_training_steps=total_steps)
        return scheduler

    def get_info_logger(self) -> InfoLogger:
        return InfoLogger(conf=self.conf)

    def get_device(self) -> torch.device:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.conf.device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device

    def train(self):
        self.model = self.model.to(self.device)
        self.loss_model = self.loss_model.to(self.device)
        # train
        global_step = 1
        logger.info("start train")
        for epoch in range(self.num_epoch):
            for step, ipt in enumerate(self.train_data_loader):
                global_step += 1
                model_output = self.model(ipt)
                loss = self.loss_model(ipt, model_output)
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                # 梯度下降，更新参数
                self.optimizer.step()
                self.lr_scheduler.step()
                # 把梯度置0
                self.model.zero_grad()
                self.optimizer.zero_grad()

                # 如果符合条件则会进行模型评估
                eval_result = self.evaluator.try_evaluate(model=self.model, global_step=global_step, )
                # 如果符合条件则会保存模型
                self.model_saver.try_save_model(model=self.model, global_step=global_step,
                                                epoch_steps=self.epoch_steps, eval_result=eval_result)
                # 如果符合条件则会输出相关信息
                self.info_logger.try_print_log(
                    loss=loss, eval_result=eval_result, step=step, global_step=global_step,
                    epoch_steps=self.epoch_steps, num_epochs=self.num_epoch, epoch=epoch + 1, ipt=ipt,
                    tokenizer=self.tokenizer)
