import json
from typing import Dict, Union, List
from Model.GPModel import GPModel
from Config.TrainConfig import TrainConfig
from Utils.LoggerUtil import LoggerUtil


logger = LoggerUtil.get_logger()


class Evaluator():
    def __init__(self, conf: TrainConfig, data_path: str):
        self.conf = conf
        self.data = []
        with open(data_path, "r", encoding="utf8") as fr:
            for item in json.load(fr):
                self.data.append([item["query"], item["doc"], item["ans_type"], item["answer"]])

    def evaluate(self, model: GPModel) -> Dict:
        logger.info("evaluate model...")
        start = 0
        num_corr = 0
        while start < len(self.data):
            batch_data = self.data[start:start + 32]
            res = model.predict(querys=[item[0] for item in batch_data], docs=[item[1] for item in batch_data])
            for idx, item in enumerate(res):
                if item[0] == batch_data[idx][2]:
                    num_corr += 1
            start += 32
        return {"acc": num_corr / len(self.data)}

    def try_evaluate(self, model: GPModel, global_step: int) -> Union[Dict, None]:
        if self.conf.eval_step > 1 and global_step % self.conf.eval_step == 0:
            return self.evaluate(model)
        else:
            return None
