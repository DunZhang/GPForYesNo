import json
import random
import torch
from torch.utils.data import Dataset
from Utils.LoggerUtil import LoggerUtil
from transformers import PreTrainedTokenizer
from Config.TrainConfig import TrainConfig
from typing import List, Dict

logger = LoggerUtil.get_logger()


# TODO 1、如何产生额外的负例 2、doc过长产生的负例应该如何解决 3、
def collect_fn(batch):
    """

    :param batch:List[data_set[i]]
    :return:
    """
    input_ids = list(map(lambda x: x[0] + x[1], batch))
    max_len = max(map(lambda x: len(x), input_ids))
    label = torch.zeros((len(batch), 2, max_len, max_len))
    for idx, item in enumerate(batch):
        if item[4] == "yes" or not item[4]:
            label[idx, 1, item[2], item[3]] = 1
        if item[4] == "no" or not item[4]:
            label[idx, 0, item[2], item[3]] = 1

    attention_mask = [[1] * len(item) + [0] * (max_len - len(item)) for item in input_ids]
    input_ids = [item + [0] * (max_len - len(item)) for item in input_ids]
    token_type_ids = [[0] * len(item[0]) + [1] * len(item[1]) for item in batch]
    token_type_ids = [item + [0] * (max_len - len(item)) for item in token_type_ids]
    ipt = {
        "input_ids": torch.LongTensor(input_ids),
        "attention_mask": torch.LongTensor(attention_mask),
        "token_type_ids": torch.LongTensor(token_type_ids),
        "lable": label
    }
    return ipt


class GPDataSet(Dataset):
    def __init__(
            self, conf: TrainConfig, data_path: str, tokenizer: PreTrainedTokenizer, *args, **kwargs):
        """

        :param conf:
        :param data_path:
        :param tokenizer:
        :param docid2doc:
        :param args:
        :param kwargs:
        """
        self.conf = conf
        # 参数初始化
        self.tokenizer = tokenizer
        self.data_path = data_path
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self.init_data_model()

    def init_data_model(self):
        """ 初始化要用到的模型数据 """
        with open(self.data_path, "r", encoding="utf8") as fr:
            self.data = json.load(fr)
        random.shuffle(self.data)
        logger.info("总数据{}条".format(len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        """
        item 为数据索引，迭代取第item条数据
        """
        # 获取目标数据
        data_item = self.data[item]
        query, doc, answer, answer_type = data_item["query"], data_item["doc"], data_item["ans"], data_item["ans_type"]
        doc = doc[:self.conf.max_len - len(query) - 2]
        query_id = self.tokenizer.encode(query, add_special_tokens=True)
        if answer and answer in doc:
            idx = doc.find(answer)
            doc_left, doc_right = doc[:idx].strip(), doc[idx + len(answer):].strip()
            doc_left_id, doc_right_id = [], []
            if doc_left:
                doc_left_id = self.tokenizer.encode(doc_left, add_special_tokens=False)
            if doc_right:
                doc_right_id = self.tokenizer.encode(doc_right, add_special_tokens=False)
            answer_id = self.tokenizer.encode(answer, add_special_tokens=False)
            doc_id = doc_left_id + answer_id + doc_right_id + [self.tokenizer.sep_token_id]
            start, end = len(doc_left_id), len(doc_left_id) + len(answer_id) - 1

        else:
            doc_id = self.tokenizer.encode(doc, add_special_tokens=False) + [self.tokenizer.sep_token_id]
            start, end = len(doc_id) - 1, len(doc_id) - 1
        start, end = start + len(query_id), end + len(query_id)
        return query_id, doc_id, start, end, answer_type
