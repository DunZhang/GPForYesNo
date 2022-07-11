import numpy as np
import torch
import torch.nn as nn
from Config.TrainConfig import TrainConfig
from typing import Dict
from typing import List, Union
from os.path import join
from transformers import BertModel, BertConfig, BertTokenizer
from Utils.LoggerUtil import LoggerUtil

logger = LoggerUtil.get_logger()


class SinusoidalPositionEmbedding(nn.Module):
    """定义Sin-Cos位置Embedding
    """

    def __init__(
            self, output_dim, merge_mode='add', custom_position_ids=False):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def forward(self, inputs):
        if self.custom_position_ids:
            seq_len = inputs.shape[1]
            inputs, position_ids = inputs
            position_ids = position_ids.type(torch.float)
        else:
            input_shape = inputs.shape
            batch_size, seq_len = input_shape[0], input_shape[1]
            position_ids = torch.arange(seq_len).type(torch.float)[None]
        indices = torch.arange(self.output_dim // 2).type(torch.float)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (-1, seq_len, self.output_dim))
        if self.merge_mode == 'add':
            return inputs + embeddings.to(inputs.device)
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0).to(inputs.device)
        elif self.merge_mode == 'zero':
            return embeddings.to(inputs.device)


class GPModel(nn.Module):
    def __init__(self, conf_or_model_dir: Union[str, TrainConfig]):
        """
        如果conf_or_model_dir为配置类则代表从预训练模型开始加载，用于训练
        如果为conf_or_model_dir为目录路径，则从该路径进行加载，该路径下必须有之前存好的模型及相关配置文件，用于继续训练或预测
        """
        super().__init__()
        self.device = None
        # 加载config
        if isinstance(conf_or_model_dir, TrainConfig):
            self.conf = conf_or_model_dir
        else:
            self.conf = TrainConfig()
            self.conf.load(conf_path=join(conf_or_model_dir, "model_conf.yml"))
        # 确定模型目录
        self.pretrained_model_dir = self.conf.pretrained_model_dir
        self.max_len = self.conf.max_len
        CONFIG, TOKENIZER, MODEL = BertConfig, BertTokenizer, BertModel
        self.ent_type_size = self.conf.ent_type_size
        self.use_rope = self.conf.use_rope

        # 加载权重
        if isinstance(conf_or_model_dir, TrainConfig):
            # 加载预训练
            self.model = MODEL.from_pretrained(self.pretrained_model_dir)
            self.tokenizer = TOKENIZER.from_pretrained(self.pretrained_model_dir)
            self.backbone_model_config = CONFIG.from_pretrained(self.pretrained_model_dir)
            self.dense_1 = nn.Linear(
                self.backbone_model_config.hidden_size, self.backbone_model_config.hidden_size * 2)
            self.dense_2 = nn.Linear(
                self.backbone_model_config.hidden_size, self.ent_type_size * 2)
        else:
            # 加载训练好的
            self.tokenizer = TOKENIZER.from_pretrained(conf_or_model_dir)
            self.backbone_model_config = CONFIG.from_pretrained(conf_or_model_dir)
            self.model = MODEL(config=self.backbone_model_config)
            self.dense_1 = nn.Linear(
                self.backbone_model_config.hidden_size, self.backbone_model_config.hidden_size * 2)
            self.dense_2 = nn.Linear(
                self.backbone_model_config.hidden_size, self.ent_type_size * 2)
            self.load_state_dict(torch.load(join(conf_or_model_dir, "model_weight.bin"), map_location="cpu"))

    def sequence_masking(self, x, mask, value='-inf', axis=None):
        if mask is None:
            return x
        else:
            if value == '-inf':
                value = -1e12
            elif value == 'inf':
                value = 1e12
            assert axis > 0, 'axis must be greater than 0'
            for _ in range(axis - 1):
                mask = torch.unsqueeze(mask, 1)
            for _ in range(x.ndim - mask.ndim):
                mask = torch.unsqueeze(mask, mask.ndim)
            return x * mask + value * (1 - mask)

    def add_mask_tril(self, logits, mask):
        if mask.dtype != logits.dtype:
            mask = mask.type(logits.dtype)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 2)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 1)
        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), diagonal=-1)
        logits = logits - mask * 1e12
        return logits

    def forward(self, ipt: Dict):
        self.device = self.get_device()
        # 变量进显存
        input_ids = ipt["input_ids"].to(self.get_device())
        token_type_ids = ipt["token_type_ids"].to(self.get_device())
        attention_mask = ipt["attention_mask"].to(self.get_device())

        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        last_hidden_state = context_outputs.last_hidden_state
        outputs = self.dense_1(last_hidden_state)
        # qw,kw  (batch_size, seq_len, inner_dim)
        qw, kw = outputs[..., ::2], outputs[..., 1::2]  # 从0,1开始间隔为2
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.inner_dim, 'zero')(outputs)
            cos_pos = pos[..., 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos[..., ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 3)
            qw2 = torch.reshape(qw2, qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 3)
            kw2 = torch.reshape(kw2, kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        # logits (batch_size, seq_len, seq_len)
        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.inner_dim ** 0.5
        # bias (batch_size, ent_type*2, seq_len)
        bias = torch.einsum('bnh->bhn', self.dense_2(last_hidden_state)) / 2
        # print("bias", bias.shape)
        # print(logits.shape)
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]  # logits[:, None] 增加一个维度
        # print(logits.shape) (bsz, ent_type, seq_len, seq_len)
        logits = self.add_mask_tril(logits, mask=token_type_ids)  # 注意要用token_type，query部分是不需要的
        return {"logits": logits}

    def predict(self, querys: List[str], docs: List[str]):
        """

        :param querys:
        :param docs:
        :return: List [ [yes/no, evidence] ]
        """
        ipt = self.tokenizer.batch_encode_plus(
            list(zip(querys, docs)), max_length=512, truncation=True, padding=True, return_tensors="pt")
        self.eval()
        with torch.no_grad():
            logits = self(ipt)["logits"].cpu().numpy()
        res = []
        for idx in range(len(querys)):
            no_logits = logits[idx, 0, :, :]
            yes_logits = logits[idx, 1, :, :]
            if np.max(no_logits) > np.max(yes_logits):
                answer_type = "no"
                target_logits = no_logits
            else:
                answer_type = "yes"
                target_logits = yes_logits
            idx = int(np.argmax(target_logits))
            x, y = idx // target_logits.shape[1], idx % target_logits.shape[1]
            answer = ipt["input_ids"][idx].cpu().numpy().tolist()[x:y + 1]
            answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(answer))
            res.append([answer_type, answer])
        self.train()
        return res

    def save(self, save_dir):
        self.conf.save(join(save_dir, "model_conf.yml"))
        torch.save(self.state_dict(), join(save_dir, "model_weight.bin"))
        self.backbone_model_config.save_pretrained(save_dir)
        self.tokenizer.save_vocabulary(save_dir)

    def get_device(self):
        if self.device is None:
            for v in self.state_dict().values():
                self.device = v.device
                break
        return self.device
