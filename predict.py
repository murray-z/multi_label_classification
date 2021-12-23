# -*- coding: utf-8 -*-

import torch
from data_preprocess import load_json
from bert_multilabel_cls import BertMultiLabelCls
from transformers import BertTokenizer

hidden_size = 768
class_num = 3
label2idx_path = "./data/label2idx.json"
save_model_path = "./model/multi_label_cls.pth"
label2idx = load_json(label2idx_path)
idx2label = {idx: label for label, idx in label2idx.items()}
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
max_len = 128

model = BertMultiLabelCls(hidden_size=hidden_size, class_num=class_num)
model.load_state_dict(torch.load(save_model_path))
model.to(device)
model.eval()


def predict(texts):
    outputs = tokenizer(texts, return_tensors="pt", max_len=max_len)
    logits = model(outputs["input_ids"], outputs["token_type_ids"], outputs["attention_mask"])
    logits = logits.cpu().tolist()
    result = []
    for sample in logits:
        pred_label = []
        for idx, logit in enumerate(sample):
            if logit > 0.5:
                pred_label.append(idx2label[logit])
        result.append(pred_label)
    return result


if __name__ == '__main__':
    texts = ["xxxx", "yyyy"]
    result = predict(texts)


