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
    outputs = tokenizer(texts, return_tensors="pt", max_length=max_len,
                        padding=True, truncation=True)
    logits = model(outputs["input_ids"].to(device),
                   outputs["attention_mask"].to(device),
                   outputs["token_type_ids"].to(device))
    logits = logits.cpu().tolist()
    # print(logits)
    result = []
    for sample in logits:
        pred_label = []
        for idx, logit in enumerate(sample):
            if logit > 0.5:
                pred_label.append(idx2label[idx])
        result.append(pred_label)
    return result


if __name__ == '__main__':
    texts = ["中超-德尔加多扳平郭田雨绝杀 泰山2-1逆转亚泰", "今日沪深两市指数整体呈现震荡调整格局"]
    result = predict(texts)
    print(result)


