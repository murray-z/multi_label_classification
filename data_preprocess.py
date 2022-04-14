# -*- coding: utf-8 -*-

"""
数据预处理
"""

import json


def load_json(data_path):
    with open(data_path, encoding="utf-8") as f:
        return json.loads(f.read())


def dump_json(project, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(project, f, ensure_ascii=False)


def preprocess(train_data_path, label2idx_path, max_len_ratio=0.9):
    """
    :param train_data_path:
    :param label2idx_path:
    :param max_len_ratio:
    :return:
    """
    labels = []
    text_length = []
    with open(train_data_path, encoding="utf-8") as f:
        for data in f:
            data = json.loads(data)
            text_length.append(len(data["text"]))
            labels.extend(data["label"])
    labels = list(set(labels))
    label2idx = {label: idx for idx, label in enumerate(labels)}

    dump_json(label2idx, label2idx_path)

    text_length.sort()

    print("当设置max_len={}时，可覆盖{}的文本".format(text_length[int(len(text_length)*max_len_ratio)], max_len_ratio))


if __name__ == '__main__':
    preprocess("./data/train.json", "./data/label2idx.json")
