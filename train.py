# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
import numpy as np
from data_preprocess import load_json
from bert_multilabel_cls import BertMultiLabelCls
from data_helper import MultiClsDataSet
from sklearn.metrics import accuracy_score


train_path = "./data/train.json"
dev_path = "./data/dev.json"
test_path = "./data/test.json"
label2idx_path = "./data/label2idx.json"
save_model_path = "./model/multi_label_cls.pth"
label2idx = load_json(label2idx_path)
class_num = len(label2idx)
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 2e-5
batch_size = 128
max_len = 128
hidden_size = 768
epochs = 10


train_dataset = MultiClsDataSet(train_path, max_len=max_len, label2idx_path=label2idx_path)
dev_dataset = MultiClsDataSet(dev_path, max_len=max_len, label2idx_path=label2idx_path)


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)


def get_acc_score(y_true_tensor, y_pred_tensor):
    y_pred_tensor = (y_pred_tensor.cpu() > 0.5).int().numpy()
    y_true_tensor = y_true_tensor.cpu().numpy()
    return accuracy_score(y_true_tensor, y_pred_tensor)


def train():
    model = BertMultiLabelCls(hidden_size=hidden_size, class_num=class_num)
    model.train()
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    dev_best_acc = 0.

    for epoch in range(1, epochs):
        model.train()
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            batch = [d.to(device) for d in batch]
            labels = batch[-1]
            logits = model(*batch[:3])
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                acc_score = get_acc_score(labels, logits)
                print("Train epoch:{} step:{}  acc: {} loss:{} ".format(epoch, i, acc_score, loss.item()))

        # 验证集合
        dev_loss, dev_acc = dev(model, dev_dataloader, criterion)
        print("Dev epoch:{} acc:{} loss:{}".format(epoch, dev_acc, dev_loss))
        if dev_acc > dev_best_acc:
            dev_best_acc = dev_acc
            torch.save(model.state_dict(), save_model_path)

    # 测试
    test_acc = test(save_model_path, test_path)
    print("Test acc: {}".format(test_acc))


def dev(model, dataloader, criterion):
    all_loss = []
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids, token_type_ids, attention_mask, labels = [d.to(device) for d in batch]
            logits = model(input_ids, token_type_ids, attention_mask)
            loss = criterion(logits, labels)
            all_loss.append(loss.item())
            true_labels.append(labels)
            pred_labels.append(logits)
    true_labels = torch.cat(true_labels, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    acc_score = get_acc_score(true_labels, pred_labels)
    return np.mean(all_loss), acc_score


def test(model_path, test_data_path):
    test_dataset = MultiClsDataSet(test_data_path, max_len=max_len, label2idx_path=label2idx_path)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = BertMultiLabelCls(hidden_size=hidden_size, class_num=class_num)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            input_ids, token_type_ids, attention_mask, labels = [d.to(device) for d in batch]
            logits = model(input_ids, token_type_ids, attention_mask)
            true_labels.append(labels)
            pred_labels.append(logits)
    true_labels = torch.cat(true_labels, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    acc_score = get_acc_score(true_labels, pred_labels)
    return acc_score


if __name__ == '__main__':
    train()
