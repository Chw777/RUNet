from torch.utils.data import DataLoader, TensorDataset
import json
from optimizer import customAdam
from Model import BaseModel
from NTXentLoss import NTXentLoss
from load_data import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


results = {}

import numpy as np
import torch


class Pretrainer:
    def __init__(self, subject_id, configs):
        self.subject_id = subject_id
        self.configs = configs
        self.device = device

        self.model = BaseModel(configs).to(device)

        # 冻结Sconv3之后的层（attention_module、log_layer1、vec），但projection要训练
        for param in self.model.PCOM.parameters():
            param.requires_grad = False
        for param in self.model.log_layer1.parameters():
            param.requires_grad = False
        for param in self.model.vec.parameters():
            param.requires_grad = False

        self.optimizer = customAdam([
            {'params': self.model.block1.parameters(), 'lr': 1e-4},
            {'params': self.model.block2.parameters(), 'lr': 1e-4},
            {'params': self.model.block3.parameters(), 'lr': 1e-4},
            {'params': self.model.Sconv3.parameters(), 'lr': 1e-4},
            {'params': self.model.projection.parameters(), 'lr': 1e-4},
        ])

        self.criterion = NTXentLoss(temperature=0.5)
        self._load_data()

    def _load_data(self):
        rest_data, rest_labels = load_contrast_data(self.subject_id)

        self.loader = DataLoader(
            TensorDataset(rest_data, rest_labels),
            batch_size=64, shuffle=True, drop_last=True
        )

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        for inputs, labels in self.loader:
            inputs = inputs.to(self.device)
            # B = inputs.size(0)

            x_i = inputs[::2]
            x_j = inputs[1::2]

            z_i = self.model(x_i, return_projection=True)
            z_j = self.model(x_j, return_projection=True)

            loss = self.criterion(z_i, z_j)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.loader)

    def save_checkpoint(self):
        checkpoint = {
            'block1': self.model.block1.state_dict(),
            'block2': self.model.block2.state_dict(),
            'block3': self.model.block3.state_dict(),
            'Sconv3': self.model.Sconv3.state_dict(),
            'attention': self.model.PCOM.state_dict(),
            'log_layer1': self.model.log_layer1.state_dict(),
            'vec': self.model.vec.state_dict(),
            'projection': self.model.projection.state_dict(),
            'FC': self.model.FC.state_dict(),
        }
        torch.save(checkpoint, f'pretrain_A{self.subject_id:02d}.pth')

    def run(self, epochs=20):
        print(f"=== Pretraining Subject A{self.subject_id:02d} ===")
        for epoch in range(epochs):
            loss = self.train_epoch()
            print(f"Pretrain Epoch {epoch + 1}/{epochs} | Loss: {loss:.4f}")
        self.save_checkpoint()


class Finetuner:
    """第二阶段：仅分类训练（移除域适应相关代码）"""
    def __init__(self, subject_id, configs, pretrain_path):
        self.subject_id = subject_id
        self.configs = configs
        self.device = device

        # 初始化模型并加载预训练参数
        self.model = self._init_model(pretrain_path)
        self.optimizer = customAdam([
            {'params': self.model.block1.parameters(), 'lr': 1e-3},
            {'params': self.model.block2.parameters(), 'lr': 1e-3},
            {'params': self.model.block3.parameters(), 'lr': 1e-3},
            {'params': self.model.Sconv3.parameters(), 'lr': 1e-3},
            {'params': self.model.PCOM.parameters(), 'lr': 1e-3},
            {'params': self.model.vec.parameters(), 'lr': 1e-3},
            {'params': self.model.FC.parameters(), 'lr': 1e-3}
        ])

        self.ce_criterion = nn.CrossEntropyLoss()
        self._load_data()

    # 新增：加载预训练模型参数的方法
    def _init_model(self, pretrain_path):
        model = BaseModel(self.configs).to(device)
        checkpoint = torch.load(pretrain_path)

        # 加载预训练参数（不冻结任何层）
        model.block1.load_state_dict(checkpoint['block1'])
        model.block2.load_state_dict(checkpoint['block2'])
        model.block3.load_state_dict(checkpoint['block3'])
        model.Sconv3.load_state_dict(checkpoint['Sconv3'])
        model.PCOM.load_state_dict(checkpoint['attention'])
        model.vec.load_state_dict(checkpoint['vec'])
        model.projection.load_state_dict(checkpoint['projection'])
        model.FC.load_state_dict(checkpoint['FC'])  # 新增：加载分类层参数

        return model  # 所有参数默认requires_grad=True
    def _load_data(self):
        # 仅加载任务数据（移除目标静息态数据加载）
        (source_train, source_labels), (self.X_test, self.y_test) = load_task_data(self.subject_id)
        self.source_loader = DataLoader(
            TensorDataset(source_train, source_labels),
            batch_size=32, shuffle=True, drop_last=True
        )
        self.test_loader = DataLoader(
            TensorDataset(self.X_test, self.y_test),
            batch_size=32
        )

    def _compute_loss(self, src_inputs, src_labels):  # 移除tgt_inputs参数
        cls_output, _ = self.model(src_inputs)
        return self.ce_criterion(cls_output, src_labels)

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for inputs, labels in self.source_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            loss = self._compute_loss(inputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss / len(self.source_loader)

    # 评估函数保持不变
    def evaluate(self):
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs, _ = self.model(inputs)
                correct += (outputs.argmax(1) == labels).sum().item()
        return 100 * correct / len(self.y_test)

    def run(self, epochs=30):
        best_acc = 0
        print(f"=== Finetuning Subject A{self.subject_id:02d} ===")
        for epoch in range(epochs):
            loss = self.train_epoch()
            acc = self.evaluate()
            if acc > best_acc:
                best_acc = acc
                torch.save(self.model.state_dict(), f'finetune_A{self.subject_id:02d}.pth')

            print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss:.4f} "
                  f"| Acc: {acc:.2f}%")
        return best_acc


if __name__ == "__main__":
    #BCIIV2A
    configs = {
        'class_num': 4,
        'channelNum': 22,
        'width': 300,
        'drop_att': 0.2,
        'p': 3
    }
    # BCIIV2B
    # configs = {
    #     'class_num': 2,
    #     'channelNum': 3,
    #     'width': 300,
    #     'drop_att': 0.2,
    #     'p': 3
    # }
    #PhysioNet
    # configs = {
    #     'class_num': 2,
    #     'channelNum': 64,
    #     'width': 300,
    #     'drop_att': 0.2,
    #     'p': 3
    # }
    for subject_id in range(1, 10):
        print(f"\n=== Processing Subject A{subject_id:02d} ===")

        # 第一阶段：对比学习预训练（使用新的数据路径）
        pretrainer = Pretrainer(subject_id, configs)
        pretrainer.run(epochs=1)

        # 第二阶段：仅分类训练（移除域适应）
        finetuner = Finetuner(subject_id, configs, f'pretrain_A{subject_id:02d}.pth')
        best_acc = finetuner.run(epochs=1)

        results[f'A{subject_id:02d}'] = best_acc
        print(f"Subject A{subject_id:02d} Best Acc: {best_acc:.2f}%")

    with open('two_stage_results.json', 'w') as f:
        json.dump(results, f, indent=4)



