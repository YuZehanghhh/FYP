
import argparse
from ast import arg
import os
import csv
import torch
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from torch.utils.data import Dataset
import sys
from models import get_model
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE





# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
])

# 加载数据集
dataset = datasets.ImageFolder(
    root='prototypical3',
    transform=preprocess
)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


model = get_model('CLIP:ViT-L/14')
model.fc = nn.Identity()
# for name, param in model.named_parameters():
#     if param.dim() > 0:  # 检查参数是否为多维张量
#         print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]}")  # 只显示每个张量的前两个元素
#     else:
#         print(f"Layer: {name} | Size: {param.size()} | Value : {param.item()}") 
# print(model)

def extract_features(dataloader, model):
    features = []
    labels = []
    with torch.no_grad():
        for imgs, lbls in dataloader:
            feats = model(imgs)
            features.append(feats)
            labels.append(lbls)
    return torch.cat(features), torch.cat(labels)

features, labels = extract_features(dataloader, model)
original_length = features.shape[0]
print("ol=", original_length)

def compute_prototype(features, labels, label):
    return features[labels == label].mean(0)

prototype_real = compute_prototype(features, labels, 0)
prototype_fake = compute_prototype(features, labels, 1)

# distance_real = torch.norm(features[labels == 0] - prototype_real, dim=1)
# distance_fake = torch.norm(features[labels == 1] - prototype_fake, dim=1)

# print("Distance from real prototype:", distance_real.mean().item())
# print("Distance from fake prototype:", distance_fake.mean().item())



extended_features = np.vstack([features, prototype_real, prototype_fake])

# 执行 t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(extended_features)

# 提取 t-SNE 结果的最后两个点为原型
real_proto_tsne = tsne_results[-2]
fake_proto_tsne = tsne_results[-1]

# 分离原始 t-SNE 结果和原型
normal_tsne_results = tsne_results[:-2]

real_indices = labels == 0
fake_indices = labels == 1

# 真实图像的t-SNE结果
real_tsne = normal_tsne_results[real_indices]
# 假图像的t-SNE结果
fake_tsne = normal_tsne_results[fake_indices]

plt.scatter(real_tsne[:, 0], real_tsne[:, 1], color='blue', alpha=0.5, label='Real Images')
plt.scatter(fake_tsne[:, 0], fake_tsne[:, 1], color='orange', alpha=0.5, label='Fake Images')
# 绘制真实图像
plt.scatter(real_proto_tsne[0], real_proto_tsne[1], color='red', marker='x', s=100, label='Prototype Real')
plt.scatter(fake_proto_tsne[0], fake_proto_tsne[1], color='green', marker='x', s=100, label='Prototype Fake')
# plt.title("")
plt.legend()
plt.grid(True)
plt.show()





# # 执行 t-SNE
# tsne = TSNE(n_components=2, random_state=42)
# # tsne_results = tsne.fit_transform(all_features)

# tsne_results = tsne.fit_transform(features)  # features 是你所有图像的特征集合

# # 使用 labels 区分真实和假图像
# colors = ['blue' if label == 0 else 'orange' for label in labels]  # 假设 label == 0 代表真实，1 代表假

# plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=colors, alpha=0.5)
# plt.title("t-SNE Visualization of Real and Fake Images")
# plt.xlabel("t-SNE Feature 1")
# plt.ylabel("t-SNE Feature 2")
# plt.show()

# plt.scatter(tsne_results[real_indices, 0], tsne_results[real_indices, 1], color='blue', alpha=0.5, label='Real Images')
# plt.scatter(tsne_results[fake_indices, 0], tsne_results[fake_indices, 1], color='red', alpha=0.5, label='Fake Images')
# # 绘制 t-SNE 结果
# plt.scatter(tsne_results[:-2, 0], tsne_results[:-2, 1], c=labels, cmap='viridis', alpha=0.5, label='Images')  

# # 真实图像的原型
# plt.scatter(tsne_results[-2, 0], tsne_results[-2, 1], color='purple', marker='x', s=100, label='Prototype Real')

# # 假图像的原型
# plt.scatter(tsne_results[-1, 0], tsne_results[-1, 1], color='orange', marker='x', s=100, label='Prototype Fake')
# plt.title("Laion-Dalle feature distribution visualization including Prototype")
# plt.legend()
# plt.grid(True)
# plt.show()


# prototypes = torch.stack([prototype_real, prototype_fake])  # 将两个原型堆叠起来

# # 使用t-SNE处理所有特征（不包括原型）
# tsne = TSNE(n_components=2, random_state=42)
# tsne_results = tsne.fit_transform(features)
# # 现在处理原型
# tsne_prototypes = tsne.fit_transform(prototypes)

# # 正确索引真实和假图像的t-SNE结果
# real_tsne = tsne_results[labels == 0]
# fake_tsne = tsne_results[labels == 1]

# # 直接处理原型的t-SNE结果
# real_proto_tsne = tsne_prototypes[0]  # 真实图像的原型
# fake_proto_tsne = tsne_prototypes[1] 

# plt.figure(figsize=(10, 6))
# plt.scatter(real_tsne[:, 0], real_tsne[:, 1], color='blue', alpha=0.5, label='Real Images')
# plt.scatter(fake_tsne[:, 0], fake_tsne[:, 1], color='red', alpha=0.5, label='Fake Images')
# plt.scatter(real_proto_tsne[0], real_proto_tsne[1], color='darkblue', marker='x', s=100, label='Prototype Real')
# plt.scatter(fake_proto_tsne[0], fake_proto_tsne[1], color='darkred', marker='x', s=100, label='Prototype Fake')
# plt.title('t-SNE Visualization of Image Features')
# plt.xlabel('t-SNE Feature 1')
# plt.ylabel('t-SNE Feature 2')
# plt.legend()
# plt.grid(True)
# plt.show()


