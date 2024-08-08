import clip
import torch

# 加载预训练的 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

last_layer = model.visual.transformer.resblocks[-1].ln_2

# 保存最后一层的状态字典
torch.save(last_layer.state_dict(), 'last_layer.pth')
# 加载保存的最后一层
last_layer_weights = torch.load('last_layer.pth')

# 打印权重和偏置的细节
print("Weight tensor:", last_layer_weights['weight'].shape)  # 如果有权重
print("Bias tensor:", last_layer_weights['bias'].shape)  # 如果有偏置
