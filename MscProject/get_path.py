import clip
import torch

# load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

last_layer = model.visual.transformer.resblocks[-1].ln_2

# Save last layer
torch.save(last_layer.state_dict(), 'last_layer.pth')
# Load last layer
last_layer_weights = torch.load('last_layer.pth')

# 
print("Weight tensor:", last_layer_weights['weight'].shape)  # 如果有权重
print("Bias tensor:", last_layer_weights['bias'].shape)  # 如果有偏置
