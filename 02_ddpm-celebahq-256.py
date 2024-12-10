import torch
import numpy as np
import PIL.Image
from diffusers import UNet2DModel, DDPMScheduler
import tqdm

# 1. 初始化模型
repo_id = "google/ddpm-celebahq-256"
model = UNet2DModel.from_pretrained(repo_id, use_safetensors=False)
model.to("cuda")  # 將模型移至 GPU
print(model.config)

# 2. 初始化調度器
scheduler = DDPMScheduler.from_pretrained(repo_id)

# 3. 創建含高斯噪聲的圖片
torch.manual_seed(0)  # 固定隨機種子，保證可重現性
noisy_sample = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size).to("cuda")
print(f"Noisy sample shape: {noisy_sample.shape}")

# 4. 定義顯示圖片的函數
def display_sample(sample, i):
    """
    將 PyTorch Tensor 轉換為 PIL Image 並顯示。
    :param sample: 噪聲樣本（Tensor）
    :param i: 當前步驟數
    """
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = PIL.Image.fromarray(image_processed[0])
    print(f"Image at step {i}")
    image_pil.show()

# 5. 逆擴散過程
sample = noisy_sample
for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
    # 1. 預測噪聲殘差
    with torch.no_grad():
        residual = model(sample, t).sample

    # 2. 計算較少噪聲的圖片，並將 x_t -> x_t-1
    sample = scheduler.step(residual, t, sample).prev_sample

    # 3. 選擇性顯示圖片（每 50 步顯示一次）
    if (i + 1) % 50 == 0:
        display_sample(sample, i + 1)

print("Denoising complete.")