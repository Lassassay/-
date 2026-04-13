
import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from matplotlib.colors import PowerNorm

# 损失 = MSE + λ * (1 - 覆盖率) + μ * 分布方差 + α * 总体密度（用于抑制过度投放）
class EfficiencyLoss(nn.Module):
    def __init__(self, lambda_cover=1.0, mu_var=0.05, alpha_density=0.05):
        super(EfficiencyLoss, self).__init__()
        self.lambda_cover = lambda_cover
        self.mu_var = mu_var
        self.alpha_density = alpha_density
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target, pop_density):
        mse_loss = self.mse(pred, target)
        # 覆盖率：汽车分布与人口密度的点积（越高越好）
        coverage = torch.mean(pred * pop_density)
        cover_loss = 1 - coverage  # 反转以最小化
        # 分布方差：惩罚过度集中（模拟交通拥堵）
        var_loss = torch.var(pred)
        # 密度惩罚：抑制总体输出密度
        density_loss = torch.mean(pred)
        total_loss = mse_loss + self.lambda_cover * cover_loss + self.mu_var * var_loss + self.alpha_density * density_loss
        return total_loss

# 2. CNN网络架构：U-Net风格，用于热图到热图映射
class CarDistributionCNN(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, img_size=128):
        super(CarDistributionCNN, self).__init__()
        self.img_size = img_size
        
        # 编码器：提取人口密度特征（如高密度聚类）
        # 使用 padding_mode='reflect' 进行镜像拓展，防止边缘权重过低
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 下采样
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 瓶颈层：学习优化映射
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, padding_mode='reflect'),
            nn.ReLU()
        )
        
        # 解码器：生成汽车分布，上采样恢复分辨率
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),  # 上采样
            nn.Conv2d(64, 64, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode='reflect'),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.Conv2d(32, 32, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, padding_mode='reflect'),
            nn.ReLU()
        )
        
        # 输出层：生成热图
        self.output = nn.Conv2d(32, output_channels, 1)
        
    def forward(self, x):
        # 编码
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        b = self.bottleneck(e2)
        
        # 解码（简单跳跃连接省略以简化；实际可添加以提升性能）
        d1 = self.decoder1(b)
        d2 = self.decoder2(d1)
        
        out = torch.sigmoid(self.output(d2))  # [0,1]范围热图
        return out

# 3. 合成数据集类：生成人口热图和高效汽车分布标签
class PopulationDataset(Dataset):
    def __init__(self, num_samples=1000, img_size=128):
        self.num_samples = num_samples
        self.img_size = img_size
        self.data = []
        self.labels = []
        for _ in range(num_samples):
            # 按照 sudom.py 的方法生成单中心二维高斯人口密度热图
            x = np.linspace(0, img_size - 1, img_size)
            y = np.linspace(0, img_size - 1, img_size)
            X, Y = np.meshgrid(x, y)

            center_x = np.random.uniform(0, img_size - 1)
            center_y = np.random.uniform(0, img_size - 1)
            sigma = np.random.uniform(img_size * 0.12, img_size * 0.22)
            amplitude = np.random.uniform(0.6, 1.0)

            pop_map = amplitude * np.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))
            pop_map = np.clip(pop_map, 0, None)
            pop_map = (pop_map - pop_map.min()) / (pop_map.max() - pop_map.min() + 1e-8)
            pop_map = torch.tensor(pop_map, dtype=torch.float32).unsqueeze(0)  # [1,H,W]

            # 生成标签：高效汽车分布（偏向高人口区，但稍分散以优化效率）
            car_map = pop_map.squeeze().numpy()
            car_map = car_map * 0.8 + np.random.rand(*car_map.shape) * 0.2  # 轻微随机化模拟优化
            high_mask = car_map > 0.7
            low_mask = car_map < 0.3
            car_map[high_mask] *= 1.2
            car_map[low_mask] *= 0.5
            car_map = np.clip(car_map, 0, 1)
            car_map = torch.tensor(car_map, dtype=torch.float32).unsqueeze(0)

            self.data.append(pop_map)
            self.labels.append(car_map)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 4. 训练函数
def train_model(num_epochs=50, batch_size=32, lr=0.005, img_size=128, model=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model is None:
        model = CarDistributionCNN(img_size=img_size).to(device)
    else:
        model = model.to(device)
    criterion = EfficiencyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    dataset = PopulationDataset(num_samples=1000, img_size=img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 记录训练数据
    history = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for pop_batch, car_batch in dataloader:
            pop_batch, car_batch = pop_batch.to(device), car_batch.to(device)
            optimizer.zero_grad()
            outputs = model(pop_batch)
            loss = criterion(outputs, car_batch, pop_batch)  # 传入人口密度用于效率计算
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
        
        # 保存每轮的 loss
        history.append({
            'Epoch': epoch + 1,
            'Loss': avg_loss
        })
    
    # 导出到 Excel
    if history:
        excel_path = 'training_history.xlsx'
        new_df = pd.DataFrame(history)
        
        if os.path.exists(excel_path):
            # 如果文件已存在，读取并追加
            old_df = pd.read_excel(excel_path)
            # 获取上一次训练的最大 Epoch，使本次编号连续
            last_epoch = old_df['Epoch'].max()
            new_df['Epoch'] = new_df['Epoch'] + last_epoch
            df = pd.concat([old_df, new_df], ignore_index=True)
            print(f'追加训练数据至 {excel_path}，当前总 Epoch: {df["Epoch"].max()}')
        else:
            df = new_df
            print(f'创建新的训练数据文件 {excel_path}')
            
        df.to_excel(excel_path, index=False)
    
    torch.save(model.state_dict(), 'car_distribution_cnn.pth')
    return model

# 5. 推理函数：输入人口热图，输出汽车分布
def predict_distribution(model, pop_heatmap_path=None, img_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    if pop_heatmap_path:
        # 加载真实热图（假设numpy数组或图像文件）
        if pop_heatmap_path.endswith('.npy'):
            pop_map = np.load(pop_heatmap_path)
        else:
            img = Image.open(pop_heatmap_path).convert('L')
            if img.size != (img_size, img_size):
                img = img.resize((img_size, img_size), Image.BILINEAR)
            pop_map = np.array(img, dtype=np.float32) / 255.0
        if pop_map.ndim == 3:
            pop_map = np.mean(pop_map, axis=2)
        pop_map = (pop_map - pop_map.min()) / (pop_map.max() - pop_map.min() + 1e-8)
        pop_map = torch.tensor(pop_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    else:
        # 示例：生成随机人口热图
        pop_map = torch.rand(1, 1, img_size, img_size)
    
    pop_map = pop_map.to(device)
    with torch.no_grad():
        car_map = model(pop_map)
    
    # 可视化
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Input: Population Density Heatmap')
    plt.imshow(pop_map.cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.title('Output: Optimized Car Distribution Heatmap')
    plt.imshow(car_map.cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
    plt.colorbar()
    plt.show()
    
    return car_map.cpu().numpy()

MODEL_PATH = 'car_distribution_cnn.pth'

def load_or_train_model(num_epochs=50, batch_size=32, lr=0.005, img_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CarDistributionCNN(img_size=img_size).to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f'Loaded saved model from {MODEL_PATH}')
        if num_epochs == 0:
            return model
        print(f'Continuing training for {num_epochs} epochs...')
    else:
        print('Saved model not found, training new model...')
        if num_epochs == 0:
            num_epochs = 20  # 如果没模型又设为0，则默认训练20轮
    return train_model(num_epochs=num_epochs, batch_size=batch_size, lr=lr, img_size=img_size, model=model)

# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and predict car distribution from a population heatmap.')
    parser.add_argument('--input_image', type=str, default=None, help='Path to input population image (.npy, .png, .jpg)')
    parser.add_argument('--train_epochs', type=int, default=0, help='Number of training epochs (set to 0 to skip training)')
    parser.add_argument('--force_train', action='store_true', help='Retrain the model from scratch')
    args = parser.parse_args()

    if args.force_train and os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
        # 如果是强制重练，且 epochs 为 0，则强制设为默认 20 轮
        if args.train_epochs == 0:
            args.train_epochs = 20

    model = load_or_train_model(num_epochs=args.train_epochs)

    output = predict_distribution(model, args.input_image) if args.input_image else predict_distribution(model)
    print("预测的汽车分布热图形状:", output.shape)
