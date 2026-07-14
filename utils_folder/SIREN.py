import torch
import torch.nn as nn
import math

class SineLayer(nn.Module):
    """
    SIREN中的单层正弦网络层
    """
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # 第一层初始化: U(-1/n, 1/n)
                bound = 1 / self.in_features
                self.linear.weight.uniform_(-bound, bound)
            else:
                # 后续层初始化: U(-sqrt(6/n)/omega_0, sqrt(6/n)/omega_0)
                bound = math.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
                
    def forward(self, input):
        # 根据公式: sin(omega_0 * W * x + b)
        return torch.sin(self.omega_0 * self.linear(input))


class SIREN(nn.Module):
    """
    完整的SIREN网络类，支持自定义输入/输出维度和隐藏层结构。
    """
    def __init__(self, in_features, out_features, hidden_features=128, 
                 hidden_layers=3, final_activation=None, omega_0=30, norm_scale=None):
        """
        参数:
            in_features (int): 输入维度 (例如，星际转移任务中为 6)
            out_features (int): 输出维度 (例如，星际转移任务中为 3)
            hidden_features (int): 隐藏层神经元数量，默认为 128
            hidden_layers (int): 隐藏层数量，默认为 3
            final_activation (nn.Module): 输出层的激活函数，如 nn.Sigmoid()。默认为 None (即Linear输出)
            omega_0 (float): 正弦函数的频率缩放常数，默认为 30
            norm_scale (float, list, tuple or torch.Tensor): 输入归一化系数，默认不进行归一化。若提供，则输入将除以该系数。
        """
        super().__init__()
        
        if norm_scale is not None:
            if not isinstance(norm_scale, torch.Tensor):
                norm_scale = torch.tensor(norm_scale, dtype=torch.float32)
            self.register_buffer('norm_scale', norm_scale)
        else:
            self.norm_scale = None

        self.net = []
        
        # 1. 第一层 (必须特殊初始化)
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=omega_0))
        
        # 2. 中间隐藏层
        for i in range(hidden_layers - 1):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=omega_0))
            
        # 3. 输出层 (通常是线性层)
        final_linear = nn.Linear(hidden_features, out_features)
        
        # 对输出层也采用类似中间层的初始化方式
        with torch.no_grad():
            bound = math.sqrt(6 / hidden_features) / omega_0
            final_linear.weight.uniform_(-bound, bound)
            
        self.net.append(final_linear)
        
        # 4. 可选的最终激活函数
        if final_activation is not None:
            self.net.append(final_activation)
            
        self.net = nn.Sequential(*self.net)
        
    def forward(self, x):
        if self.norm_scale is not None:
            x = x / self.norm_scale
        return self.net(x)

# --- 使用示例 ---
if __name__ == "__main__":
    # 模拟论文中 Asteroid Landing (小行星着陆) 的架构：7个输入，使用线性+部分Sigmoid输出
    # 简单起见，这里演示一个纯线性输出
    model_asteroid = SIREN(in_features=7, out_features=4, hidden_features=128, hidden_layers=3)
    
    # 模拟 Drone Racing (无人机竞速)：16个输入，4个输出，使用Sigmoid
    model_drone = SIREN(in_features=16, out_features=4, hidden_features=128, 
                        hidden_layers=3, final_activation=nn.Sigmoid())
    
    # 测试前向传播
    dummy_input = torch.randn(32, 16) # Batch size 32, 16 features
    output = model_drone(dummy_input)
    print("Output shape:", output.shape)