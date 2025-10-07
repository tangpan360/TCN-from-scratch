"""
TCN (Temporal Convolutional Network) 从零实现
包含完整的注释和调试功能，帮助理解每个组件的作用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import math

class CausalConv1d(nn.Module):
    """
    因果卷积层 (Causal Convolution)
    
    核心作用：确保t时刻的输出只依赖于t及之前时刻的输入
    实现方式：通过左填充(left padding)实现
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 dilation: int = 1, bias: bool = True):
        super(CausalConv1d, self).__init__()
        
        # 计算因果填充大小
        # padding = (kernel_size - 1) * dilation 确保输出长度等于输入长度
        self.padding = (kernel_size - 1) * dilation
        self.dilation = dilation
        self.kernel_size = kernel_size
        
        # 标准1D卷积层
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels, 
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, channels, sequence_length)
            
        Returns:
            输出张量，形状与输入相同
        """
        # 左填充，确保因果性
        # F.pad的参数格式：(左填充, 右填充, 上填充, 下填充, ...)
        padded_x = F.pad(x, (self.padding, 0))
        
        # 进行卷积操作
        output = self.conv(padded_x)
        
        return output
    
    def get_receptive_field_info(self) -> dict:
        """获取感受野信息，用于调试和理解"""
        return {
            'kernel_size': self.kernel_size,
            'dilation': self.dilation,
            'padding': self.padding,
            'effective_kernel_size': (self.kernel_size - 1) * self.dilation + 1
        }


class TemporalBlock(nn.Module):
    """
    时序块 (Temporal Block) - TCN的基本构建单元
    
    结构：
    输入 → 卷积1 → 激活 → Dropout → 卷积2 → 激活 → Dropout → 输出
     ↓                                                      ↑
     └─────────────── 残差连接 ──────────────────────────────┘
    """
    
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int,
                 dilation: int, dropout: float = 0.2):
        super(TemporalBlock, self).__init__()
        
        # 存储参数用于调试
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.dilation = dilation
        
        # 第一个因果卷积层
        self.conv1 = CausalConv1d(n_inputs, n_outputs, kernel_size, dilation)
        
        # 第二个因果卷积层
        self.conv2 = CausalConv1d(n_outputs, n_outputs, kernel_size, dilation)
        
        # Dropout层用于正则化
        self.dropout = nn.Dropout(dropout)
        
        # 激活函数
        self.relu = nn.ReLU()
        
        # 残差连接的维度匹配层
        # 如果输入输出维度不同，需要用1x1卷积进行维度变换
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        
        # 权重初始化
        self.init_weights()
    
    def init_weights(self):
        """权重初始化 - 使用Kaiming初始化提高训练稳定性"""
        nn.init.kaiming_normal_(self.conv1.conv.weight)
        nn.init.kaiming_normal_(self.conv2.conv.weight)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight)
    
    def forward(self, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, channels, sequence_length)
            debug: 是否输出调试信息
            
        Returns:
            输出张量，形状与输入相同
        """
        # 保存原始输入用于残差连接
        residual = x
        
        # 第一个卷积 + 激活 + dropout
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        if debug:
            print(f"After conv1: shape={out.shape}, mean={out.mean():.4f}, std={out.std():.4f}")
        
        # 第二个卷积 + 激活 + dropout  
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        if debug:
            print(f"After conv2: shape={out.shape}, mean={out.mean():.4f}, std={out.std():.4f}")
        
        # 残差连接：如果维度不匹配，先进行维度变换
        if self.downsample is not None:
            residual = self.downsample(residual)
            
        # 残差连接 + 最终激活
        output = self.relu(out + residual)
        
        if debug:
            print(f"After residual: shape={output.shape}, mean={output.mean():.4f}, std={output.std():.4f}")
            print(f"Residual contribution: {(residual.abs().mean() / output.abs().mean()).item():.4f}")
        
        return output
    
    def get_receptive_field(self) -> int:
        """计算该块的感受野大小"""
        return 2 * (self.conv1.kernel_size - 1) * self.dilation + 1


class TemporalConvNet(nn.Module):
    """
    完整的时序卷积网络 (TCN)
    
    架构：多个TemporalBlock的级联，膨胀率呈指数增长
    """
    
    def __init__(self, num_inputs: int, num_channels: List[int], 
                 kernel_size: int = 2, dropout: float = 0.2):
        """
        初始化TCN
        
        Args:
            num_inputs: 输入特征维度
            num_channels: 每层的通道数列表，例如[25, 25, 25, 25]
            kernel_size: 卷积核大小
            dropout: dropout概率
        """
        super(TemporalConvNet, self).__init__()
        
        # 存储配置参数
        self.num_inputs = num_inputs
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.num_levels = len(num_channels)
        
        # 构建网络层
        layers = []
        
        for i in range(self.num_levels):
            # 膨胀率呈指数增长：1, 2, 4, 8, 16, ...
            dilation_size = 2 ** i
            
            # 确定输入输出通道数
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # 创建时序块
            layers.append(TemporalBlock(
                in_channels, 
                out_channels,
                kernel_size, 
                dilation=dilation_size, 
                dropout=dropout
            ))
        
        # 将所有层组合成序列
        self.network = nn.Sequential(*layers)
        
        # 计算总感受野
        self.receptive_field = self.calculate_receptive_field()
    
    def forward(self, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, num_inputs, sequence_length)
            debug: 是否输出详细调试信息
            
        Returns:
            输出张量 (batch_size, num_channels[-1], sequence_length)
        """
        if debug:
            print(f"TCN Input: shape={x.shape}, mean={x.mean():.4f}, std={x.std():.4f}")
            print(f"Total receptive field: {self.receptive_field}")
            print("-" * 50)
        
        # 逐层传播
        for i, layer in enumerate(self.network):
            x = layer(x, debug=debug)
            if debug:
                print(f"After layer {i+1}: shape={x.shape}")
                print(f"Layer {i+1} dilation: {layer.dilation}")
                print(f"Layer {i+1} receptive field: {layer.get_receptive_field()}")
                print("-" * 30)
        
        return x
    
    def calculate_receptive_field(self) -> int:
        """
        计算整个网络的感受野
        
        公式：receptive_field = 1 + 2 * (kernel_size - 1) * sum(dilations)
        """
        total_dilation = sum(2**i for i in range(self.num_levels))
        receptive_field = 1 + 2 * (self.kernel_size - 1) * total_dilation
        return receptive_field
    
    def get_network_info(self) -> dict:
        """获取网络结构信息，用于分析和调试"""
        info = {
            'num_levels': self.num_levels,
            'num_channels': self.num_channels,
            'kernel_size': self.kernel_size,
            'receptive_field': self.receptive_field,
            'dilations': [2**i for i in range(self.num_levels)],
            'parameters': sum(p.numel() for p in self.parameters()),
            'layers_info': []
        }
        
        for i, layer in enumerate(self.network):
            layer_info = {
                'layer_idx': i,
                'dilation': 2**i,
                'in_channels': layer.n_inputs,
                'out_channels': layer.n_outputs,
                'receptive_field': layer.get_receptive_field(),
                'parameters': sum(p.numel() for p in layer.parameters())
            }
            info['layers_info'].append(layer_info)
        
        return info


class TCNClassifier(nn.Module):
    """
    基于TCN的分类器
    用于时序分类任务，如动作识别、情感分析等
    """
    
    def __init__(self, num_inputs: int, num_channels: List[int], 
                 num_classes: int, kernel_size: int = 2, dropout: float = 0.2):
        super(TCNClassifier, self).__init__()
        
        # TCN主干网络
        self.tcn = TemporalConvNet(num_inputs, num_channels, kernel_size, dropout)
        
        # 分类头：全连接层
        self.classifier = nn.Linear(num_channels[-1], num_classes)
        
        # 全局平均池化（可选）
        self.use_global_pooling = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入序列 (batch_size, num_inputs, sequence_length)
            
        Returns:
            类别概率 (batch_size, num_classes)
        """
        # TCN特征提取
        features = self.tcn(x)  # (batch_size, num_channels[-1], sequence_length)
        
        if self.use_global_pooling:
            # 全局平均池化：对时间维度求平均
            pooled = features.mean(dim=2)  # (batch_size, num_channels[-1])
        else:
            # 使用最后一个时间步的特征
            pooled = features[:, :, -1]  # (batch_size, num_channels[-1])
        
        # 分类
        output = self.classifier(pooled)  # (batch_size, num_classes)
        
        return output


class TCNPredictor(nn.Module):
    """
    基于TCN的时序预测器
    用于时间序列预测任务，如股价预测、温度预测等
    """
    
    def __init__(self, num_inputs: int, num_channels: List[int], 
                 prediction_length: int, kernel_size: int = 2, dropout: float = 0.2):
        super(TCNPredictor, self).__init__()
        
        # TCN主干网络
        self.tcn = TemporalConvNet(num_inputs, num_channels, kernel_size, dropout)
        
        # 预测头：卷积层或全连接层
        self.predictor = nn.Conv1d(num_channels[-1], prediction_length, 1)
        
        self.prediction_length = prediction_length
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入序列 (batch_size, num_inputs, sequence_length)
            
        Returns:
            预测结果 (batch_size, prediction_length, sequence_length)
        """
        # TCN特征提取
        features = self.tcn(x)  # (batch_size, num_channels[-1], sequence_length)
        
        # 预测
        predictions = self.predictor(features)  # (batch_size, prediction_length, sequence_length)
        
        return predictions


def create_sample_data(batch_size: int = 32, sequence_length: int = 100, 
                      num_features: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    创建示例数据用于测试和调试
    
    生成余弦波 + 噪声的时序数据
    """
    t = torch.linspace(0, 4*np.pi, sequence_length)
    
    # 创建多个余弦波的组合
    data = []
    for i in range(batch_size):
        # 不同频率和相位的余弦波
        freq1 = 1 + 0.5 * np.random.randn()
        freq2 = 2 + 0.3 * np.random.randn() 
        phase1 = 2 * np.pi * np.random.rand()
        phase2 = 2 * np.pi * np.random.rand()
        
        signal = (torch.cos(freq1 * t + phase1) + 
                 0.5 * torch.cos(freq2 * t + phase2) + 
                 0.1 * torch.randn(sequence_length))
        
        if num_features > 1:
            # 添加更多特征维度
            extra_features = torch.randn(num_features - 1, sequence_length) * 0.3
            signal = torch.stack([signal] + [extra_features[j] for j in range(num_features - 1)])
        else:
            signal = signal.unsqueeze(0)
            
        data.append(signal)
    
    # 组装成批次
    X = torch.stack(data)  # (batch_size, num_features, sequence_length)
    
    # 创建标签（用于分类任务的示例）
    y = torch.randint(0, 3, (batch_size,))  # 3个类别
    
    return X, y


if __name__ == "__main__":
    print("=" * 60)
    print("TCN (时序卷积网络) 从零实现 - 测试和调试")
    print("=" * 60)
    
    # 设置随机种子确保结果可复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 网络配置
    num_inputs = 3  # 输入特征维度
    num_channels = [25, 25, 25, 25]  # 每层通道数
    kernel_size = 3
    dropout = 0.2
    
    print(f"\n网络配置:")
    print(f"输入特征维度: {num_inputs}")
    print(f"网络层数: {len(num_channels)}")
    print(f"每层通道数: {num_channels}")
    print(f"卷积核大小: {kernel_size}")
    
    # 创建TCN网络
    tcn = TemporalConvNet(num_inputs, num_channels, kernel_size, dropout)
    
    # 打印网络信息
    network_info = tcn.get_network_info()
    print(f"\n网络结构分析:")
    print(f"总参数量: {network_info['parameters']:,}")
    print(f"理论感受野: {network_info['receptive_field']}")
    print(f"膨胀率序列: {network_info['dilations']}")
    
    print(f"\n各层详细信息:")
    for layer_info in network_info['layers_info']:
        print(f"层{layer_info['layer_idx']+1}: "
              f"膨胀率={layer_info['dilation']}, "
              f"通道数={layer_info['in_channels']}→{layer_info['out_channels']}, "
              f"感受野={layer_info['receptive_field']}, "
              f"参数={layer_info['parameters']:,}")
    
    # 创建测试数据
    batch_size = 8
    sequence_length = 50
    X, y = create_sample_data(batch_size, sequence_length, num_inputs)
    
    print(f"\n测试数据:")
    print(f"输入形状: {X.shape}")
    print(f"标签形状: {y.shape}")
    print(f"输入数据范围: [{X.min():.3f}, {X.max():.3f}]")
    
    # 前向传播测试
    print(f"\n=" * 40)
    print("前向传播测试 (带调试信息)")
    print("=" * 40)
    
    with torch.no_grad():
        output = tcn(X, debug=True)
    
    print(f"\n最终输出:")
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min():.3f}, {output.max():.3f}]")
    print(f"输出均值: {output.mean():.3f}")
    print(f"输出标准差: {output.std():.3f}")
