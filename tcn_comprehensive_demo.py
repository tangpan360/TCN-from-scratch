"""
TCN 综合演示脚本
一次性展示所有核心概念，无需交互
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tcn_implementation import CausalConv1d, TemporalBlock, TemporalConvNet, create_sample_data

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def demo_causal_convolution():
    """演示因果卷积的工作原理"""
    print("=" * 80)
    print("🔍 第一部分：因果卷积 (Causal Convolution)")
    print("=" * 80)
    
    print("\n核心思想：")
    print("- 确保t时刻的输出只依赖于t及之前的输入")
    print("- 通过左填充实现'只看过去'的效果")
    print("- 这对时序预测任务至关重要\n")
    
    # 创建简单的示例
    sequence_length = 10
    input_data = torch.randn(1, 1, sequence_length)
    
    print(f"📊 输入序列长度: {sequence_length}")
    print(f"输入数据形状: {input_data.shape}")
    
    input_sample = input_data.squeeze().numpy()[:5]
    input_str = ", ".join([f"{x:.3f}" for x in input_sample])
    print(f"输入数据: [{input_str}]... (显示前5个值)\n")
    
    # 测试不同kernel size的因果卷积
    kernel_sizes = [2, 3, 5]
    
    print("🔧 不同卷积核大小的效果:")
    for kernel_size in kernel_sizes:
        causal_conv = CausalConv1d(1, 1, kernel_size, dilation=1)
        
        with torch.no_grad():
            output = causal_conv(input_data)
        
        receptive_info = causal_conv.get_receptive_field_info()
        print(f"  核大小 {kernel_size}:")
        print(f"    - 填充大小: {causal_conv.padding}")
        print(f"    - 输出形状: {output.shape}")
        print(f"    - 有效核大小: {receptive_info['effective_kernel_size']}")
        print(f"    - 输入范围: [{input_data.min():.3f}, {input_data.max():.3f}]")
        print(f"    - 输出范围: [{output.min():.3f}, {output.max():.3f}]")
    
    print("\n💡 关键观察：")
    print("1. 输出序列长度与输入相同（因果填充的作用）")
    print("2. 不同核大小产生不同的平滑效果")
    print("3. 输出只依赖于当前和过去的输入")


def demo_dilated_convolution():
    """演示膨胀卷积的效果"""
    print("\n" + "=" * 80)
    print("🚀 第二部分：膨胀卷积 (Dilated Convolution)")
    print("=" * 80)
    
    print("\n核心思想：")
    print("- 通过'跳跃'采样扩大感受野")
    print("- 用少量参数捕获长期依赖")
    print("- 膨胀率决定采样的间隔\n")
    
    # 创建长序列进行演示
    sequence_length = 20
    input_data = torch.cos(torch.linspace(0, 4*np.pi, sequence_length)).unsqueeze(0).unsqueeze(0)
    
    print(f"📊 使用余弦波作为输入 (长度={sequence_length})")
    print(f"输入数据形状: {input_data.shape}\n")
    
    # 测试不同膨胀率
    dilations = [1, 2, 4, 8]
    kernel_size = 3
    
    print("🔧 不同膨胀率的效果:")
    for dilation in dilations:
        dilated_conv = CausalConv1d(1, 1, kernel_size, dilation)
        
        with torch.no_grad():
            output = dilated_conv(input_data)
        
        receptive_info = dilated_conv.get_receptive_field_info()
        
        print(f"  膨胀率 {dilation}:")
        print(f"    - 有效核大小: {receptive_info['effective_kernel_size']}")
        print(f"    - 填充大小: {receptive_info['padding']}")
        print(f"    - 理论感受野: {receptive_info['effective_kernel_size']}")

    print("\n💡 关键观察：")
    print("1. 膨胀率越大，感受野越大")
    print("2. 大膨胀率能捕获更长期的模式")
    print("3. 输出平滑程度随膨胀率变化")


def demo_residual_connection():
    """演示残差连接的作用"""
    print("\n" + "=" * 80)
    print("🔗 第三部分：残差连接 (Residual Connection)")
    print("=" * 80)
    
    print("\n核心思想：")
    print("- 允许信息直接传播，缓解梯度消失")
    print("- output = F(x) + x，学习残差而非直接映射")
    print("- 使深层网络更容易训练\n")
    
    # 创建测试数据
    batch_size = 4
    sequence_length = 20
    input_channels = 8
    output_channels = 8
    
    input_data = torch.randn(batch_size, input_channels, sequence_length)
    
    print(f"📊 测试数据: {input_data.shape}")
    print(f"输入统计: 均值={input_data.mean():.3f}, 标准差={input_data.std():.3f}\n")
    
    # 创建时序块（包含残差连接）
    temporal_block = TemporalBlock(
        n_inputs=input_channels,
        n_outputs=output_channels, 
        kernel_size=3,
        dilation=2,
        dropout=0.0  # 关闭dropout便于观察
    )
    
    print("🔧 时序块配置:")
    print(f"  - 输入通道: {input_channels}")
    print(f"  - 输出通道: {output_channels}")
    print(f"  - 膨胀率: 2")
    print(f"  - 核大小: 3\n")
    
    # 前向传播并收集中间结果
    temporal_block.eval()
    with torch.no_grad():
        # 保存原始输入
        residual = input_data
        
        # 第一个卷积
        out = temporal_block.conv1(input_data)
        out = temporal_block.relu(out)
        conv1_output = out.clone()
        
        # 第二个卷积  
        out = temporal_block.conv2(out)
        out = temporal_block.relu(out)
        conv2_output = out.clone()
        
        # 残差连接
        final_output = temporal_block.relu(out + residual)
    
    # 分析残差连接的数值特征（非正式指标，仅用于直观理解）
    residual_magnitude = residual.abs().mean()
    final_magnitude = final_output.abs().mean()
    residual_ratio = residual_magnitude / final_magnitude
    
    print("📈 前向传播分析:")
    print(f"  - 输入统计: 均值={input_data.mean():.3f}, 标准差={input_data.std():.3f}")
    print(f"  - Conv1输出: 均值={conv1_output.mean():.3f}, 标准差={conv1_output.std():.3f}")
    print(f"  - Conv2输出: 均值={conv2_output.mean():.3f}, 标准差={conv2_output.std():.3f}")
    print(f"  - 最终输出: 均值={final_output.mean():.3f}, 标准差={final_output.std():.3f}")
    print(f"  - 残差vs输出幅度比: {residual_ratio:.3f} (原始输入在输出中的数值占比)")
    
    print("\n💡 关键观察：")
    print("1. 残差连接保留了原始信息")
    print("2. 输出结合了新学习的特征和原始输入")
    print("3. 幅度比反映了原始信息在输出中的相对强度（非标准学术指标）")


def demo_stacked_tcn():
    """演示堆叠TCN的效果"""
    print("\n" + "=" * 80)
    print("🏗️ 第四部分：堆叠的TCN")
    print("=" * 80)
    
    print("\n核心思想：")
    print("- 多层级联，膨胀率指数增长")
    print("- 每层捕获不同时间尺度的模式")
    print("- 感受野呈指数级扩大\n")
    
    # 创建测试数据
    batch_size = 2
    sequence_length = 50
    num_inputs = 3
    
    X, _ = create_sample_data(batch_size, sequence_length, num_inputs)
    
    print(f"📊 测试数据: {X.shape}")
    print(f"序列长度: {sequence_length}")
    
    # 创建不同深度的TCN进行对比
    configurations = [
        {"name": "浅层TCN", "channels": [16, 16]},
        {"name": "中层TCN", "channels": [16, 16, 16, 16]},
        {"name": "深层TCN", "channels": [16, 16, 16, 16, 16, 16]}
    ]
    
    print("\n🔧 不同深度TCN对比:")
    for config in configurations:
        tcn = TemporalConvNet(num_inputs, config["channels"], kernel_size=3, dropout=0.1)
        network_info = tcn.get_network_info()
        
        print(f"\n  {config['name']}:")
        print(f"    - 层数: {network_info['num_levels']}")
        print(f"    - 理论感受野: {network_info['receptive_field']}")
        print(f"    - 参数量: {network_info['parameters']:,}")
        print(f"    - 膨胀率: {network_info['dilations']}")
        
        # 分析感受野vs序列长度的关系
        rf_ratio = network_info['receptive_field'] / sequence_length
        print(f"    - 感受野覆盖率: {rf_ratio:.2%}")
    
    print("\n💡 关键观察：")
    print("1. 更深的网络具有更大的感受野")
    print("2. 感受野应该与任务需求匹配")
    print("3. 过大的感受野可能导致过拟合")


def demo_practical_application():
    """演示实际应用：时间序列预测"""
    print("\n" + "=" * 80)
    print("🎯 第五部分：实际应用 - 时间序列预测")
    print("=" * 80)
    
    print("\n🎯 让我们用TCN解决一个实际问题：时间序列预测")
    print("任务：预测余弦波的下一个值\n")
    
    # 生成时间序列数据
    def generate_cosine_wave(length, freq=1, noise_level=0.1):
        t = np.linspace(0, 4*np.pi, length)
        signal = np.cos(freq * t) + noise_level * np.random.randn(length)
        return signal
    
    # 创建数据集
    sequence_length = 100
    prediction_length = 10
    num_samples = 50  # 减少样本数量用于演示
    
    print(f"📊 数据集配置:")
    print(f"  - 样本数量: {num_samples}")
    print(f"  - 序列长度: {sequence_length}")
    print(f"  - 预测长度: {prediction_length}")
    
    # 生成训练数据
    X_train = []
    y_train = []
    
    for i in range(num_samples):
        # 随机频率和噪声
        freq = 0.5 + np.random.rand() * 2
        noise = 0.05 + np.random.rand() * 0.1
        
        # 生成完整序列
        full_sequence = generate_cosine_wave(sequence_length + prediction_length, freq, noise)
        
        # 分割输入和目标
        X_train.append(full_sequence[:sequence_length])
        y_train.append(full_sequence[sequence_length:sequence_length + prediction_length])
    
    X_train = torch.FloatTensor(X_train).unsqueeze(1)  # (samples, 1, seq_len)
    y_train = torch.FloatTensor(y_train).unsqueeze(1)  # (samples, 1, pred_len)
    
    print(f"  - 训练输入形状: {X_train.shape}")
    print(f"  - 训练目标形状: {y_train.shape}\n")
    
    # 创建TCN预测模型
    from tcn_implementation import TCNPredictor

    model = TCNPredictor(
        num_inputs=1,
        num_channels=[32, 32, 32, 32],
        prediction_length=prediction_length,
        kernel_size=3,
        dropout=0.1
    )

    print(f"🔧 模型配置:")
    network_info = model.tcn.get_network_info()
    print(f"  - 参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - 感受野: {network_info['receptive_field']}")
    print(f"  - TCN层数: {network_info['num_levels']}\n")

    # 简单训练（只做几个epoch演示）
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print("🚀 开始训练...")
    model.train()
    losses = []

    for epoch in range(5):  # 只训练5个epoch用于演示
        epoch_loss = 0
        num_batches = 0

        for i in range(0, len(X_train), 16):  # batch_size=16
            batch_x = X_train[i:i+16]
            batch_y = y_train[i:i+16]

            if len(batch_x) < 2:  # 跳过太小的批次
                continue

            optimizer.zero_grad()
            predictions = model(batch_x)

            # 只使用最后一个时间步的预测
            pred_last = predictions[:, :, -1:]  # (batch, pred_len, 1)
            target = batch_y  # (batch, 1, pred_len)

            # 调整维度匹配
            loss = criterion(pred_last.squeeze(-1), target.squeeze(1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            print(f"  Epoch {epoch+1}/5: Loss = {avg_loss:.6f}")

    print("✅ 训练完成!\n")

    # 测试预测效果
    model.eval()
    test_idx = 0
    test_input = X_train[test_idx:test_idx+1]
    test_target = y_train[test_idx:test_idx+1]

    with torch.no_grad():
        prediction = model(test_input)
        pred_values = prediction[0, :, -1].numpy()  # 最后一个时间步的预测
        true_values = test_target[0, 0].numpy()

    # 计算预测误差
    mse = np.mean((pred_values - true_values) ** 2)
    mae = np.mean(np.abs(pred_values - true_values))

    print(f"📊 预测性能:")
    print(f"  - 均方误差 (MSE): {mse:.6f}")
    print(f"  - 平均绝对误差 (MAE): {mae:.6f}")

    if len(pred_values) == len(true_values):
        corr = np.corrcoef(pred_values, true_values)[0,1]
        print(f"  - 相关系数: {corr:.4f}")

    print("\n💡 应用总结:")
    print("1. TCN能够有效学习时间序列模式")
    print("2. 适合需要长期依赖的预测任务")
    print("3. 训练速度比RNN/LSTM快")
    print("4. 可以通过调整网络深度控制感受野")


def main():
    """主函数：运行完整的演示"""
    print("🎓 TCN (时序卷积网络) 综合演示")
    print("本演示将全面展示TCN的核心概念和实际应用")
    print("=" * 80)

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        # 逐步演示各个概念
        demo_causal_convolution()
        demo_dilated_convolution()
        demo_residual_connection()
        demo_stacked_tcn()
        demo_practical_application()

        print("\n" + "=" * 80)
        print("🎉 TCN综合演示完成！")
        print("=" * 80)
        print("\n📝 核心知识总结:")
        print("1. 因果卷积确保时序建模的因果性")
        print("2. 膨胀卷积有效扩大感受野")
        print("3. 残差连接解决深网络训练问题")
        print("4. 堆叠结构捕获多时间尺度特征")
        print("5. TCN在许多时序任务上优于RNN/LSTM")
        print("\n🔬 实验要点:")
        print("- 感受野大小应该匹配任务需求")
        print("- 网络深度影响模型容量和计算成本")
        print("- 膨胀卷积是TCN的核心创新")
        print("- 因果卷积保证了预测任务的合理性")
        print("\n🎯 你现在已经完全理解了TCN的工作原理！")

    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        print("这通常是环境配置问题，不影响TCN概念的理解")

if __name__ == "__main__":
    main()
