import qw_map
import math
import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple
import pickle

import torch
import torch.nn as nn
import pennylane as qml
import einops


class QDenseUndirected_old(torch.nn.Module):
    """Dense variational circuit. Undirected"""

    def __init__(self, qdepth, shape) -> None:
        super().__init__()
        self.qdepth = qdepth
        if isinstance(shape, int):
            shape = (shape, shape)
        self.width, self.height = shape
        self.pixels = self.width * self.height  # 像素总数
        self.wires = math.ceil(math.log2(self.width * self.height))  # 量子比特数目
        self.qdev = qml.device("default.qubit.torch", wires=self.wires)
        # self.qdev = qml.device("lightning.qubit", wires=self.wires)
        # self.qdev = qml.device("default.qubit.jax", wires=self.wires)
        weight_shape = qml.StronglyEntanglingLayers.shape(self.qdepth, self.wires)
        self.weights = torch.nn.Parameter(
            torch.randn(weight_shape, requires_grad=True) * 0.4
        )
        self.qnode = qml.QNode(
            func=self._circuit,
            device=self.qdev,
            interface="torch",
            diff_method="backprop",
        )

    def _circuit(self, inp):
        qml.AmplitudeEmbedding(
            features=inp, wires=range(self.wires), normalize=True, pad_with=0.1
        )
        qml.StronglyEntanglingLayers(
            weights=qw_map.tanh(self.weights), wires=range(self.wires)
        )
        return qml.probs(wires=range(self.wires))

    def _post_process(self, probs):
        # probs = probs[:, ::2] # Drop all probabilities for |xxxx1>
        probs = probs[:, : self.pixels]
        probs = probs * self.pixels
        probs = torch.clamp(probs, 0, 1)
        return probs

    def forward(self, x):
        x = einops.rearrange(x, "b 1 w h -> b (w h)")
        x = self.qnode(x)
        x = self._post_process(x)
        x = einops.rearrange(x, "b (w h) -> b 1 w h", w=self.width, h=self.height)

        return x

    def __repr__(self):
        return f"QDenseUndirected_old(qdepth={self.qdepth}, wires={self.wires})"

    def save_name(self) -> str:
        return f"QDenseUndirected_old{self.qdepth}_w{self.width}_h{self.height}"


class QDenseUndirected_old_noise(torch.nn.Module):
    def __init__(self, qdepth, shape, add_noise=0, device_type="default.qubit.torch") -> None:
        super().__init__()
        self.qdepth = qdepth
        self.add_noise = add_noise
        if isinstance(shape, int):
            shape = (shape, shape)
        self.width, self.height = shape
        self.pixels = self.width * self.height
        self.wires = math.ceil(math.log2(self.width * self.height))
        self.qdev = qml.device(device_type, wires=self.wires)

        weight_shape = qml.StronglyEntanglingLayers.shape(self.qdepth, self.wires)
        self.weights = torch.nn.Parameter(
            torch.randn(weight_shape, requires_grad=True) * 0.4
        )

        self.qnode = qml.QNode(
            func=self._circuit,
            device=self.qdev,
            interface="torch",
            diff_method="backprop"
        )

    def _circuit(self, inp):
        qml.AmplitudeEmbedding(features=inp, wires=range(self.wires), normalize=True, pad_with=0.1)
        qml.StronglyEntanglingLayers(weights=torch.tanh(self.weights), wires=range(self.wires))
        for wire in range(self.wires):
            if self.add_noise == 1:
                qml.PhaseShift(0.05, wires=wire)
            elif self.add_noise == 2:
                qml.AmplitudeDamping(0.05, wires=wire)
            elif self.add_noise == 3:
                qml.DepolarizingChannel(0.02, wires=wire)
        return qml.probs(wires=range(self.wires))

    def _post_process(self, probs):
        probs = probs[:, :self.pixels]
        probs = probs * self.pixels
        probs = torch.clamp(probs, 0, 1)
        return probs

    def forward(self, x):
        x = einops.rearrange(x, "b 1 w h -> b (w h)")
        x = self.qnode(x)
        x = self._post_process(x)
        x = einops.rearrange(x, "b (w h) -> b 1 w h", w=self.width, h=self.height)

        return x

    def __repr__(self):
        return f"QDenseUndirected_old_noise(qdepth={self.qdepth}, wires={self.wires}, add_noise={self.add_noise})"

    def save_name(self) -> str:
        return f"QDenseUndirected_old_noise{self.qdepth}_w{self.width}_h{self.height}"


class QNN_A(nn.Module):
    """Dense variational circuit with angle encoding and dimensionality reduction"""

    def __init__(self, qdepth, shape, add_noise=0, device_type="default.qubit.torch", diff_method="backprop") -> None:
        super().__init__()
        self.qdepth = qdepth
        self.add_noise = add_noise  # 噪声参数
        self.device_type = device_type  # 设备类型参数
        self.diff_method = diff_method  # 微分方法

        if isinstance(shape, int):
            shape = (shape, shape)
        self.width, self.height = shape
        self.pixels = self.width * self.height  # 像素总数
        self.wires = math.ceil(math.log2(self.pixels))  # 量子比特数目
        self.qdev = qml.device(self.device_type, wires=self.wires)

        # 线性降维层，将输入数据降维到量子比特数
        self.linear_down = nn.Linear(self.pixels, self.wires, dtype=torch.double)

        # 定义量子线路的权重形状
        weight_shape = qml.StronglyEntanglingLayers.shape(self.qdepth, self.wires)
        self.weights = torch.nn.Parameter(
            torch.randn(weight_shape, dtype=torch.double) * 0.4, requires_grad=True
        )

        # 定义量子节点
        self.qnode = qml.QNode(
            func=self._circuit,
            device=self.qdev,
            interface="torch",
            diff_method=self.diff_method,
        )

    def _circuit(self, inp):
        """量子线路，使用角度编码和纠缠层"""

        # 角度编码
        qml.AngleEmbedding(
            features=inp, wires=range(self.wires), rotation="Y"
        )

        # 应用强纠缠层
        qml.StronglyEntanglingLayers(weights=self.weights, wires=range(self.wires))

        # 根据 add_noise 参数选择性添加噪声
        for wire in range(self.wires):
            if self.add_noise == 1:
                qml.PhaseDamping(0.05, wires=wire)  # Phase Damping
            elif self.add_noise == 2:
                qml.AmplitudeDamping(0.05, wires=wire)  # Amplitude Damping
            elif self.add_noise == 3:
                qml.DepolarizingChannel(0.02, wires=wire)  # Depolarizing Channel

        # 返回概率测量结果
        return qml.probs(wires=range(self.wires))

    def _post_process(self, probs):
        """后处理：截取前 self.pixels 个概率值并缩放"""
        probs = probs[:, :self.pixels]
        probs = probs * self.pixels
        probs = torch.clamp(probs, 0, 1)
        return probs

    def forward(self, x):
        # Reshape 输入数据并降维
        x = einops.rearrange(x, "b 1 w h -> b (w h)")
        x = self.linear_down(x)  # 线性降维

        # 将降维后的数据传入量子线路
        x = self.qnode(x)

        # 后处理并恢复为原始图像格式
        x = self._post_process(x)
        x = einops.rearrange(x, "b (w h) -> b 1 w h", w=self.width, h=self.height)

        return x

    def __repr__(self):
        return f"QNN_A(qdepth={self.qdepth}, wires={self.wires}, add_noise={self.add_noise})"

    def save_name(self) -> str:
        return f"QNN_A{self.qdepth}_w{self.width}_h{self.height}_noise{self.add_noise}"


class QNN(nn.Module):
    def __init__(self, input_dim, hidden_features, qdepth: int) -> None:
        super().__init__()
        if isinstance(input_dim, str):
            input_dim = eval(input_dim)  # 将字符串解析为表达式，例如 "28 * 28" -> 784
        self.hidden_features = hidden_features  # num of qubits
        self.qdepth = qdepth
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 定义线性层替代 PCA 进行降维
        self.linear_down = nn.Linear(input_dim, hidden_features, dtype=torch.double).to(self.device)

        # 定义线性层恢复维度
        self.linear_up = nn.Linear(hidden_features, input_dim, dtype=torch.double).to(self.device)

        # 量子电路相关参数
        self.qdev = qml.device("lightning.qubit", wires=self.hidden_features)
        weight_shape = qml.StronglyEntanglingLayers.shape(self.qdepth, self.hidden_features)
        self.weights = torch.nn.Parameter(
            torch.randn(weight_shape, dtype=torch.double, requires_grad=True).to(self.device) * 0.4
        )
        self.qnode = qml.QNode(
            func=self._circuit,
            device=self.qdev,
            interface="torch",
            diff_method="parameter-shift"
        )

    def _circuit(self, inputs, weights):
        inputs = inputs.flatten()  # 确保输入是一维
        for j in range(self.hidden_features):
            qml.RZ(inputs[j], wires=j)
        qml.StronglyEntanglingLayers(weights, wires=range(self.hidden_features), imprimitive=qml.ops.CZ)
        res = [qml.expval(qml.PauliZ(i)) for i in range(self.hidden_features)]
        return res  # 返回期望值列表

    def forward(self, x):
        b, c, w, h = x.shape
        x = x.view(b, -1).to(self.device).to(torch.double)

        # 使用线性层进行降维
        x_reduced = self.linear_down(x)

        # Reshape to match hidden_features dimension for quantum processing
        x_reduced = einops.rearrange(x_reduced, "b (c w h) -> b c w h", c=1, w=1, h=self.hidden_features)

        # Process through quantum layer
        x_processed = [
            torch.tensor(self.qnode(x_reduced[i], self.weights), dtype=torch.double).to(self.device)
            for i in range(b)
        ]
        x_reduced = torch.stack(x_processed)

        # Reshape back to flat format and restore dimensions
        x_reduced = x_reduced.view(b, -1)
        x_restored = self.linear_up(x_reduced)
        x_restored = x_restored.view(b, c, w, h)

        return x_restored

    def __repr__(self):
        return f"QNN(qdepth={self.qdepth}, features={self.hidden_features})"

    def save_name(self) -> str:
        return f"QNN_linear_features={self.hidden_features}_qdepth={self.qdepth}"

    def save_model(self, path, loss_values, epochs):
        model_dict = {
            'model_state_dict': self.state_dict(),
            'loss_values': loss_values,
            'epochs': epochs
        }
        torch.save(model_dict, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])


class QNN_noise(nn.Module):
    def __init__(self, input_dim, hidden_features, qdepth: int, add_noise=0, noise_intensity=0.1) -> None:
        super().__init__()
        if isinstance(input_dim, str):
            input_dim = eval(input_dim)  # 将字符串解析为表达式，例如 "28 * 28" -> 784
        self.hidden_features = hidden_features  # num of qubits
        self.qdepth = qdepth
        self.add_noise = add_noise  # 噪声类型的控制
        self.noise_intensity = noise_intensity  # 噪声强度


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 定义线性层替代 PCA 进行降维
        self.linear_down = nn.Linear(input_dim, hidden_features, dtype=torch.double).to(self.device)

        # 定义线性层恢复维度
        self.linear_up = nn.Linear(hidden_features, input_dim, dtype=torch.double).to(self.device)

        # 量子电路相关参数
        self.qdev = qml.device("lightning.qubit", wires=self.hidden_features)
        weight_shape = qml.StronglyEntanglingLayers.shape(self.qdepth, self.hidden_features)
        self.weights = torch.nn.Parameter(
            torch.randn(weight_shape, dtype=torch.double, requires_grad=True).to(self.device) * 0.4
        )
        self.qnode = qml.QNode(
            func=self._circuit,
            device=self.qdev,
            interface="torch",
            diff_method="parameter-shift"
        )

    def _circuit(self, inputs, weights):
        inputs = inputs.flatten()  # 确保输入是一维

        for j in range(self.hidden_features):
            qml.RZ(inputs[j], wires=j)

            # 根据 add_noise 参数选择性添加噪声
            if self.add_noise == 1:
                qml.PhaseDamping(self.noise_intensity, wires=j)  # Phase Damping
            elif self.add_noise == 2:
                qml.AmplitudeDamping(self.noise_intensity, wires=j)  # Amplitude Damping
            elif self.add_noise == 3:
                qml.DepolarizingChannel(self.noise_intensity, wires=j)  # Depolarizing Channel

        qml.StronglyEntanglingLayers(weights, wires=range(self.hidden_features), imprimitive=qml.ops.CZ)
        res = [qml.expval(qml.PauliZ(i)) for i in range(self.hidden_features)]
        return res  # 返回期望值列表

    def forward(self, x):
        b, c, w, h = x.shape
        x = x.view(b, -1).to(self.device).to(torch.double)

        # 使用线性层进行降维
        x_reduced = self.linear_down(x)

        # Reshape to match hidden_features dimension for quantum processing
        x_reduced = einops.rearrange(x_reduced, "b (c w h) -> b c w h", c=1, w=1, h=self.hidden_features)

        # Process through quantum layer
        x_processed = [
            torch.tensor(self.qnode(x_reduced[i], self.weights), dtype=torch.double).to(self.device)
            for i in range(b)
        ]
        x_reduced = torch.stack(x_processed)

        # Reshape back to flat format and restore dimensions
        x_reduced = x_reduced.view(b, -1)
        x_restored = self.linear_up(x_reduced)
        x_restored = x_restored.view(b, c, w, h)

        return x_restored

    def __repr__(self):
        return f"QNN(qdepth={self.qdepth}, features={self.hidden_features}, add_noise={self.add_noise})"

    def save_name(self) -> str:
        return f"QNN_linear_features={self.hidden_features}_qdepth={self.qdepth}_add_noise={self.add_noise}"

    def save_model(self, path, loss_values, epochs):
        model_dict = {
            'model_state_dict': self.state_dict(),
            'loss_values': loss_values,
            'epochs': epochs
        }
        torch.save(model_dict, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])


class differN_noise(nn.Module):
    """Dense variational circuit with batch processing enabled"""

    def __init__(self, shape, spectrum_layer, N, add_noise=0, noise_intensity=0.1) -> None:
        super().__init__()
        if isinstance(shape, int):
            shape = (shape, shape)
        self.spectrum_layer = spectrum_layer  # L
        self.N = N  # N
        self.add_noise = add_noise  # 新增噪声参数
        self.noise_intensity = noise_intensity  # 噪声强度

        self.width, self.height = shape
        self.pixels = self.width * self.height  # 总像素数
        self.wires = math.ceil(math.log2(self.pixels))  # 量子比特数目
        # 使用PCA进行降维
        self.pca = PCA(n_components=self.wires)

        # 量子线路
        self.qdev = qml.device("default.qubit.torch", wires=self.wires)
        # 定义权重的形状： spectrum_layer 表示量子纠缠层数，2 是两个旋转参数，wires 是量子比特数，3 表示每个比特的旋转门数量
        weight_shape = (N, self.spectrum_layer, 2, self.wires, 3)
        # 初始化权重参数
        self.weights = nn.Parameter(
            torch.randn(weight_shape, requires_grad=True) * 0.4
        )

        # 定义量子节点，启用批量处理
        self.qnode = qml.QNode(
            func=self._circuit,
            device=self.qdev,
            interface="torch",
            diff_method="backprop",
        )

    def _circuit(self, inputs, weights):
        # 对每个样本进行角度编码，并在纠缠层应用旋转门
        for i in range(self.spectrum_layer):
            for j in range(self.wires):
                qml.RZ(inputs[:, j], wires=j)  # 对整个批次的数据执行角度编码
            qml.StronglyEntanglingLayers(weights[i], wires=range(self.wires), imprimitive=qml.ops.CZ)

        # 添加噪声
        if self.add_noise == 1:
            for wire in range(self.wires):
                qml.PhaseShift(self.noise_intensity, wires=wire)
        elif self.add_noise == 2:
            for wire in range(self.wires):
                qml.AmplitudeDamping(self.noise_intensity, wires=wire)
        elif self.add_noise == 3:
            for wire in range(self.wires):
                qml.DepolarizingChannel(self.noise_intensity, wires=wire)

        return qml.probs(wires=range(self.wires))

    def _post_process(self, probs):
        # 批量后处理：截取前 self.pixels 个概率值
        probs = probs[:, :self.pixels]
        probs = probs * self.pixels  # 将概率缩放至像素范围
        probs = torch.clamp(probs, 0, 1)  # 限制概率值在 [0, 1] 范围内
        return probs

    def forward(self, x):
        b, c, w, h = x.shape
        # 将输入 reshape 为 [batch_size, pixels] 形状
        x = einops.rearrange(x, "b 1 w h -> b (w h)")

        # 应用PCA降维，将输入维度降至量子比特数
        x = self.pca.fit_transform(x.cpu().detach().numpy())  # PCA降维到量子比特数
        x = torch.tensor(x, dtype=torch.float32).to(next(self.parameters()).device)  # 转回tensor，保证设备一致

        # 通过量子电路批量处理
        for n in range(self.N):
            x = self.qnode(x, self.weights[n])

        # 后处理量子电路输出的概率分布
        probs = self._post_process(x)

        # 将概率重新排列为 [batch_size, 1, width, height] 形状
        x = einops.rearrange(probs, "b (w h) -> b 1 w h", w=self.width, h=self.height)
        return x

    def __repr__(self):
        return f"differN_old_pca={self.spectrum_layer}_N={self.N}_w{self.width}_h{self.height}"

    def save_name(self) -> str:
        return f"differN_old_pca={self.spectrum_layer}_N={self.N}_w{self.width}_h{self.height}_noise{self.add_noise}"


class QIDDM_PL_noise(nn.Module):
    def __init__(self, input_dim, hidden_features, spectrum_layer, N: int, add_noise=0, device_type="lightning.qubit") -> None:
        super().__init__()
        self.hidden_features = hidden_features
        self.spectrum_layer = spectrum_layer
        self.N = N
        self.add_noise = add_noise

        # 使用 PCA 进行降维
        self.pca = PCA(n_components=hidden_features)
        self.linear_up = nn.Linear(hidden_features, input_dim)

        # 动态选择设备类型和 diff_method
        self.device_type = device_type
        self.diff_method = "parameter-shift" if device_type == "lightning.qubit" else "backprop"
        self.qdev = qml.device(device_type, wires=self.hidden_features)

        weight_shape1 = (N, self.spectrum_layer, 2, hidden_features, 3)
        self.weights1 = nn.Parameter(
            torch.randn(weight_shape1, requires_grad=True) * 0.4
        )

        self.qnode = qml.QNode(
            func=self._circuit,
            device=self.qdev,
            interface="torch",
            diff_method=self.diff_method
        )

    def _circuit(self, inputs, weights1):
        inputs = inputs.flatten()  # Ensure inputs are flattened to 1D

        for i in range(self.spectrum_layer):
            for j in range(self.hidden_features):
                qml.RZ(inputs[j], wires=j)

                # 根据 add_noise 参数选择性添加噪声
                if self.add_noise == 1:
                    qml.PhaseDamping(0.03, wires=j)  # Phase Damping
                elif self.add_noise == 2:
                    qml.AmplitudeDamping(0.05, wires=j)  # Amplitude Damping
                elif self.add_noise == 3:
                    qml.DepolarizingChannel(0.02, wires=j)  # Depolarizing Channel

            qml.StronglyEntanglingLayers(weights1[i], wires=range(self.hidden_features), imprimitive=qml.ops.CZ)

        res = [qml.expval(qml.PauliZ(i)) for i in range(self.hidden_features)]
        return res

    # 定义前向传播和后处理
    def forward(self, x):
        b, c, w, h = x.shape
        x = x.view(b, -1)  # b,w*h

        # 使用 PCA 进行降维
        x_reduced = self.pca.fit_transform(x.cpu().numpy())
        x_reduced = torch.tensor(x_reduced).to(self.linear_up.weight.device).to(
            self.linear_up.weight.dtype)

        # Reshape to 2D image-like format
        x_reduced = einops.rearrange(x_reduced, "b (c w h) -> b c w h", c=1, w=1, h=self.hidden_features)

        for n in range(self.N):
            x_reduced = x_reduced.to(self.linear_up.weight.device)  # Ensure x_reduced is on the right device
            x_processed = [self.qnode(x_reduced[i], self.weights1[n]).clone().detach().requires_grad_(True).to(
                self.linear_up.weight.device) for i in range(b)]
            x_reduced = torch.stack(x_processed)  # Stack the results to form the correct shape

        x_reduced = x_reduced.view(b, -1)
        x_reduced = x_reduced.to(self.linear_up.weight.device).to(self.linear_up.weight.dtype)
        x_restored = self.linear_up(x_reduced)
        x_restored = x_restored.view(b, c, w, h)

        return x_restored

    def __repr__(self):
        return f"QIDDM_PL_noise(qlayer={self.spectrum_layer}, features={self.hidden_features}, N={self.N}, add_noise={self.add_noise})"

    def save_name(self) -> str:
        return f"QIDDM_PL_noise={self.hidden_features}_L={self.spectrum_layer}_N={self.N}"

    def save_model(self, path, loss_values, epochs):
        model_dict = {
            'model_state_dict': self.state_dict(),
            'loss_values': loss_values,
            'epochs': epochs
        }
        torch.save(model_dict, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])


class QIDDM_PL_new(nn.Module):
    def __init__(self, input_dim, hidden_features, spectrum_layer, N: int) -> None:
        super().__init__()
        self.hidden_features = hidden_features  # num of qubits
        self.spectrum_layer = spectrum_layer  # L
        self.N = N  # 增加N参数

        # 使用PCA进行降维
        self.pca = PCA(n_components=hidden_features)

        # 定义线性层恢复维度
        self.linear_up = nn.Linear(hidden_features, input_dim)

        # 量子电路相关参数
        self.qdev = qml.device("lightning.qubit", wires=self.hidden_features)
        weight_shape1 = (N, self.spectrum_layer, 2, hidden_features, 3)  # k由2改成3

        self.weights1 = nn.Parameter(
            torch.randn(weight_shape1, requires_grad=True) * 0.4
        )

        self.qnode = qml.QNode(
            func=self._circuit,
            device=self.qdev,
            interface="torch",
            diff_method="parameter-shift"  # parameter-shift, adjoint
        )

    def _circuit(self, inputs, weights1):
        inputs = inputs.flatten()  # Ensure inputs are flattened to 1D

        for i in range(self.spectrum_layer):
            for j in range(self.hidden_features):
                qml.RZ(inputs[j], wires=j)
            qml.StronglyEntanglingLayers(weights1[i], wires=range(self.hidden_features), imprimitive=qml.ops.CZ)

        res = [qml.expval(qml.PauliZ(i)) for i in range(self.hidden_features)]
        return res  # Return the list of expectation values directly

    def forward(self, x):
        b, c, w, h = x.shape
        x = x.view(b, -1)  # b,w*h
        # 使用PCA进行降维
        x_reduced = self.pca.fit_transform(x.cpu().numpy())  # 使用PCA降维，并将tensor转换为numpy进行计算
        x_reduced = torch.tensor(x_reduced).to(self.linear_up.weight.device).to(
            self.linear_up.weight.dtype)  # 将降维后的数据转换为tensor

        # Process through quantum layer N times
        for n in range(self.N):
            x_reduced = x_reduced.to(self.linear_up.weight.device)  # Ensure x_reduced is on the right device
            x_processed = [self.qnode(x_reduced[i], self.weights1[n]).clone().detach().requires_grad_(True).to(
                self.linear_up.weight.device) for i in range(b)]
            x_reduced = torch.stack(x_processed)  # Stack the results to form the correct shape

        # Restore dimensions with the linear layer
        x_restored = self.linear_up(x_reduced).view(b, c, w, h)

        return x_restored

    def __repr__(self):
        return f"QIDDM(qlayer={self.spectrum_layer}, features={self.hidden_features}, N={self.N})"

    def save_name(self) -> str:
        return f"QIDDM_PL_new_q={self.hidden_features}_L={self.spectrum_layer}_N={self.N}"

    def save_model(self, path, loss_values, epochs):
        """
        Save the model state.
        """
        model_dict = {
            'model_state_dict': self.state_dict(),
            'loss_values': loss_values,
            'epochs': epochs
        }
        torch.save(model_dict, path)

    def load_model(self, path):
        """
        Load the model state.
        """
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])


class QIDDM_L(nn.Module):
    def __init__(self, input_dim, hidden_features, spectrum_layer, N: int) -> None:
        super().__init__()
        self.hidden_features = hidden_features  # num of qubits
        self.spectrum_layer = spectrum_layer  # L
        self.N = N  # 增加N参数

        # 定义线性层替代 PCA 进行降维
        self.linear_down = nn.Linear(input_dim, hidden_features)

        # 定义线性层恢复维度
        self.linear_up = nn.Linear(hidden_features, input_dim)

        # 量子电路相关参数
        self.qdev = qml.device("lightning.qubit", wires=self.hidden_features)
        weight_shape1 = (N, self.spectrum_layer, 2, hidden_features, 3)  # k由2改成3

        self.weights1 = nn.Parameter(
            torch.randn(weight_shape1, requires_grad=True) * 0.4
        )

        self.qnode = qml.QNode(
            func=self._circuit,
            device=self.qdev,
            interface="torch",
            diff_method="parameter-shift"  # parameter-shift, adjoint
        )

    def _circuit(self, inputs, weights1):
        inputs = inputs.flatten()  # Ensure inputs are flattened to 1D

        for i in range(self.spectrum_layer):
            for j in range(self.hidden_features):
                qml.RZ(inputs[j], wires=j)
            qml.StronglyEntanglingLayers(weights1[i], wires=range(self.hidden_features), imprimitive=qml.ops.CZ)
            # weights=qw_map.tanh(self.weights)
        res = [qml.expval(qml.PauliZ(i)) for i in range(self.hidden_features)]
        return res  # Return the list of expectation values directly

    def forward(self, x):
        b, c, w, h = x.shape
        x = x.view(b, -1)

        # 确保输入的 dtype 与 linear_down 的权重 dtype 一致，并移动到同一设备
        x = x.to(self.linear_down.weight.device).to(self.linear_down.weight.dtype)

        # 使用线性层进行降维
        x_reduced = self.linear_down(x)

        # Reshape to 2D image-like format with width=1 and height=hidden_features
        x_reduced = einops.rearrange(x_reduced, "b (c w h) -> b c w h", c=1, w=1, h=self.hidden_features)

        # Process through quantum layer N times
        for n in range(self.N):
            x_reduced = x_reduced.to(self.linear_up.weight.device)  # Ensure x_reduced is on the right device
            x_processed = [torch.tensor(self.qnode(x_reduced[i], self.weights1[n])).to(self.linear_up.weight.device) for
                           i in range(b)]
            x_reduced = torch.stack(x_processed)  # Stack the results to form the correct shape

        # Reshape back to flat format
        x_reduced = x_reduced.view(b, -1)

        # 确保 x_reduced 和 linear_up 的权重具有相同的 dtype 并在同一设备上
        x_reduced = x_reduced.to(self.linear_up.weight.device).to(self.linear_up.weight.dtype)

        # 使用线性层恢复维度
        x_restored = self.linear_up(x_reduced)
        # x_restored= torch.clamp(x_restored, 0, 1)
        # Reshape to original image format
        x_restored = x_restored.view(b, c, w, h)

        return x_restored

    def __repr__(self):
        return f"QIDDM(qlayer={self.spectrum_layer}, features={self.hidden_features}, N={self.N})"

    def save_name(self) -> str:
        return f"QIDDM_linear_features={self.hidden_features}_L={self.spectrum_layer}_N={self.N}"

    def save_model(self, path, loss_values, epochs):
        """
        Save the model state.
        """
        model_dict = {
            'model_state_dict': self.state_dict(),
            'loss_values': loss_values,
            'epochs': epochs
        }
        torch.save(model_dict, path)

    def load_model(self, path):
        """
        Load the model state.
        """
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])

