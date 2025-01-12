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
import os


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
                qml.AmplitudeDamping(0.1, wires=wire)
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
        return f"QDenseUndirected_old_noise{self.qdepth}_w{self.width}_h{self.height}_noise{self.add_noise}"


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


import torch
import torch.nn as nn
import pennylane as qml
import einops


class QNN_noise(nn.Module):
    def __init__(self, input_dim, hidden_features, qdepth: int, add_noise=0) -> None:
        super().__init__()
        if isinstance(input_dim, str):
            input_dim = eval(input_dim)  # 将字符串解析为表达式，例如 "28 * 28" -> 784
        self.hidden_features = hidden_features  # num of qubits
        self.qdepth = qdepth
        self.add_noise = add_noise  # 噪声类型的控制

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
                qml.PhaseDamping(0.03, wires=j)  # Phase Damping
            elif self.add_noise == 2:
                qml.AmplitudeDamping(0.05, wires=j)  # Amplitude Damping
            elif self.add_noise == 3:
                qml.DepolarizingChannel(0.02, wires=j)  # Depolarizing Channel

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


class differN_noise(nn.Module):
    """Dense variational circuit with batch processing enabled"""

    def __init__(self, shape, spectrum_layer, N, add_noise=0) -> None:
        super().__init__()
        if isinstance(shape, int):
            shape = (shape, shape)
        self.spectrum_layer = spectrum_layer  # L
        self.N = N  # N
        self.add_noise = add_noise  # 新增噪声参数
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
                # inputs 的每一行是一个展平的样本
                qml.RZ(inputs[:, j], wires=j)  # 对整个批次的数据执行角度编码
            qml.StronglyEntanglingLayers(weights[i], wires=range(self.wires), imprimitive=qml.ops.CZ)

        # 添加噪声（根据 add_noise 参数）
        if self.add_noise == 1:
            for wire in range(self.wires):
                qml.PhaseShift(0.05, wires=wire)
        elif self.add_noise == 2:
            for wire in range(self.wires):
                qml.AmplitudeDamping(0.1, wires=wire)
        elif self.add_noise == 3:
            for wire in range(self.wires):
                qml.DepolarizingChannel(0.02, wires=wire)

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
        # print("After PCA, x:", x)
        x = torch.tensor(x, dtype=torch.float32).to(next(self.parameters()).device)  # 转回tensor，保证设备一致
        # print("Weights values:")
        # for weight in self.weights:
        #     print(weight)

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


class differN_noise_befor(nn.Module):
    """Dense variational circuit with batch processing enabled"""

    def __init__(self, shape, spectrum_layer, N, add_noise=0, device_type="default.qubit.torch") -> None:
        super().__init__()
        if isinstance(shape, int):
            shape = (shape, shape)
        self.spectrum_layer = spectrum_layer  # L
        self.N = N  # N
        self.width, self.height = shape
        self.pixels = self.width * self.height  # 总像素数
        self.wires = math.ceil(math.log2(self.pixels))  # 量子比特数目
        # 使用PCA进行降维
        self.pca = PCA(n_components=self.wires)
        # 加噪方式
        self.add_noise = add_noise

        # 量子线路
        self.qdev = qml.device(device_type, wires=self.wires)
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
                qml.RZ(inputs[:, j], wires=j)
                # 根据 add_noise 参数选择性添加噪声
                if self.add_noise == 1:
                    qml.PhaseDamping(0.03, wires=j)  # Phase Damping
                elif self.add_noise == 2:
                    qml.AmplitudeDamping(0.05, wires=j)  # Amplitude Damping
                elif self.add_noise == 3:
                    qml.DepolarizingChannel(0.02, wires=j)  # Depolarizing Channel

            qml.StronglyEntanglingLayers(weights[i], wires=range(self.wires), imprimitive=qml.ops.CZ)
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
        return f"differN_noise={self.spectrum_layer}_N={self.N}_w{self.width}_h{self.height}"

    def save_name(self) -> str:
        return f"differN_noise={self.spectrum_layer}_N={self.N}_w{self.width}_h{self.height}"


class QIDDM_PL_noise1(nn.Module):
    def __init__(self, input_dim, hidden_features, spectrum_layer, N: int, add_noise=0,
                 device_type="lightning.qubit") -> None:
        super().__init__()
        self.hidden_features = hidden_features
        self.spectrum_layer = spectrum_layer
        self.N = N
        self.add_noise = add_noise

        # 使用 PCA 进行降维
        self.pca = PCA(n_components=hidden_features)
        self.linear_up = nn.Linear(hidden_features, input_dim)

        # 动态选择设备类型和 diff_method
        # self.device_type = "lightning.qubit"
        self.device_type = device_type
        # self.diff_method = "parameter-shift" if device_type == "lightning.qubit" else "backprop"
        self.qdev = qml.device(device_type, wires=self.hidden_features)

        weight_shape1 = (N, self.spectrum_layer, 2, hidden_features, 3)
        self.weights1 = nn.Parameter(
            torch.randn(weight_shape1, requires_grad=True) * 0.4
        )

        self.qnode = qml.QNode(
            func=self._circuit,
            device=self.qdev,
            interface="torch",
            # diff_method=self.diff_method
            diff_method="parameter-shift"
        )

    def _circuit(self, inputs, weights1):
        inputs = inputs.flatten()  # Ensure inputs are flattened to 1D

        for i in range(self.spectrum_layer):
            for j in range(self.hidden_features):
                qml.RY(inputs[j], wires=j)

                # 根据 add_noise 参数选择性添加噪声
                if self.add_noise == 1:
                    qml.PhaseDamping(0.03, wires=j)  # Phase Damping
                elif self.add_noise == 2:
                    qml.AmplitudeDamping(0.05, wires=j)  # Amplitude Damping
                elif self.add_noise == 3:
                    qml.DepolarizingChannel(0.9, wires=j)  # Depolarizing Channel

            qml.StronglyEntanglingLayers(weights1[i], wires=range(self.hidden_features), imprimitive=qml.ops.CZ)

        res = [qml.expval(qml.PauliZ(i)) for i in range(self.hidden_features)]
        # a = qml.probs(wires=range(self.hidden_features))
        # print("prob:", a)
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
        print("After PCA, x:", x_reduced)
        for n in range(self.N):
            x_reduced = x_reduced.to(self.linear_up.weight.device)
            x_processed = [torch.tensor(self.qnode(x_reduced[i], self.weights1[n]), dtype=torch.float64).to(
                self.linear_up.weight.device) for i in range(b)]
            x_reduced = torch.stack(x_processed)
            print("Weights values1:", self.weights1[n])
        # 输出线性层的参数
        print("Linear layer weights (linear_up):", self.linear_up.weight)
        print("Linear layer bias (linear_up):", self.linear_up.bias)

        x_reduced = x_reduced.view(b, -1)
        print("ater circuit:", x_reduced)
        x_reduced = x_reduced.to(self.linear_up.weight.device).to(self.linear_up.weight.dtype)
        x_restored = self.linear_up(x_reduced)
        print("end:", x_restored)
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


# old代表整个batch直接放入，new代表逐个样本
class differN_old_pca(nn.Module):
    """Dense variational circuit with batch processing enabled"""

    def __init__(self, shape, spectrum_layer, N) -> None:
        super().__init__()
        if isinstance(shape, int):
            shape = (shape, shape)
        self.spectrum_layer = spectrum_layer  # L
        self.N = N  # N
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
                # inputs 的每一行是一个展平的样本
                qml.RZ(inputs[:, j], wires=j)  # 对整个批次的数据执行角度编码
            qml.StronglyEntanglingLayers(weights[i], wires=range(self.wires), imprimitive=qml.ops.CZ)
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
        return f"differN_old_pca={self.spectrum_layer}_N={self.N}_w{self.width}_h{self.height}"


# pca+样本逐个处理
class differN_new_pca(nn.Module):
    """Dense variational circuit with individual sample processing"""

    def __init__(self, shape, spectrum_layer, N) -> None:
        super().__init__()
        if isinstance(shape, int):
            shape = (shape, shape)
        self.spectrum_layer = spectrum_layer  # L
        self.N = N  # N
        self.width, self.height = shape
        self.pixels = self.width * self.height  # 总像素数
        self.wires = math.ceil(math.log2(self.pixels))  # 量子比特数目
        # 使用PCA进行降维
        self.pca = PCA(n_components=self.wires)
        # Quantum device setup
        self.qdev = qml.device("default.qubit.torch", wires=self.wires)

        # 定义权重的形状： spectrum_layer 表示量子纠缠层数，2 是两个旋转参数，wires 是量子比特数，3 表示每个比特的旋转门数量
        weight_shape = (N, self.spectrum_layer, 2, self.wires, 3)
        self.weights = nn.Parameter(
            torch.randn(weight_shape, requires_grad=True) * 0.4
        )

        # 定义量子节点
        self.qnode = qml.QNode(
            func=self._circuit,
            device=self.qdev,
            interface="torch",
            diff_method="backprop",
        )

    def _circuit(self, inputs, weights):
        # 对样本进行角度编码，并在纠缠层应用旋转门
        for i in range(self.spectrum_layer):
            for j in range(self.wires):
                qml.RZ(inputs[0, j], wires=j)  # 对单个样本进行角度编码
            qml.StronglyEntanglingLayers(weights[i], wires=range(self.wires), imprimitive=qml.ops.CZ)

        # 获取量子线路的输出概率分布
        probs = qml.probs(wires=range(self.wires))
        return probs

    def _post_process(self, probs):
        # 后处理：截取前 self.pixels 个概率值并缩放
        probs = probs[:self.pixels]
        probs = probs * self.pixels  # 将概率缩放至像素范围
        probs = torch.clamp(probs, 0, 1)  # 限制概率值在 [0, 1] 范围内
        return probs

    def forward(self, x):
        b, c, w, h = x.shape
        x = einops.rearrange(x, "b 1 w h -> b (w h)")  # #####后面加一步pca，把b 1 w h变成b 1 1 wires

        # 应用PCA降维，将输入维度降至量子比特数
        x = self.pca.fit_transform(x.cpu().detach().numpy())  # PCA降维到量子比特数
        x = torch.tensor(x, dtype=torch.float32).to(next(self.parameters()).device)  # 转回tensor，保证设备一致

        # 存储每个样本处理后的输出
        processed_outputs = []

        # 逐个样本通过量子电路处理
        for i in range(b):
            single_sample = x[i].unsqueeze(0)  # 将单个样本形状转换为 (1, wires)

            # 对同一输入执行 N 次量子处理
            for n in range(self.N):
                # print(f"Sample {i + 1}, iteration {n + 1} - Quantum circuit input shape: {single_sample.shape}")
                single_sample = self.qnode(single_sample, self.weights[n])  # 直接传递 self.weights[n]
                single_sample = self._post_process(single_sample)  # 进行后处理
                single_sample = single_sample.unsqueeze(0)  # 确保形状为 (1, wires)
                # print(f"Sample {i + 1}, iteration {n + 1} - Quantum circuit output shape after post-process: {single_sample.shape}")

            # 将后处理后的结果存入列表
            processed_outputs.append(single_sample)

        # 将所有样本的输出堆叠为 (b, k)
        processed_outputs = torch.stack(processed_outputs).squeeze(1)
        # print(f"Processed outputs shape after stack: {processed_outputs.shape}")

        # 将最终的输出重新排列为 [b, 1, width, height]
        x = einops.rearrange(processed_outputs, "b (w h) -> b 1 w h", w=self.width, h=self.height)
        # print(f"Final output shape: {x.shape}")
        return x

    def __repr__(self):
        return f"differN_new_pca={self.spectrum_layer}_N={self.N}_w{self.width}_h{self.height}"

    def save_name(self) -> str:
        return f"differN_new_pca={self.spectrum_layer}_N={self.N}_w{self.width}_h{self.height}"


class differN_new_conv(nn.Module):
    """Dense variational circuit with individual sample processing"""

    def __init__(self, shape, spectrum_layer, N) -> None:
        super().__init__()
        if isinstance(shape, int):
            shape = (shape, shape)
        self.spectrum_layer = spectrum_layer  # L
        self.N = N  # N
        self.width, self.height = shape
        self.pixels = self.width * self.height  # 总像素数
        self.wires = math.ceil(math.log2(self.pixels))  # 量子比特数目
        # 使用PCA进行降维
        self.pca = PCA(n_components=self.wires)
        # 使用卷积层进行降维
        self.conv_layer = nn.Conv2d(in_channels=1, out_channels=self.wires, kernel_size=3, stride=2, padding=1)

        # Quantum device setup
        self.qdev = qml.device("default.qubit.torch", wires=self.wires)

        # 定义权重的形状： spectrum_layer 表示量子纠缠层数，2 是两个旋转参数，wires 是量子比特数，3 表示每个比特的旋转门数量
        weight_shape = (N, self.spectrum_layer, 2, self.wires, 3)
        self.weights = nn.Parameter(
            torch.randn(weight_shape, requires_grad=True) * 0.4
        )

        # 定义量子节点
        self.qnode = qml.QNode(
            func=self._circuit,
            device=self.qdev,
            interface="torch",
            diff_method="backprop",
        )

    def _circuit(self, inputs, weights):
        # 对样本进行角度编码，并在纠缠层应用旋转门
        for i in range(self.spectrum_layer):
            for j in range(self.wires):
                qml.RZ(inputs[0, j], wires=j)  # 对单个样本进行角度编码
            qml.StronglyEntanglingLayers(weights[i], wires=range(self.wires), imprimitive=qml.ops.CZ)

        # 获取量子线路的输出概率分布
        probs = qml.probs(wires=range(self.wires))
        return probs

    def _post_process(self, probs):
        # 后处理：截取前 self.pixels 个概率值并缩放
        probs = probs[:self.pixels]
        probs = probs * self.pixels  # 将概率缩放至像素范围
        probs = torch.clamp(probs, 0, 1)  # 限制概率值在 [0, 1] 范围内
        return probs

    def forward(self, x):
        b, c, w, h = x.shape

        # x = einops.rearrange(x, "b 1 w h -> b (w h)")
        # 应用PCA降维，将输入维度降至量子比特数
        # x = self.pca.fit_transform(x.cpu().detach().numpy())  # PCA降维到量子比特数
        # x = torch.tensor(x, dtype=torch.float32).to(next(self.parameters()).device)  # 转回tensor，保证设备一致

        # 通过卷积层进行降维
        x = self.conv_layer(x)  # 将输入降维到 (b, wires, new_w, new_h)
        # 将输出展平为 (b, wires)
        x = x.view(b, self.wires, -1).mean(dim=2)  # 对空间维度求平均，得到 (b, wires)

        print(f"shape after down : {x.shape}")

        # 存储每个样本处理后的输出
        processed_outputs = []

        # 逐个样本通过量子电路处理
        for i in range(b):
            single_sample = x[i].unsqueeze(0)  # 将单个样本形状转换为 (1, wires)
            # 对同一输入执行 N 次量子处理
            for n in range(self.N):
                # print(f"Sample {i + 1}, iteration {n + 1} - Quantum circuit input shape: {single_sample.shape}")
                single_sample = self.qnode(single_sample, self.weights[n])  # 直接传递 self.weights[n]
                single_sample = self._post_process(single_sample)  # 进行后处理
                single_sample = single_sample.unsqueeze(0)  # 确保形状为 (1, wires)
                # print(f"Sample {i + 1}, iteration {n + 1} - Quantum circuit output shape after post-process: {single_sample.shape}")

            # 将后处理后的结果存入列表
            processed_outputs.append(single_sample)

        # 将所有样本的输出堆叠为 (b, k)
        processed_outputs = torch.stack(processed_outputs).squeeze(1)
        # print(f"Processed outputs shape after stack: {processed_outputs.shape}")

        # 将最终的输出重新排列为 [b, 1, width, height]
        x = einops.rearrange(processed_outputs, "b (w h) -> b 1 w h", w=self.width, h=self.height)
        # print(f"Final output shape: {x.shape}")
        return x

    def __repr__(self):
        return f"differN_new_conv={self.spectrum_layer}_N={self.N}_w{self.width}_h{self.height}"

    def save_name(self) -> str:
        return f"differN_new_conv={self.spectrum_layer}_N={self.N}_w{self.width}_h{self.height}"


# differn old conv
class differN_old_conv(nn.Module):

    def __init__(self, shape, spectrum_layer, N) -> None:
        super().__init__()
        if isinstance(shape, int):
            shape = (shape, shape)
        self.spectrum_layer = spectrum_layer  # L
        self.N = N  # N
        self.width, self.height = shape
        self.pixels = self.width * self.height  # 总像素数
        self.wires = math.ceil(math.log2(self.pixels))  # 量子比特数目
        # 使用PCA进行降维
        self.pca = PCA(n_components=self.wires)
        # 使用卷积层进行降维
        self.conv_layer = nn.Conv2d(in_channels=1, out_channels=self.wires, kernel_size=3, stride=2, padding=1)

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
                # inputs 的每一行是一个展平的样本
                qml.RZ(inputs[:, j], wires=j)  # 对整个批次的数据执行角度编码
            qml.StronglyEntanglingLayers(weights[i], wires=range(self.wires), imprimitive=qml.ops.CZ)
        return qml.probs(wires=range(self.wires))

    def _post_process(self, probs):
        # 批量后处理：截取前 self.pixels 个概率值
        probs = probs[:, :self.pixels]
        probs = probs * self.pixels  # 将概率缩放至像素范围
        probs = torch.clamp(probs, 0, 1)  # 限制概率值在 [0, 1] 范围内
        return probs

    def forward(self, x):
        b, c, w, h = x.shape

        # 通过卷积层进行降维
        x = self.conv_layer(x)  # 将输入降维到 (b, wires, new_w, new_h)
        # 将输出展平为 (b, wires)
        x = x.view(b, self.wires, -1).mean(dim=2)  # 对空间维度求平均，得到 (b, wires)

        # 通过量子电路批量处理
        for n in range(self.N):
            x = self.qnode(x, self.weights[n])

        # 后处理量子电路输出的概率分布
        probs = self._post_process(x)

        # 将概率重新排列为 [batch_size, 1, width, height] 形状
        x = einops.rearrange(probs, "b (w h) -> b 1 w h", w=self.width, h=self.height)
        return x

    def __repr__(self):
        return f"differN_old_conv={self.spectrum_layer}_N={self.N}_w{self.width}_h{self.height}"

    def save_name(self) -> str:
        return f"differN_old_conv={self.spectrum_layer}_N={self.N}_w{self.width}_h{self.height}"


class QIDDM_CL_new(nn.Module):
    def __init__(self, input_dim, hidden_features, spectrum_layer, N: int) -> None:
        super().__init__()
        self.hidden_features = hidden_features  # num of qubits
        self.spectrum_layer = spectrum_layer  # L
        self.N = N  # 增加N参数

        # 使用卷积层进行降维,参数可调
        self.conv_layer = nn.Conv2d(in_channels=1, out_channels=self.hidden_features, kernel_size=3, stride=2,
                                    padding=1)

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
        # 通过卷积层进行降维(b, wires, new_w, new_h)
        x = self.conv_layer(x)
        # 将输出展平为 (b, wires)
        x_reduced = x.view(b, self.hidden_features, -1).mean(dim=2)
        x_reduced = einops.rearrange(x_reduced, "b (c w h) -> b c w h", c=1, w=1, h=self.hidden_features)

        # Process through quantum layer N times
        for n in range(self.N):
            x_reduced = x_reduced.to(self.linear_up.weight.device)  # Ensure x_reduced is on the right device
            x_processed = [self.qnode(x_reduced[i], self.weights1[n]).clone().detach().requires_grad_(True).to(
                self.linear_up.weight.device) for i in range(b)]
            x_reduced = torch.stack(x_processed)  # Stack the results to form the correct shape

        # Reshape back to flat format
        x_reduced = x_reduced.view(b, -1)
        x_reduced = x_reduced.to(self.linear_up.weight.device).to(self.linear_up.weight.dtype)
        # 线性层恢复维度
        x_restored = self.linear_up(x_reduced)
        # Reshape to original image format
        x_restored = x_restored.view(b, c, w, h)

        return x_restored

    def __repr__(self):
        return f"QIDDM(qlayer={self.spectrum_layer}, features={self.hidden_features}, N={self.N})"

    def save_name(self) -> str:
        return f"QIDDM_CL_new_q={self.hidden_features}_L={self.spectrum_layer}_N={self.N}"

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


class QIDDM_CL_old(nn.Module):
    def __init__(self, input_dim, hidden_features, spectrum_layer, N: int) -> None:
        super().__init__()
        self.hidden_features = hidden_features  # num of qubits
        self.spectrum_layer = spectrum_layer  # L
        self.N = N  # 增加N参数

        # 使用卷积层进行降维,参数可调
        self.conv_layer = nn.Conv2d(in_channels=1, out_channels=self.hidden_features, kernel_size=3, stride=2,
                                    padding=1)

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
        # 通过卷积层进行降维
        x = self.conv_layer(x)
        # 将输出展平为 (b, wires)
        x_reduced = x.view(b, self.hidden_features, -1).mean(dim=2)
        print(f"shape after down:{x_reduced.shape}")

        # Process through quantum layer N times
        for n in range(self.N):
            x_reduced = self.qnode(x_reduced, self.weights1[n])  # 直接传递整个 batch

        print(f"shape after qnode:{x_reduced.shape}")
        # Reshape back to flat format
        x_reduced = x_reduced.view(b, -1)
        x_reduced = x_reduced.to(self.linear_up.weight.device).to(self.linear_up.weight.dtype)
        print(f"shape after reshape:{x_reduced.shape}")
        # 线性层恢复维度
        x_restored = self.linear_up(x_reduced)
        print(f"shape after linear up:{x_restored.shape}")
        # Reshape to original image format
        x_restored = x_restored.view(b, c, w, h)

        return x_restored

    def __repr__(self):
        return f"QIDDM(qlayer={self.spectrum_layer}, features={self.hidden_features}, N={self.N})"

    def save_name(self) -> str:
        return f"QIDDM_CL_old_q={self.hidden_features}_L={self.spectrum_layer}_N={self.N}"


class QIDDM_PL_old(nn.Module):
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

        # Reshape to 2D image-like format with width=1 and height=hidden_features
        x_reduced = einops.rearrange(x_reduced, "b (c w h) -> b c w h", c=1, w=1, h=self.hidden_features)

        # Process through quantum layer N times
        for n in range(self.N):
            x_reduced = x_reduced.to(self.linear_up.weight.device)  # Ensure x_reduced is on the right device
            x_processed = self.qnode(x_reduced, self.weights1[n])  # 直接传递整个 batch
            x_reduced = torch.stack(x_processed)  # Stack the results to form the correct shape

        # Reshape back to flat format
        x_reduced = x_reduced.view(b, -1)

        # 确保 x_reduced 和 linear_up 的权重具有相同的 dtype 并在同一设备上
        x_reduced = x_reduced.to(self.linear_up.weight.device).to(self.linear_up.weight.dtype)

        # 使用线性层恢复维度
        x_restored = self.linear_up(x_reduced)

        # Reshape to original image format
        x_restored = x_restored.view(b, c, w, h)

        return x_restored

    def __repr__(self):
        return f"QIDDM(qlayer={self.spectrum_layer}, features={self.hidden_features}, N={self.N})"

    def save_name(self) -> str:
        return f"QIDDM_PL_old_q={self.hidden_features}_L={self.spectrum_layer}_N={self.N}"

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


class QIDDM_PL(nn.Module):
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

        # 使用 PCA 进行降维
        x_reduced = self.pca.fit_transform(x.cpu().numpy())  # 使用 PCA 降维
        x_reduced = torch.tensor(x_reduced).to(self.linear_up.weight.device).to(
            self.linear_up.weight.dtype)  # 转换为 tensor

        # Reshape to 2D image-like format with width=1 and height=hidden_features
        x_reduced = einops.rearrange(x_reduced, "b (c w h) -> b c w h", c=1, w=1, h=self.hidden_features)

        # Process through quantum layer N times
        for n in range(self.N):
            x_reduced = x_reduced.to(self.linear_up.weight.device)  # Ensure x_reduced is on the right device

            # 调用量子电路并将结果转换为 tensor
            x_processed = [torch.tensor(self.qnode(x_reduced[i], self.weights1[n]), dtype=torch.float64).to(
                self.linear_up.weight.device) for i in range(b)]

            # 堆叠结果以形成正确的形状
            x_reduced = torch.stack(x_processed)  # Stack the results to form the correct shape

        # Reshape back to flat format
        x_reduced = x_reduced.view(b, -1)

        # 确保 x_reduced 和 linear_up 的权重具有相同的 dtype 并在同一设备上
        x_reduced = x_reduced.to(self.linear_up.weight.device).to(self.linear_up.weight.dtype)

        # 使用线性层恢复维度
        x_restored = self.linear_up(x_reduced)

        # Reshape to original image format
        x_restored = x_restored.view(b, c, w, h)

        return x_restored

    def __repr__(self):
        return f"QIDDM_PL(qlayer={self.spectrum_layer}, features={self.hidden_features}, N={self.N})"

    def save_name(self) -> str:
        return f"QIDDM_PL={self.hidden_features}_L={self.spectrum_layer}_N={self.N}"

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


class QIDDM_PL_noise(nn.Module):
    def __init__(self, input_dim, hidden_features, spectrum_layer, N: int, add_noise=0,
                 device_type="lightning.qubit") -> None:
        super().__init__()
        self.hidden_features = hidden_features
        self.spectrum_layer = spectrum_layer
        self.N = N
        self.add_noise = add_noise

        # 使用 PCA 进行降维
        self.pca = PCA(n_components=hidden_features)
        self.linear_up = nn.Linear(hidden_features, input_dim)

        # 动态选择设备类型和 diff_method
        # self.device_type = "lightning.qubit"
        self.device_type = device_type
        # self.diff_method = "parameter-shift" if device_type == "lightning.qubit" else "backprop"
        self.qdev = qml.device(device_type, wires=self.hidden_features)

        weight_shape1 = (N, self.spectrum_layer, 2, hidden_features, 3)
        self.weights1 = nn.Parameter(
            torch.randn(weight_shape1, requires_grad=True) * 0.4
        )

        self.qnode = qml.QNode(
            func=self._circuit,
            device=self.qdev,
            interface="torch",
            # diff_method=self.diff_method
            diff_method="parameter-shift"
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
                    qml.DepolarizingChannel(0.9, wires=j)  # Depolarizing Channel

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
        # print("After PCA, x:", x)

        for n in range(self.N):
            x_reduced = x_reduced.to(self.linear_up.weight.device)
            x_processed = [torch.tensor(self.qnode(x_reduced[i], self.weights1[n]), dtype=torch.float64).to(
                self.linear_up.weight.device) for i in range(b)]
            x_reduced = torch.stack(x_processed)
        # print("Weights values1:", weights1)
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


class QIDDM_LL_relu_noise(nn.Module):
    def __init__(self, input_dim, hidden_features, spectrum_layer, N: int, add_noise=0,
                 device_type="lightning.qubit") -> None:
        super().__init__()
        self.hidden_features = hidden_features
        self.spectrum_layer = spectrum_layer
        self.N = N
        self.add_noise = add_noise

        # 使用线性层进行降维
        self.linear_down = nn.Linear(input_dim, hidden_features)  # 线性降维层
        self.linear_up = nn.Linear(hidden_features, input_dim)

        # 动态选择设备类型
        self.device_type = device_type
        self.qdev = qml.device(device_type, wires=self.hidden_features)

        # 初始化权重
        weight_shape1 = (N, self.spectrum_layer, 2, hidden_features, 3)
        self.weights1 = nn.Parameter(
            torch.randn(weight_shape1, requires_grad=True) * 0.4
        )

        # 定义量子节点
        self.qnode = qml.QNode(
            func=self._circuit,
            device=self.qdev,
            interface="torch",
            diff_method="parameter-shift"
        )
        # ReLU 激活函数
        self.relu = nn.ReLU()

    def _circuit(self, inputs, weights1):
        inputs = inputs.flatten()  # 确保输入为一维

        for i in range(self.spectrum_layer):
            for j in range(self.hidden_features):
                qml.RZ(inputs[j], wires=j)

                # 根据 add_noise 参数选择性添加噪声
                if self.add_noise == 1:
                    qml.PhaseDamping(0.03, wires=j)  # Phase Damping
                elif self.add_noise == 2:
                    qml.AmplitudeDamping(0.05, wires=j)  # Amplitude Damping
                elif self.add_noise == 3:
                    qml.DepolarizingChannel(0.9, wires=j)  # Depolarizing Channel

            qml.StronglyEntanglingLayers(weights1[i], wires=range(self.hidden_features), imprimitive=qml.ops.CZ)

        res = [qml.expval(qml.PauliZ(i)) for i in range(self.hidden_features)]
        return res

    # 定义前向传播
    def forward(self, x):
        b, c, w, h = x.shape
        x = x.view(b, -1)  # 展开为 b, c*w*h

        # 使用线性层进行降维
        x_reduced = self.linear_down(x)

        # Reshape to 2D image-like format
        x_reduced = einops.rearrange(x_reduced, "b (c w h) -> b c w h", c=1, w=1, h=self.hidden_features)

        for n in range(self.N):
            x_reduced = x_reduced.to(self.linear_up.weight.device)
            x_processed = [torch.tensor(self.qnode(x_reduced[i], self.weights1[n]), dtype=torch.float64).to(
                self.linear_up.weight.device) for i in range(b)]
            x_reduced = torch.stack(x_processed)

        # ReLU 激活函数
        self.relu = nn.ReLU()
        x_reduced = x_reduced.view(b, -1)
        x_reduced = x_reduced.to(self.linear_up.weight.device).to(self.linear_up.weight.dtype)
        x_restored = self.linear_up(x_reduced)
        x_restored = x_restored.view(b, c, w, h)

        return x_restored

    def __repr__(self):
        return f"QIDDM_LL_noise(qlayer={self.spectrum_layer}, features={self.hidden_features}, N={self.N}, add_noise={self.add_noise})"

    def save_name(self) -> str:
        return f"QIDDM_LL_noise={self.hidden_features}_L={self.spectrum_layer}_N={self.N}"

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


class QIDDM_LL_noise(nn.Module):
    def __init__(self, input_dim, hidden_features, spectrum_layer, N: int, add_noise=0,
                 device_type="lightning.qubit") -> None:
        super().__init__()
        self.hidden_features = hidden_features
        self.spectrum_layer = spectrum_layer
        self.N = N
        self.add_noise = add_noise

        # 使用 PCA 进行降维
        self.linear_down = nn.Linear(input_dim, hidden_features)
        self.linear_up = nn.Linear(hidden_features, input_dim)

        # 动态选择设备类型和 diff_method
        # self.device_type = "lightning.qubit"
        self.device_type = device_type
        # self.diff_method = "parameter-shift" if device_type == "lightning.qubit" else "backprop"
        self.qdev = qml.device(device_type, wires=self.hidden_features)

        weight_shape1 = (N, self.spectrum_layer, 2, hidden_features, 3)
        self.weights1 = nn.Parameter(
            torch.randn(weight_shape1, requires_grad=True) * 0.4
        )

        self.qnode = qml.QNode(
            func=self._circuit,
            device=self.qdev,
            interface="torch",
            # diff_method=self.diff_method
            diff_method="parameter-shift"
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
                    qml.DepolarizingChannel(0.9, wires=j)  # Depolarizing Channel

            qml.StronglyEntanglingLayers(weights1[i], wires=range(self.hidden_features), imprimitive=qml.ops.CZ)

        res = [qml.expval(qml.PauliZ(i)) for i in range(self.hidden_features)]
        return res

    # 定义前向传播和后处理
    def forward(self, x):
        b, c, w, h = x.shape
        x = x.view(b, -1)  # b,w*h

        # 降维
        x_reduced = self.linear_down(x)

        # Reshape to 2D image-like format
        x_reduced = einops.rearrange(x_reduced, "b (c w h) -> b c w h", c=1, w=1, h=self.hidden_features)
        # print("After PCA, x:", x)

        for n in range(self.N):
            x_reduced = x_reduced.to(self.linear_up.weight.device)
            x_processed = [torch.tensor(self.qnode(x_reduced[i], self.weights1[n]), dtype=torch.float64).to(
                self.linear_up.weight.device) for i in range(b)]
            x_reduced = torch.stack(x_processed)
        # print("Weights values1:", weights1)
        x_reduced = x_reduced.view(b, -1)
        x_reduced = x_reduced.to(self.linear_up.weight.device).to(self.linear_up.weight.dtype)
        x_restored = self.linear_up(x_reduced)
        x_restored = x_restored.view(b, c, w, h)

        return x_restored

    def __repr__(self):
        return f"QIDDM_LL_noise(qlayer={self.spectrum_layer}, features={self.hidden_features}, N={self.N}, add_noise={self.add_noise})"

    def save_name(self) -> str:
        return f"QIDDM_LL_noise={self.hidden_features}_L={self.spectrum_layer}_N={self.N}"

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


class QIDDM_PP_noise(nn.Module):
    def __init__(self, input_dim, hidden_features, spectrum_layer, N: int, add_noise=0,
                 device_type="lightning.qubit") -> None:
        super().__init__()
        self.hidden_features = hidden_features
        self.spectrum_layer = spectrum_layer
        self.N = N
        self.add_noise = add_noise

        # 使用 PCA 进行降维
        self.pca = PCA(n_components=hidden_features)

        # 动态选择设备类型和 diff_method
        self.device_type = device_type
        self.qdev = qml.device(device_type, wires=self.hidden_features)

        weight_shape1 = (N, self.spectrum_layer, 2, hidden_features, 3)
        self.weights1 = nn.Parameter(
            torch.randn(weight_shape1, requires_grad=True) * 0.4
        )

        self.qnode = qml.QNode(
            func=self._circuit,
            device=self.qdev,
            interface="torch",
            diff_method="parameter-shift"
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
                    qml.DepolarizingChannel(0.9, wires=j)  # Depolarizing Channel

            qml.StronglyEntanglingLayers(weights1[i], wires=range(self.hidden_features), imprimitive=qml.ops.CZ)

        res = [qml.expval(qml.PauliZ(i)) for i in range(self.hidden_features)]
        return res

    # 定义前向传播和后处理
    def forward(self, x):
        b, c, w, h = x.shape
        x = x.view(b, -1)  # b,w*h

        # 使用 PCA 进行降维
        x_reduced = self.pca.fit_transform(x.cpu().numpy())
        x_reduced = torch.tensor(x_reduced).to(self.weights1.device).to(self.weights1.dtype)

        # Reshape to 2D image-like format
        x_reduced = einops.rearrange(x_reduced, "b (c w h) -> b c w h", c=1, w=1, h=self.hidden_features)

        for n in range(self.N):
            x_reduced = x_reduced.to(self.weights1.device)
            x_processed = [torch.tensor(self.qnode(x_reduced[i], self.weights1[n]), dtype=torch.float64).to(
                self.weights1.device) for i in range(b)]
            x_reduced = torch.stack(x_processed)

        # 使用 PCA 恢复原始维度
        x_reduced = x_reduced.view(b, -1).cpu().numpy()
        x_restored = self.pca.inverse_transform(x_reduced)
        x_restored = torch.tensor(x_restored, device=x.device, dtype=x.dtype, requires_grad=True)
        x_restored = x_restored.view(b, c, w, h)

        return x_restored

    def __repr__(self):
        return f"QIDDM_PP_noise(qlayer={self.spectrum_layer}, features={self.hidden_features}, N={self.N}, add_noise={self.add_noise})"

    def save_name(self) -> str:
        return f"QIDDM_PP_noise={self.hidden_features}_L={self.spectrum_layer}_N={self.N}"

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


class QIDDM_PP_old(nn.Module):
    def __init__(self, input_dim, hidden_features, spectrum_layer, N: int) -> None:
        super().__init__()
        self.hidden_features = hidden_features  # Number of qubits
        self.spectrum_layer = spectrum_layer  # L (spectrum layer depth)
        self.input_dim = input_dim  # Input dimension
        self.N = N  # N (number of quantum processing iterations)

        # Initialize the PCA, but don't fit yet
        self.pca = None

        # Batch normalization layer after PCA
        self.batch_norm = nn.BatchNorm1d(2 * hidden_features)

        # Linear layer to further reduce dimension after BatchNorm
        self.linear_down = nn.Linear(2 * hidden_features, hidden_features)

        # Linear layer to increase dimension before inverse PCA
        self.linear_up = nn.Linear(hidden_features, 2 * hidden_features)

        # Quantum device setup
        self.qdev = qml.device("lightning.qubit", wires=self.hidden_features)
        weight_shape1 = (N, self.spectrum_layer, 2, hidden_features, 3)  # Adjust weight shape

        # Initialize quantum weights
        self.weights1 = nn.Parameter(
            torch.randn(weight_shape1, requires_grad=True) * 0.4
        )

        self.qnode = qml.QNode(
            func=self._circuit,
            device=self.qdev,
            interface="torch",
            diff_method="parameter-shift"
        )

    def _circuit(self, inputs, weights1):
        inputs = inputs.flatten()  # Flatten inputs to 1D

        for i in range(self.spectrum_layer):
            for j in range(self.hidden_features):
                qml.RZ(inputs[j], wires=j)  # Apply RZ rotation to each qubit
            qml.StronglyEntanglingLayers(weights1[i], wires=range(self.hidden_features), imprimitive=qml.ops.CZ)

        res = [qml.expval(qml.PauliZ(i)) for i in range(self.hidden_features)]  # Get expectation values
        return res

    def forward(self, x):
        b, c, w, h = x.shape  # 获取批次大小、通道数、宽度和高度
        x = x.view(b, -1)  # 将输入展平成 2D 张量

        # 如果 PCA 尚未初始化，则初始化并进行拟合，降维到 2 * hidden_features
        if self.pca is None:
            self.pca = PCA(n_components=2 * self.hidden_features)
            self.pca.fit(x.detach().cpu().numpy())

        # 使用 PCA 降维到 2 * hidden_features
        x_reduced = self.pca.transform(x.detach().cpu().numpy())
        x_reduced = torch.tensor(x_reduced, device=x.device, dtype=x.dtype, requires_grad=True)

        # 通过 BatchNorm 对 PCA 的输出进行标准化
        x_reduced = self.batch_norm(x_reduced)

        # 通过线性层将维度从 2 * hidden_features 降到 hidden_features
        x_reduced = self.linear_down(x_reduced)

        # 重新调整为 2D 形状
        x_reduced = einops.rearrange(x_reduced, "b (c w h) -> b c w h", c=1, w=1, h=self.hidden_features)

        # 通过量子层处理 N 次
        for n in range(self.N):
            x_processed = [torch.tensor(self.qnode(x_reduced[i], self.weights1[n]), device=x.device, dtype=x.dtype) for
                           i in range(b)]
            x_reduced = torch.stack(x_processed)  # 确保每个元素都是张量并堆叠它们

        # 将量子层输出经过线性层升维到 2 * hidden_features
        x_reduced = self.linear_up(x_reduced)

        # 将结果展平成 2 * hidden_features
        x_reduced = x_reduced.view(b, -1)

        # 使用 PCA 的逆变换恢复原始维度
        x_restored = self.pca.inverse_transform(x_reduced.detach().cpu().numpy())
        x_restored = torch.tensor(x_restored, device=x.device, dtype=x.dtype, requires_grad=True)

        # 恢复到原始图像格式
        x_restored = x_restored.view(b, c, w, h)

        return x_restored

    def __repr__(self):
        return f"QIDDM_PP(qlayer={self.spectrum_layer}, features={self.hidden_features}, N={self.N})"

    def save_name(self) -> str:
        return f"QIDDM_PP_features={self.hidden_features}_L={self.spectrum_layer}_N={self.N}"

    def save_model(self, path):
        """
        Save the model state and PCA state if available.
        """
        model_dict = {
            'model_state_dict': self.state_dict(),
        }
        if self.pca is not None:
            model_dict['pca_state'] = pickle.dumps(self.pca)
        torch.save(model_dict, path)

    def load_model(self, path):
        """
        Load the model state and PCA state if available.
        """
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        if 'pca_state' in checkpoint:
            self.pca = pickle.loads(checkpoint['pca_state'])


class QIDDM_LL_old(nn.Module):
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


class QIDDM_bias_false(nn.Module):
    def __init__(self, input_dim, hidden_features, spectrum_layer, N: int) -> None:
        super().__init__()
        self.hidden_features = hidden_features  # num of qubits
        self.spectrum_layer = spectrum_layer  # L
        self.N = N  # 增加N参数

        # 定义线性层替代 PCA 进行降维
        self.linear_down = nn.Linear(input_dim, hidden_features, bias=False)

        # 定义线性层恢复维度
        self.linear_up = nn.Linear(hidden_features, input_dim, bias=False)

        # 量子电路相关参数
        self.qdev = qml.device("lightning.qubit", wires=self.hidden_features)
        # self.qdev = qml.device("default.qubit.jax", wires=self.hidden_features)
        # self.qdev = qml.device("default.qubit.torch", wires=self.hidden_features)
        weight_shape1 = (N, self.spectrum_layer, 3, hidden_features, 3)  # k由2改成3

        self.weights1 = nn.Parameter(
            torch.randn(weight_shape1, requires_grad=True) * 0.4
        )

        # self.qnode = qml.QNode(
        #     func=self._circuit,
        #     device=self.qdev,
        #     interface="torch",
        #     diff_method="backprop",
        # )
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


class QIDDM_L_B(nn.Module):
    def __init__(self, input_dim, hidden_features, spectrum_layer, N: int) -> None:
        super().__init__()
        self.hidden_features = hidden_features  # num of qubits
        self.spectrum_layer = spectrum_layer  # L
        self.N = N  # 增加N参数

        # 定义线性层替代 PCA 进行降维
        self.linear_down = nn.Linear(input_dim, hidden_features)

        # 定义 BatchNorm 层用于每次量子线路前的输出
        self.batchnorm = nn.BatchNorm1d(hidden_features)

        # 定义线性层恢复维度
        self.linear_up = nn.Linear(hidden_features, input_dim)

        # 量子电路相关参数
        # self.qdev = qml.device("default.qubit.torch", wires=self.hidden_features)
        # self.qdev = qml.device("lightning.qubit", wires=self.hidden_features, use_cuda=True)
        self.qdev = qml.device("default.qubit.jax", wires=self.hidden_features)

        weight_shape1 = (N, self.spectrum_layer, 3, hidden_features, 3)  # k由2改成3

        self.weights1 = nn.Parameter(
            torch.randn(weight_shape1, requires_grad=True) * 0.4
        )

        self.qnode = qml.QNode(
            func=self._circuit,
            device=self.qdev,
            interface="torch",
            diff_method="backprop",
        )

    def _circuit(self, inputs, weights1):
        inputs = inputs.flatten()  # Ensure inputs are flattened to 1D

        for i in range(self.spectrum_layer):
            for j in range(self.hidden_features):
                qml.RZ(inputs[j], wires=j)
            qml.StronglyEntanglingLayers(weights1[i], wires=range(self.hidden_features), imprimitive=qml.ops.CZ)

        res = [qml.expval(qml.PauliZ(i)) for i in range(self.hidden_features)]
        return res

    def forward(self, x):
        b, c, w, h = x.shape
        x = x.view(b, -1)  # Flatten the input

        # 确保输入的 dtype 与 linear_down 的权重 dtype 一致
        x = x.to(self.linear_down.weight.dtype)

        # 使用线性层进行降维
        x_reduced = self.linear_down(x)

        # Process through quantum layer N times, applying BatchNorm before each quantum layer
        for n in range(self.N):
            # Apply BatchNorm and ensure consistent dtype
            x_reduced = x_reduced.to(self.batchnorm.weight.dtype)  # Ensure the dtype matches BatchNorm parameters
            x_reduced = self.batchnorm(x_reduced)
            x_reduced = x_reduced.to(self.weights1.dtype)  # Ensure dtype consistency for quantum circuit

            # Pass through quantum circuit
            x_processed = torch.stack([self.qnode(x_reduced[i], self.weights1[n]) for i in range(b)])
            x_reduced = x_processed  # Update x_reduced with the output from quantum circuit

        # Reshape back to flat format
        x_reduced = x_reduced.view(b, -1)

        # 确保 x_reduced 和 linear_up 的权重具有相同的 dtype
        x_reduced = x_reduced.to(self.linear_up.weight.dtype)

        # 使用线性层恢复维度
        x_restored = self.linear_up(x_reduced)

        # Reshape to original image format
        x_restored = x_restored.view(b, c, w, h)

        return x_restored

    def __repr__(self):
        return f"QIDDM_L_B(qlayer={self.spectrum_layer}, features={self.hidden_features}, N={self.N})"

    def save_name(self) -> str:
        return f"QIDDM_linear_batch_features={self.hidden_features}_L={self.spectrum_layer}_N={self.N}"

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


class QIDDM_A_differN_basePL(nn.Module):
    def __init__(self, input_dim, spectrum_layer, N: int) -> None:
        super().__init__()
        self.spectrum_layer = spectrum_layer  # L
        self.width = input_dim
        self.height = input_dim
        self.pixels = self.width * self.height  # 总像素数
        self.hidden_features = math.ceil(math.log2(self.pixels))  # num of qubits
        self.N = N  # 增加N参数

        # 使用PCA进行降维
        self.pca = PCA(n_components=self.hidden_features)

        # 量子电路相关参数
        self.qdev = qml.device("lightning.qubit", wires=self.hidden_features)
        weight_shape1 = (N, self.spectrum_layer, 2, self.hidden_features, 3)  # k由2改成3

        self.weights1 = nn.Parameter(
            torch.randn(weight_shape1, requires_grad=True) * 0.4
        )

        self.qnode = qml.QNode(
            func=self._circuit,
            device=self.qdev,
            interface="torch",
            diff_method="parameter-shift"  # 使用parameter-shift微分方法
        )

    def _circuit(self, inputs, weights1):
        inputs = inputs.flatten()  # 确保输入是一维

        for i in range(self.spectrum_layer):
            for j in range(self.hidden_features):
                qml.RZ(torch.pi * 0.5 * inputs[j], wires=j)
            qml.StronglyEntanglingLayers(weights1[i], wires=range(self.hidden_features), imprimitive=qml.ops.CZ)

        # 获取量子线路的输出概率分布
        probs = qml.probs(wires=range(self.hidden_features))
        return probs

    def _post_process(self, probs):
        """后处理：截取前 self.pixels 个概率值并缩放"""
        probs = probs[:self.pixels]  # 仅保留前 self.pixels 个值
        probs = probs * self.pixels  # 将概率缩放至像素范围
        probs = torch.clamp(probs, 0, 1)  # 限制概率值在 [0, 1] 范围内
        return probs

    def forward(self, x):
        b, c, w, h = x.shape
        x = x.view(b, -1)

        # 使用PCA进行降维
        x_reduced = self.pca.fit_transform(x.cpu().numpy())
        x_reduced = torch.tensor(x_reduced).to(x.device).to(x.dtype)

        # Reshape to match hidden_features dimension for quantum processing
        x_reduced = einops.rearrange(x_reduced, "b (c w h) -> b c w h", c=1, w=1, h=self.hidden_features)

        # 量子电路处理和后处理循环
        for n in range(self.N):
            x_processed = []
            for i in range(b):
                # 将量子电路的概率输出转换为张量以便进行切片
                probs = self.qnode(x_reduced[i], self.weights1[n])
                probs_tensor = probs.clone().detach().requires_grad_(True)  # 保留计算图信息
                processed_probs = self._post_process(probs_tensor)
                x_processed.append(processed_probs)

            x_reduced = torch.stack(x_processed)  # 将处理后的结果堆叠以形成正确的形状

        # 将 x_reduced 恢复为原始图像格式
        x_restored = x_reduced.view(b, c, w, h)

        return x_restored

    def __repr__(self):
        return f"QIDDM(qlayer={self.spectrum_layer}, features={self.hidden_features}, N={self.N})"

    def save_name(self) -> str:
        return f"QIDDM_pca_features={self.hidden_features}_L={self.spectrum_layer}_N={self.N}"

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


class QIDDM_A_sameN(nn.Module):
    """Dense variational circuit with batch processing enabled"""

    def __init__(self, shape, spectrum_layer, N) -> None:
        super().__init__()
        if isinstance(shape, int):
            shape = (shape, shape)
        self.spectrum_layer = spectrum_layer  # L
        self.N = N  # N
        self.width, self.height = shape
        self.pixels = self.width * self.height  # 总像素数
        self.wires = math.ceil(math.log2(self.pixels))  # 量子比特数目
        self.qdev = qml.device("default.qubit.torch", wires=self.wires)

        # 定义权重的形状： spectrum_layer 表示量子纠缠层数，2 是两个旋转参数，wires 是量子比特数，3 表示每个比特的旋转门数量
        weight_shape = (self.spectrum_layer, 2, self.wires, 3)

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
                # inputs 的每一行是一个展平的样本
                qml.RZ(inputs[:, j], wires=j)  # 对整个批次的数据执行角度编码
            qml.StronglyEntanglingLayers(weights[i], wires=range(self.wires), imprimitive=qml.ops.CZ)
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

        # 通过量子电路批量处理
        for n in range(self.N):
            x = self.qnode(x, self.weights)

        # 后处理量子电路输出的概率分布
        probs = self._post_process(x)

        # 将概率重新排列为 [batch_size, 1, width, height] 形状
        x = einops.rearrange(probs, "b (w h) -> b 1 w h", w=self.width, h=self.height)
        return x

    def __repr__(self):
        return f"QIDDM_A_sameN={self.spectrum_layer}_N={self.N}_w{self.width}_h{self.height}"

    def save_name(self) -> str:
        return f"QIDDM_A_sameN={self.spectrum_layer}_N={self.N}_w{self.width}_h{self.height}"


class QIDDM_A_differN_NEW(nn.Module):
    def __init__(self, input_dim, spectrum_layer, N: int) -> None:
        super().__init__()
        self.spectrum_layer = spectrum_layer  # L
        self.width = input_dim
        self.height = input_dim
        self.pixels = self.width * self.height  # 总像素数
        self.hidden_features = math.ceil(math.log2(self.pixels))  # num of qubits
        self.N = N  # 增加N参数

        # 使用PCA进行降维
        self.pca = PCA(n_components=self.hidden_features)

        # 量子电路相关参数
        self.qdev = qml.device("lightning.qubit", wires=self.hidden_features)
        weight_shape1 = (N, self.spectrum_layer, 2, self.hidden_features, 3)  # k由2改成3

        self.weights1 = nn.Parameter(
            torch.randn(weight_shape1, requires_grad=True) * 0.4
        )

        self.qnode = qml.QNode(
            func=self._circuit,
            device=self.qdev,
            interface="torch",
            diff_method="parameter-shift"  # 使用parameter-shift微分方法
        )

    def _circuit(self, inputs, weights1):
        inputs = inputs.flatten()  # 确保输入是一维

        for i in range(self.spectrum_layer):
            for j in range(self.hidden_features):
                qml.RZ(torch.pi * 0.5 * inputs[j], wires=j)
            qml.StronglyEntanglingLayers(weights1[i], wires=range(self.hidden_features), imprimitive=qml.ops.CZ)

        # 获取量子线路的输出概率分布
        probs = qml.probs(wires=range(self.hidden_features))
        return probs

    def _post_process(self, probs):
        """后处理：截取前 self.pixels 个概率值并缩放"""
        probs = probs[:self.pixels]  # 仅保留前 self.pixels 个值
        probs = probs * self.pixels  # 将概率缩放至像素范围
        probs = torch.clamp(probs, 0, 1)  # 限制概率值在 [0, 1] 范围内
        return probs

    def forward(self, x):
        b, c, w, h = x.shape
        x = x.view(b, -1)

        # 使用PCA进行降维
        x_reduced = self.pca.fit_transform(x.cpu().numpy())
        x_reduced = torch.tensor(x_reduced).to(x.device).to(x.dtype)

        # Reshape to match hidden_features dimension for quantum processing
        x_reduced = einops.rearrange(x_reduced, "b (c w h) -> b c w h", c=1, w=1, h=self.hidden_features)

        # 量子电路处理和后处理循环
        for n in range(self.N):
            x_processed = []
            for i in range(b):
                # 将量子电路的概率输出转换为张量以便进行切片
                probs = self.qnode(x_reduced[i], self.weights1[n])
                probs_tensor = probs.clone().detach().requires_grad_(True)  # 保留计算图信息
                processed_probs = self._post_process(probs_tensor)
                x_processed.append(processed_probs)

            x_reduced = torch.stack(x_processed)  # 将处理后的结果堆叠以形成正确的形状

        # 将 x_reduced 恢复为原始图像格式
        x_restored = x_reduced.view(b, c, w, h)

        return x_restored

    def __repr__(self):
        return f"QIDDM(qlayer={self.spectrum_layer}, features={self.hidden_features}, N={self.N})"

    def save_name(self) -> str:
        return f"QIDDM_pca_new={self.hidden_features}_L={self.spectrum_layer}_N={self.N}"

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
