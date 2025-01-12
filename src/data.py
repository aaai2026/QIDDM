import torch
import torchvision
import torchvision.transforms as T
from sklearn import datasets
import pandas as pd
from collections import Counter  # 添加这一行
from torch.utils.data import Subset, DataLoader  # 添加这一行


def mnist_8x8(n_classes=10, ds_size=100):
    x_train, y_train = datasets.load_digits(n_class=n_classes, return_X_y=True)
    x_train /= 16
    x_train = x_train.reshape(-1, 64)
    x_train = torch.tensor(x_train, dtype=torch.double)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_train, y_train = x_train[:ds_size], y_train[:ds_size]
    return x_train, y_train, 8, 8


def mnist_28x28(n_classes=10, ds_size=100):
    ds = torchvision.datasets.MNIST(root="~/mnist", download=True, transform=T.ToTensor())
    idx = ds.targets < n_classes
    ds.data = ds.data[idx]
    ds.targets = ds.targets[idx]
    x_train, y_train = torch.utils.data.DataLoader(ds, batch_size=ds_size, shuffle=False).__iter__().__next__()
    x_train = x_train.flatten(start_dim=1)
    x_train = x_train.to(torch.double)
    y_train = y_train.to(torch.long)
    return x_train, y_train, 28, 28


def mnist_32x32(n_classes=10, ds_size=100):
    tra = T.Compose([
        T.ToTensor(),
        T.Resize((32, 32)),
    ])
    ds = torchvision.datasets.MNIST(root="~/mnist", download=True, transform=tra)
    idx = ds.targets < n_classes
    ds.data = ds.data[idx]
    ds.targets = ds.targets[idx]
    x_train, y_train = torch.utils.data.DataLoader(ds, batch_size=ds_size, shuffle=False).__iter__().__next__()
    x_train = x_train.flatten(start_dim=1)
    x_train = x_train.to(torch.double)
    y_train = y_train.to(torch.long)
    return x_train, y_train, 32, 32


def cifar10_32x32(n_classes=10, ds_size=100):
    transformation = T.Compose([T.functional.rgb_to_grayscale, T.ToTensor(), T.Lambda(lambda x: torch.flatten(x))])
    ds = torchvision.datasets.CIFAR10(root="~/cifar", download=True, transform=transformation)
    ds.targets = torch.tensor(ds.targets)
    idx = torch.tensor(ds.targets) < n_classes
    ds.data = ds.data[idx.numpy()]
    ds.targets = ds.targets[idx.numpy()]
    x_train, y_train = torch.utils.data.DataLoader(ds, batch_size=ds_size, shuffle=False).__iter__().__next__()
    x_train = x_train.to(torch.double)
    y_train = y_train.to(torch.long)
    return x_train, y_train, 32, 32


def fashion_28x28(n_classes=10, ds_size=100):
    ds = torchvision.datasets.FashionMNIST(root="~/fashion", download=True, transform=T.ToTensor())
    ds.targets = torch.tensor(ds.targets)
    idx = ds.targets < n_classes
    ds.data = ds.data[idx]
    ds.targets = ds.targets[idx]
    x_train, y_train = torch.utils.data.DataLoader(ds, batch_size=ds_size, shuffle=False).__iter__().__next__()
    x_train = einops.rearrange(x_train, 'b 1 h w -> b (h w)')
    x_train = x_train.to(torch.double)
    y_train = y_train.to(torch.long)
    return x_train, y_train, 28, 28


def celeba_32x32(label):
    labels_path = '~/celeba/celeba/identity_CelebA.txt'

    # Load identity labels from the TXT file
    labels = pd.read_csv(labels_path, delim_whitespace=True, header=None)
    # Define transformations
    transformation = T.Compose([
        T.Resize((32, 32)),
        T.Grayscale(),  # Convert RGB images to grayscale
        T.ToTensor(),  # Convert image to tensor
        T.Lambda(lambda x: torch.flatten(x))  # Flatten the image into a 1D vector
    ])
    # Load the CelebA dataset
    ds = torchvision.datasets.CelebA(root="~/celeba", download=True, split='train', transform=transformation)

    # Get indices for the specified identity
    identity_indices = []
    for i in range(labels[0].size):
        if labels[1][i] == label:
            identity_indices.append(i)

    # Filter the dataset for the specified identity
    filtered_data = [ds[i] for i in identity_indices]

    # Prepare x_train and y_train
    x_train = torch.stack([data[0] for data in filtered_data])
    y_train = torch.stack([data[1] for data in filtered_data])

    return x_train, y_train, 32, 32


def celeba_64x64(label):
    labels_path = '~/celeba/celeba/identity_CelebA.txt'

    # Load identity labels from the TXT file
    labels = pd.read_csv(labels_path, delim_whitespace=True, header=None)
    # Define transformations
    transformation = T.Compose([
        T.Resize((64, 64)),
        T.Grayscale(),  # Convert RGB images to grayscale
        T.ToTensor(),  # Convert image to tensor
        T.Lambda(lambda x: torch.flatten(x))  # Flatten the image into a 1D vector
    ])
    # Load the CelebA dataset
    ds = torchvision.datasets.CelebA(root="~/celeba", download=True, split='train', transform=transformation)

    # Get indices for the specified identity
    identity_indices = []
    for i in range(labels[0].size):
        if labels[1][i] == label:
            identity_indices.append(i)

    # Filter the dataset for the specified identity
    filtered_data = [ds[i] for i in identity_indices]

    # Prepare x_train and y_train
    x_train = torch.stack([data[0] for data in filtered_data])
    y_train = torch.stack([data[1] for data in filtered_data])

    return x_train, y_train, 64, 64



def lfw_28x28(n_classes=10, ds_size=1000):
    # 定义图像转换：调整尺寸为28x28，并转换为灰度图像，扁平化
    transform = T.Compose([
        T.Resize((28, 28)),
        T.Grayscale(num_output_channels=1),  # 如果需要彩色图像则移除这一行
        T.ToTensor(),
        T.Lambda(lambda x: torch.flatten(x))  # 将图像展平成1D向量
    ])

    # 加载LFWPeople数据集
    ds = torchvision.datasets.LFWPeople(root="~/lfw", download=True, transform=transform)

    # 转换目标为张量并获取唯一的类标签
    targets = torch.tensor(ds.targets)

    # 统计每个标签的样本数量
    label_counts = Counter(targets.numpy())

    # 找到样本数量最多的前10个标签
    most_common_labels = label_counts.most_common(n_classes)

    # 创建标签映射表，将 top 10 标签重新映射为 0-9
    label_mapping = {label: i for i, (label, _) in enumerate(most_common_labels)}

    # 获取所有属于 top 10 标签的样本索引，并进行重新标记
    sampled_indices = []
    new_targets = []
    for original_label, new_label in label_mapping.items():
        # 获取属于该标签的所有样本索引
        class_indices = torch.where(targets == original_label)[0]
        sampled_indices.append(class_indices)
        new_targets.extend([new_label] * len(class_indices))

    # 将所有类别的样本组合在一起
    sampled_indices = torch.cat(sampled_indices)
    new_targets = torch.tensor(new_targets)

    # 创建子集并加载数据
    subset = Subset(ds, sampled_indices)
    loader = DataLoader(subset, batch_size=len(sampled_indices), shuffle=False)
    x_data, _ = next(iter(loader))

    # 确保数据类型正确
    x_data = x_data.to(torch.double)
    new_targets = new_targets.to(torch.long)

    # 从中随机采样 ds_size 个样本
    sampled_dataset = torch.randperm(len(x_data))[:ds_size]
    x_train = x_data[sampled_dataset]
    y_train = new_targets[sampled_dataset]

    # 返回 28x28 图像数据及其标签
    return x_train, y_train, 28, 28


import torch
import torchvision
import torchvision.transforms as T
import einops
import torchvision.transforms.functional as F


def emnist_28x28(n_classes=52, ds_size=1000):
    # 加载 EMNIST 字母数据集
    ds = torchvision.datasets.EMNIST(root="~/emnist", split='letters', download=True, transform=T.ToTensor())

    # EMNIST Letters 子集中的标签为1到26，表示A到Z的字母（1是'A', 2是'B', ..., 26是'Z'）
    # 我们需要调整标签，使其从0开始，便于分类任务（即0对应'a', 1对应'b', ..., 25对应'z'）
    ds.targets = torch.tensor(ds.targets) - 1

    # 仅选择前n_classes个类别的小写字母
    idx = ds.targets < n_classes
    ds.data = ds.data[idx]
    ds.targets = ds.targets[idx]

    # 从数据集中抽取数据
    x_train, y_train = torch.utils.data.DataLoader(ds, batch_size=ds_size, shuffle=False).__iter__().__next__()

    # 将数据调整为需要的形状，并旋转图像
    x_train = einops.rearrange(x_train, 'b 1 h w -> b h w')

    # 恢复到图像的三维形状，旋转并镜面翻转后再展平
    x_train = torch.stack([F.hflip(F.rotate(img.unsqueeze(0), angle=-90)) for img in x_train])  # 旋转90度并水平镜面翻转
    x_train = x_train.view(-1, 28 * 28)

    x_train = x_train.to(torch.double)
    y_train = y_train.to(torch.long)

    return x_train, y_train, 28, 28


def lfw_28x28(n_classes=10, ds_size=1000):
    # 定义图像转换：调整尺寸为28x28，并转换为灰度图像，扁平化
    transform = T.Compose([
        T.Resize((28, 28)),
        T.Grayscale(num_output_channels=1),  # 如果需要彩色图像则移除这一行
        T.ToTensor(),
        T.Lambda(lambda x: torch.flatten(x))  # 将图像展平成1D向量
    ])

    # 加载LFWPeople数据集
    ds = torchvision.datasets.LFWPeople(root="~/lfw", download=True, transform=transform)

    # 转换目标为张量并获取唯一的类标签
    targets = torch.tensor(ds.targets)

    # 统计每个标签的样本数量
    label_counts = Counter(targets.numpy())

    # 找到样本数量最多的前10个标签
    most_common_labels = label_counts.most_common(n_classes)

    # 创建标签映射表，将 top 10 标签重新映射为 0-9
    label_mapping = {label: i for i, (label, _) in enumerate(most_common_labels)}

    # 获取所有属于 top 10 标签的样本索引，并进行重新标记
    sampled_indices = []
    new_targets = []
    for original_label, new_label in label_mapping.items():
        # 获取属于该标签的所有样本索引
        class_indices = torch.where(targets == original_label)[0]
        sampled_indices.append(class_indices)
        new_targets.extend([new_label] * len(class_indices))

    # 将所有类别的样本组合在一起
    sampled_indices = torch.cat(sampled_indices)
    new_targets = torch.tensor(new_targets)

    # 创建子集并加载数据
    subset = Subset(ds, sampled_indices)
    loader = DataLoader(subset, batch_size=len(sampled_indices), shuffle=False)
    x_data, _ = next(iter(loader))

    # 确保数据类型正确
    x_data = x_data.to(torch.double)
    new_targets = new_targets.to(torch.long)

    # 从中随机采样 ds_size 个样本
    sampled_dataset = torch.randperm(len(x_data))[:ds_size]
    x_train = x_data[sampled_dataset]
    y_train = new_targets[sampled_dataset]

    # 返回 28x28 图像数据及其标签
    return x_train, y_train, 28, 28


def lfw_64x64(n_classes=10, ds_size=1000):
    # 定义图像转换：调整尺寸为28x28，并转换为灰度图像，扁平化
    transform = T.Compose([
        T.Resize((64, 64)),
        T.Grayscale(num_output_channels=1),  # 如果需要彩色图像则移除这一行
        T.ToTensor(),
        T.Lambda(lambda x: torch.flatten(x))  # 将图像展平成1D向量
    ])

    # 加载LFWPeople数据集
    ds = torchvision.datasets.LFWPeople(root="~/lfw", download=True, transform=transform)

    # 转换目标为张量并获取唯一的类标签
    targets = torch.tensor(ds.targets)

    # 统计每个标签的样本数量
    label_counts = Counter(targets.numpy())

    # 找到样本数量最多的前10个标签
    most_common_labels = label_counts.most_common(n_classes)

    # 创建标签映射表，将 top 10 标签重新映射为 0-9
    label_mapping = {label: i for i, (label, _) in enumerate(most_common_labels)}

    # 获取所有属于 top 10 标签的样本索引，并进行重新标记
    sampled_indices = []
    new_targets = []
    for original_label, new_label in label_mapping.items():
        # 获取属于该标签的所有样本索引
        class_indices = torch.where(targets == original_label)[0]
        sampled_indices.append(class_indices)
        new_targets.extend([new_label] * len(class_indices))

    # 将所有类别的样本组合在一起
    sampled_indices = torch.cat(sampled_indices)
    new_targets = torch.tensor(new_targets)

    # 创建子集并加载数据
    subset = Subset(ds, sampled_indices)
    loader = DataLoader(subset, batch_size=len(sampled_indices), shuffle=False)
    x_data, _ = next(iter(loader))

    # 确保数据类型正确
    x_data = x_data.to(torch.double)
    new_targets = new_targets.to(torch.long)

    # 从中随机采样 ds_size 个样本
    sampled_dataset = torch.randperm(len(x_data))[:ds_size]
    x_train = x_data[sampled_dataset]
    y_train = new_targets[sampled_dataset]

    # 返回 28x28 图像数据及其标签
    return x_train, y_train, 64, 64


def lfw_128x128(n_classes=10, ds_size=1000):
    # 定义图像转换：调整尺寸为28x28，并转换为灰度图像，扁平化
    transform = T.Compose([
        T.Resize((128, 128)),
        T.Grayscale(num_output_channels=1),  # 如果需要彩色图像则移除这一行
        T.ToTensor(),
        T.Lambda(lambda x: torch.flatten(x))  # 将图像展平成1D向量
    ])

    # 加载LFWPeople数据集
    ds = torchvision.datasets.LFWPeople(root="~/lfw", download=True, transform=transform)

    # 转换目标为张量并获取唯一的类标签
    targets = torch.tensor(ds.targets)

    # 统计每个标签的样本数量
    label_counts = Counter(targets.numpy())

    # 找到样本数量最多的前10个标签
    most_common_labels = label_counts.most_common(n_classes)

    # 创建标签映射表，将 top 10 标签重新映射为 0-9
    label_mapping = {label: i for i, (label, _) in enumerate(most_common_labels)}

    # 获取所有属于 top 10 标签的样本索引，并进行重新标记
    sampled_indices = []
    new_targets = []
    for original_label, new_label in label_mapping.items():
        # 获取属于该标签的所有样本索引
        class_indices = torch.where(targets == original_label)[0]
        sampled_indices.append(class_indices)
        new_targets.extend([new_label] * len(class_indices))

    # 将所有类别的样本组合在一起
    sampled_indices = torch.cat(sampled_indices)
    new_targets = torch.tensor(new_targets)

    # 创建子集并加载数据
    subset = Subset(ds, sampled_indices)
    loader = DataLoader(subset, batch_size=len(sampled_indices), shuffle=False)
    x_data, _ = next(iter(loader))

    # 确保数据类型正确
    x_data = x_data.to(torch.double)
    new_targets = new_targets.to(torch.long)

    # 从中随机采样 ds_size 个样本
    sampled_dataset = torch.randperm(len(x_data))[:ds_size]
    x_train = x_data[sampled_dataset]
    y_train = new_targets[sampled_dataset]

    # 返回 28x28 图像数据及其标签
    return x_train, y_train, 128, 128


def lfw_512x512(n_classes=10, ds_size=1000):
    # 定义图像转换：调整尺寸为28x28，并转换为灰度图像，扁平化
    transform = T.Compose([
        T.Resize((512, 512)),
        T.Grayscale(num_output_channels=1),  # 如果需要彩色图像则移除这一行
        T.ToTensor(),
        T.Lambda(lambda x: torch.flatten(x))  # 将图像展平成1D向量
    ])

    # 加载LFWPeople数据集
    ds = torchvision.datasets.LFWPeople(root="~/lfw", download=True, transform=transform)

    # 转换目标为张量并获取唯一的类标签
    targets = torch.tensor(ds.targets)

    # 统计每个标签的样本数量
    label_counts = Counter(targets.numpy())

    # 找到样本数量最多的前10个标签
    most_common_labels = label_counts.most_common(n_classes)

    # 创建标签映射表，将 top 10 标签重新映射为 0-9
    label_mapping = {label: i for i, (label, _) in enumerate(most_common_labels)}

    # 获取所有属于 top 10 标签的样本索引，并进行重新标记
    sampled_indices = []
    new_targets = []
    for original_label, new_label in label_mapping.items():
        # 获取属于该标签的所有样本索引
        class_indices = torch.where(targets == original_label)[0]
        sampled_indices.append(class_indices)
        new_targets.extend([new_label] * len(class_indices))

    # 将所有类别的样本组合在一起
    sampled_indices = torch.cat(sampled_indices)
    new_targets = torch.tensor(new_targets)

    # 创建子集并加载数据
    subset = Subset(ds, sampled_indices)
    loader = DataLoader(subset, batch_size=len(sampled_indices), shuffle=False)
    x_data, _ = next(iter(loader))

    # 确保数据类型正确
    x_data = x_data.to(torch.double)
    new_targets = new_targets.to(torch.long)

    # 从中随机采样 ds_size 个样本
    sampled_dataset = torch.randperm(len(x_data))[:ds_size]
    x_train = x_data[sampled_dataset]
    y_train = new_targets[sampled_dataset]

    # 返回 28x28 图像数据及其标签
    return x_train, y_train, 512, 512
