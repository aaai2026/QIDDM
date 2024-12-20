import pathlib
import matplotlib.pyplot as plt
import shutil
import os
import glob
import torchvision.utils as vutils
from scipy.linalg import sqrtm

from pytorch_fid import fid_score

from scipy.interpolate import make_interp_spline
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import pytorch_fid
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import matplotlib.pyplot as plt
import pathlib


def show_histogram(score_dict, metric, args, model_name=None, model_params=None, filename=None):
    # 模型名称和分数
    models = list(score_dict.keys())  # 模型名字
    scores = np.array(list(score_dict.values()))  # 转换为 NumPy 数组 (shape: num_models x num_labels)

    num_models = len(models)  # 模型数量
    num_labels = len(scores[0])  # label 数量

    # 横轴位置和柱宽
    x = np.arange(num_labels)  # 每个 label 的位置
    bar_width = 0.5 / num_models  # 动态调整柱宽

    # 颜色列表，每个模型对应一个颜色
    colors = [
        '#9FABB9', '#D4E1F5', '#7EA6E0', '#D3E2B7', '#7CB862', '#FFCE9F', '#9467bd', '#7f7f7f',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#cfcfcf'
    ]
    # 如果模型数量超过颜色列表的长度，循环使用颜色
    color_cycle = (colors * ((num_models // len(colors)) + 1))[:num_models]

    # 绘制柱状图
    plt.figure(figsize=(12, 6))
    for i, model in enumerate(models):
        # 为每个模型选择固定的颜色
        color = color_cycle[i]  # 每个模型分配一个固定颜色

        # 绘制每个模型的柱子，所有 label 下的柱子使用相同颜色
        for j in range(num_labels):
            plt.bar(
                x[j] + i * bar_width,  # 每个模型的柱子位置
                scores[i, j],  # 分数
                width=bar_width,  # 柱子宽度
                color=color,  # 当前模型的颜色
                label=model if j == 0 else "",  # 只在第一个柱子上显示标签
            )

    # 添加图例、标题和轴标签
    plt.title(f'{metric} of Models Across Labels', fontsize=18)
    plt.xlabel(f"{args.data} Labels", fontsize=16)
    plt.ylabel(f'{metric}', fontsize=16)
    plt.xticks(x + bar_width * (num_models - 1) / 2, [f'Label {i + 1}' for i in range(num_labels)], fontsize=14)  # 居中每组柱状图的刻度
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14, markerscale=1.5)  # 调整图例字体大小为 14，标记大小为 1.5 倍

    # 设置 y 轴的范围，增加最大值的间距
    max_score = np.max(scores)  # 获取最大分数
    plt.ylim(0, max_score * 1.1)  # 设置 y 轴最大值为最大分数的 1.1 倍

    # 保存图像，文件名包含模型名称和参数信息
    if model_name and model_params:
        model_params_str = "_".join(map(str, model_params))
        model_info = f"{model_name}_{model_params_str}"
    else:
        model_info = model_name or "unknown_model"


    # 如果指定了保存路径，则保存图像
    if args.save_path:
        sp = pathlib.Path(args.save_path) / f"{metric}_{model_info}_{args.label}.png"
        plt.tight_layout()
        plt.savefig(sp)  # 保存图像
        print(f"{metric} plot saved to {sp}")

    # 显示图像
    # plt.show()
    plt.close()



def show_metrics(values_dict, metric, Xlabel, args, model_name=None, model_params=None,
                 colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#7f7f7f'],
                 legend_labels=None, xlabel=None, ylabel=None, is_loss=False,
                 marker_size=7, line_width=2):  # 新增的参数：marker_size 和 line_width
    plt.figure(figsize=(10, 6))

    # 默认颜色列表（RGB）
    if colors is None:
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # 默认的7种颜色
    if legend_labels is None:
        legend_labels = list(values_dict.keys())  # 使用字典的键作为默认图例标签

    markers = ['o', 's', '^', 'd', 'x', '*', '+', 'v', '<', '>', 'p', 'h']  # 可选择的标记样式

    for idx, (model_name_key, values) in enumerate(values_dict.items()):
        # 如果颜色不足，循环使用颜色列表
        color = colors[idx % len(colors)]  # 循环使用颜色
        marker = markers[idx % len(markers)]  # 循环使用标记样式
        
        x = range(len(values))

        if is_loss:  # 如果是loss图，不进行平滑处理
            plt.plot(x, values, linestyle='-', label=legend_labels[idx], color=color, marker=marker, 
                     markersize=marker_size, linewidth=line_width)
        else:  # 否则进行平滑处理
            x_smooth = torch.linspace(0, len(values) - 1, 25)
            values_smooth = make_interp_spline(x, values)(x_smooth)
            plt.plot(x_smooth, values_smooth, linestyle='-', label=legend_labels[idx], color=color, marker=marker, 
                     markersize=marker_size, linewidth=line_width)

    plt.title(f'{metric} over {Xlabel}')

    # 设置横轴和纵轴名称，如果未传入，则使用默认
    plt.xlabel(xlabel or f'{Xlabel}')
    plt.ylabel(ylabel or f'{metric}')
    
    plt.grid(True)
    plt.legend()

    # 将模型参数转为字符串，确保文件名合法
    if model_name and model_params:
        model_params_str = "_".join(map(str, model_params))
        model_info = f"{model_name}_{model_params_str}"
    else:
        model_info = model_name or "unknown_model"

    # 保存图像，文件名包含模型名称和参数信息
    sp = pathlib.Path(args.save_path) / f"{metric}-{Xlabel}_{args.label}.png"
    plt.tight_layout()
    plt.savefig(sp)  # 保存图像
    print(f"{metric}-{Xlabel} plot saved to {sp}")

    # plt.show()
    plt.close()


def print_image_count(folder, label):
    images = glob.glob(f'{folder}/*.png')
    print(f"{label}: {len(images)} images")


def calculate_cos(v1, v2):
    # print(f"v2 shape: {v2.shape}")
    _, height, width = v2.shape
    pixels = height * width

    v1 = v1.detach().cpu().numpy().reshape(-1, pixels)
    v2 = v2.detach().cpu().numpy().reshape(-1, pixels)
    num = np.dot(v1, np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res


def get_cosine_similarity(generated_images_dict, real_images_dict, args, gen_img_count=None, real_img_count=None):
    cos_values_dict = {}

    for model_name in generated_images_dict.keys():
        generated_images = generated_images_dict[model_name]
        real_images = real_images_dict[model_name]

        # 调整生成图像和真实图像的数量
        if gen_img_count is not None and gen_img_count < generated_images.shape[1]:
            generated_images = generated_images[:, :gen_img_count, :, :, :]
        if real_img_count is not None and real_img_count < real_images.shape[0]:
            real_images = real_images[:real_img_count, :, :, :]

        cosine_similarity_values = []

        for iteration in range(generated_images.shape[0]):
            iteration_cosine_values = []

            for i in range(generated_images.shape[1]):
                for j in range(real_images.shape[0]):
                    cosine_value = calculate_cos(generated_images[iteration, i], real_images[j])
                    iteration_cosine_values.append(cosine_value)

            avg_cosine_similarity = torch.tensor(iteration_cosine_values).mean().item()
            cosine_similarity_values.append(avg_cosine_similarity)
            # print(f"Iteration {iteration}, Average Cosine Similarity: {avg_cosine_similarity}")

        cos_values_dict[model_name] = cosine_similarity_values

    # 在这里传递 model_name 和 model_params
    show_metrics(cos_values_dict, 'Cosine Similarity', 'Iteration', args, model_name=model_name,
                 model_params=args.model[0][1:])

    return cos_values_dict


def get_ssim(generated_images_dict, real_images_dict, args, gen_img_count=None, real_img_count=None):
    ssim_values_dict = {}

    for model_name in generated_images_dict.keys():
        generated_images = generated_images_dict[model_name]
        real_images = real_images_dict[model_name]

        if gen_img_count is not None and gen_img_count < generated_images.shape[1]:
            generated_images = generated_images[:, :gen_img_count, :, :, :]
        if real_img_count is not None and real_img_count < real_images.shape[0]:
            real_images = real_images[:real_img_count, :, :, :]

        # print(real_images.shape)
        # real_image = real_images[0].squeeze().cpu().numpy()
        # print(real_image.shape)
        ssim_values = []
        # print(generated_images.shape)

        for iteration in range(generated_images.shape[0]):
            iteration_ssim_values = []

            for i in range(generated_images.shape[1]):
                for j in range(real_images.shape[0]):
                    generated_image = generated_images[iteration, i].squeeze().cpu().numpy()
                    real_image = real_images[j].squeeze().cpu().numpy()
                    ssim_value = ssim(generated_image, real_image, data_range=generated_image.max() - generated_image.min())
                    iteration_ssim_values.append(ssim_value)

            avg_ssim = torch.tensor(iteration_ssim_values).mean().item()
            ssim_values.append(avg_ssim)
            print(f"Iteration {iteration}, Average SSIM: {avg_ssim}")

        ssim_values_dict[model_name] = ssim_values

    show_metrics(ssim_values_dict, 'SSIM', 'Iteration', args, model_name=model_name, model_params=args.model)

    return ssim_values_dict

def get_ssim_single(generated_images, real_images, args, gen_img_count=None, real_img_count=None):

    if gen_img_count is not None and gen_img_count < generated_images.shape[1]:
        generated_images = generated_images[:, :gen_img_count, :, :, :]
    if real_img_count is not None and real_img_count < real_images.shape[0]:
        real_images = real_images[:real_img_count, :, :, :]

    real_image = real_images[0].squeeze().cpu().numpy()
    ssim_values = []

    for iteration in range(generated_images.shape[0]):
        iteration_ssim_values = []

        for i in range(generated_images.shape[1]):
            generated_image = generated_images[iteration, i].squeeze().cpu().numpy()
            ssim_value = ssim(generated_image, real_image, data_range=generated_image.max() - generated_image.min())
            iteration_ssim_values.append(ssim_value)

        avg_ssim = torch.tensor(iteration_ssim_values).mean().item()
        ssim_values.append(avg_ssim)

    # print(f"last ssim: { ssim_values[-1]}")
    return ssim_values



def get_psnr(generated_images_dict, real_images_dict, args, gen_img_count=None, real_img_count=None):
    psnr_values_dict = {}

    for model_name in generated_images_dict.keys():
        generated_images = generated_images_dict[model_name]
        real_images = real_images_dict[model_name]

        if gen_img_count is not None and gen_img_count < generated_images.shape[1]:
            generated_images = generated_images[:, :gen_img_count, :, :, :]
        if real_img_count is not None and real_img_count < real_images.shape[0]:
            real_images = real_images[:real_img_count, :, :, :]

        
        psnr_values = []

        for iteration in range(generated_images.shape[0]):
            iteration_psnr_values = []

            for i in range(generated_images.shape[1]):
                for j in range(real_images.shape[0]):
                    generated_image = generated_images[iteration, i].squeeze().cpu().numpy()
                    real_image = real_images[j].squeeze().cpu().numpy()
                    psnr_value = psnr(real_image, generated_image, data_range=generated_image.max() - generated_image.min())
                    iteration_psnr_values.append(psnr_value)

            avg_psnr = torch.tensor(iteration_psnr_values).mean().item()
            psnr_values.append(avg_psnr)
            # print(f"Iteration {iteration}, Average PSNR: {avg_psnr}")

        psnr_values_dict[model_name] = psnr_values

    show_metrics(psnr_values_dict, 'PSNR', 'Iteration', args, model_name=model_name, model_params=args.model)

    return psnr_values_dict


def get_fid(generated_images_dict, real_images_dict, args, gen_img_count=None, real_img_count=None):
    fid_values_dict = {}

    for model_name in generated_images_dict.keys():
        # gen images: torch.Size([21, 15, 1, 28, 28])
        # real images: torch.Size([10, 1, 28, 28])
        generated_images = generated_images_dict[model_name]
        real_images = real_images_dict[model_name]

        if gen_img_count is not None and gen_img_count < generated_images.shape[1]:
            generated_images = generated_images[:, :gen_img_count, :, :, :]
        if real_img_count is not None and real_img_count < real_images.shape[0]:
            real_images = real_images[:real_img_count, :, :, :]

        fid_values = []

        for iteration in range(generated_images.shape[0]):
            iteration_fid_values = []
            generated_image = generated_images[iteration].squeeze().cpu().numpy()
            real_image = real_images.squeeze().cpu().numpy()
            fid_value = calculate_fid(generated_image, real_image, gen_img_count, real_img_count)
            iteration_fid_values.append(fid_value)

            avg_fid = torch.tensor(iteration_fid_values).mean().item()
            fid_values.append(avg_fid)  # every batch's fid

        fid_values_dict[model_name] = fid_values

    show_metrics(fid_values_dict, 'fid', args, model_name=model_name, model_params=args.model)

    return fid_values_dict


def calculate_fid(act1, act2, n1, n2):
    #act1 = act1.detach().cpu().numpy().reshape([1, 784])
    act1 = act1.reshape([n1, -1])
    act2 = act2.reshape([n2, -1])
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def show_images(images, num_images=5, img_size=(8, 8)):
    num = min(num_images, len(images))

    fig, axes = plt.subplots(1, num, figsize=(15, 3))

    if num == 1:
        axes = [axes]

    for i in range(num):
        ax = axes[i]
        img = images[i].cpu().numpy().reshape(img_size)
        ax.imshow(img, cmap='gray')
        ax.axis('off')

    plt.show() 