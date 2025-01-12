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


def map_model_name(model_name):
    # 直接映射字典
    model_name_mapping = {
        # 'UNetUndirected': 'U-net(27.7k params)',
        'UNetUndirected': 'U-net',
        'differN_noise': 'QIDDMA',
        'QDenseUndirected_old_noise': 'Qdense',
        'QIDDM_PL_noise': 'QIDDML',
        'QNN_noise': 'QNN'
    }

    if model_name is None:
        return model_name

    # 尝试从字典直接获取模型名称映射
    mapped_name = model_name_mapping.get(model_name)
    if mapped_name:
        return mapped_name

    # 如果字典中没有找到，继续使用子串匹配逻辑
    model_name = model_name.lower()  # 转为小写以进行不区分大小写的比较

    if 'differn' in model_name:
        return 'QIDDMA'
    if 'qdenseundirected' in model_name:
        return 'Qdense'
    if 'qiddm_pl' in model_name:
        return 'QIDDML'
    if 'qnn' in model_name:
        return 'QNN'
    if 'unet_undirected' in model_name:
        # return 'U-net(27.7k params)'
        return 'U-net'

    # 如果都没有匹配，返回原始模型名称
    return model_name


def show_histogram(score_dict, metric, args, model_name=None, model_params=None, filename=None):
    models = list(score_dict.keys())
    scores = np.array(list(score_dict.values()))
    num_models = len(models)
    num_labels = len(scores[0])

    x = np.arange(num_labels)  # Label positions
    bar_width = 0.5 / num_models  # Adjust bar width dynamically

    colors = ['#9FABB9', '#D4E1F5', '#7EA6E0', '#D3E2B7', '#7CB862', '#FFCE9F', '#9467bd', '#7f7f7f']
    color_cycle = (colors * ((num_models // len(colors)) + 1))[:num_models]

    plt.figure(figsize=(12, 6))
    for i, model in enumerate(models):
        color = color_cycle[i]
        model_label = map_model_name(model)

        for j in range(num_labels):
            plt.bar(x[j] + i * bar_width, scores[i, j], width=bar_width, color=color,
                    label=model_label if j == 0 else "")

    plt.title(f'{metric} of Models Across Labels', fontsize=18)
    plt.xlabel(f"{args.data} Labels", fontsize=16)
    plt.ylabel(f'{metric}', fontsize=16)
    plt.xticks(x + bar_width * (num_models - 1) / 2, [f'Label {i}' for i in range(num_labels)], fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14, markerscale=1.5)

    max_score = np.max(scores)
    plt.ylim(0, max_score * 1.1)

    model_info = f"{map_model_name(model_name)}_{'_'.join(map(str, model_params))}" if model_name and model_params else "unknown_model"
    if hasattr(args, 'save_path') and args.save_path:
        sp = pathlib.Path(args.save_path) / f"{metric}_{model_info}_{args.label}.png"
        plt.tight_layout()
        plt.savefig(sp, dpi=300)
        print(f"{metric} plot saved to {sp}")

    # plt.show()
    plt.close()


def show_metrics(values_dict, name, args, model_name=None, model_params=None,
                 colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#7f7f7f'],
                 legend_labels=None, xlabel=None, ylabel=None, is_loss=False,
                 marker_size=7, line_width=3):

    if legend_labels is None:
        legend_labels = list(values_dict.keys())  # Use the keys of the dictionary as default legend labels

    legend_labels = [map_model_name(label) for label in legend_labels]

    # Determine x-axis label based on `is_loss` flag
    xlabel = xlabel or ("Epochs" if is_loss else "Denoising steps")

    # plt.figure(figsize=(10, 6))
    plt.figure(figsize=(8, 6))

    markers = ['o', 's', '^', 'd', 'x', '*', '+', 'v', '<', '>', 'p', 'h']  # Available marker styles

    for idx, (model_name_key, values) in enumerate(values_dict.items()):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        x = range(len(values))

        if is_loss:
            # Plot only the line (no markers)
            plt.plot(x, values, linestyle='-', color=color,
                     linewidth=line_width, label=legend_labels[idx])
        else:
            # Plot with markers for non-loss case
            plt.plot(x, values, linestyle='-', label=legend_labels[idx], color=color, marker=marker,
                     markersize=marker_size, linewidth=line_width)

    plt.title(f'{name}', fontsize=24)
    plt.xlabel(xlabel, fontsize=22)  # Set x-axis label dynamically
    plt.ylabel(ylabel or f'{name}', fontsize=22)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=18)

    model_info = f"{model_name}_{'_'.join(map(str, model_params))}" if model_name and model_params else "unknown_model"
    if hasattr(args, 'save_path') and args.save_path:
        sp = pathlib.Path(args.save_path) / f"{name}_{model_info}_{args.label}.png"
        plt.tight_layout()
        plt.savefig(sp, dpi=300)  # Save high-quality image with dpi=300
        print(f"{name} plot saved to {sp}")

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
    show_metrics(cos_values_dict, 'Cosine Similarity', args, model_name=model_name,
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

    show_metrics(ssim_values_dict, 'SSIM', args, model_name=model_name, model_params=args.model)

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

    show_metrics(psnr_values_dict, 'PSNR', args, model_name=model_name, model_params=args.model)

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

    # plt.show()