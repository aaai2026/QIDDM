import matplotlib.pyplot as plt
import torch
from einops import einops
import pennylane as qml
import data
import noise
import models
import nn
import argparse
import sys
import inspect
import warnings
import tqdm
import pathlib
import numpy as np

# log
import os
from Log import Logger
import time

# model
from metrics import *
from nn import QDenseUndirected_old_noise, QNN_A, QNN, differN_noise, QIDDM_PL_noise, UNetUndirected, QIDDM_L

from ray import tune
from ray.tune.schedulers import ASHAScheduler, AsyncHyperBandScheduler

all_nn = [name for name, obj in inspect.getmembers(nn) if inspect.isclass(obj)]
all_ds = [
    name
    for name, obj in inspect.getmembers(data)
    if inspect.isfunction(obj) and not name.startswith("_")
]


def parse_args(args):
    parser = argparse.ArgumentParser(description="Quantum Denoising Diffusion Model")
    parser.add_argument("--data", type=str, default="mnist_28x28",
                        help=f"Dataset to use. Available datasets: {', '.join(all_ds)}.")
    parser.add_argument("--img_size", type=int, default=28, help="Image size.")
    parser.add_argument("--label", type=int, default=4, help="Specify the label to be used for training.")
    parser.add_argument("--reduced_size", type=float, default=1.0, help="Reduced dataset size ratio.")
    parser.add_argument("--load-path", type=str, default="", help="Load model path.")
    parser.add_argument("--save-path", type=str, default="", help="Path to save results.")
    parser.add_argument("--n_classes", type=int, default=10, help="Number of label classes to use.")
    parser.add_argument("--target", type=str, default="data", help="Generate noise or data.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use.")
    parser.add_argument("--tau", type=int, default=10, help="Number of iterations.")
    parser.add_argument("--ds-size", type=int, default=500, help="Dataset size.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training.")

    return parser.parse_args(args)


def initial_log():
    log_path = './Logs/ray[tune]'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file_name = log_path + 'log-' + time.strftime("%m%d-%H%M}", time.localtime()) + '.log'
    sys.stdout = Logger(log_file_name)
    sys.stderr = Logger(log_file_name)


def train_fmnist28(config):

    args.lr = config["lr"]
    args.batch_size = config["batch_size"]
    args.epochs = config["epochs"]
    print(f"**************{args.batch_size}")
    hidden_features = config["hidden_features"]
    L = config["L"]
    N = config["N"]

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if args.device == "cuda":
        warnings.warn("CUDA performance is worse than CPU for most models.")
        if not torch.cuda.is_available():
            warnings.warn("CUDA is not available, using CPU.")
            args.device = "cpu"

    # Load dataset
    x_train, y_train, height, width = data.mnist_28x28(n_classes=args.n_classes, ds_size=args.ds_size)
    if args.label is not None:
        mask = y_train == args.label
        x_train = x_train[mask]

    reduced_size = int(len(x_train) * args.reduced_size)
    x_train = x_train[:reduced_size]
    x_train = x_train.to(args.device, dtype=torch.double)

    train_cutoff = int(len(x_train) * 0.8)
    x_train, x_test = x_train[:train_cutoff], x_train[train_cutoff:]

    if args.batch_size > len(x_train):
        args.batch_size = len(x_train)

    ds = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train), batch_size=args.batch_size, shuffle=True)

    net = QIDDM_PL_noise(28 * 28, hidden_features=hidden_features, spectrum_layer=L, N=N)
    # net = differN_noise(28, spectrum_layer=L, N=N)

    diff = models.Diffusion(net=net, noise_f=noise.add_normal_noise_multiple, prediction_goal=args.target,
                            shape=(height, width), loss=torch.nn.MSELoss()).to(args.device, dtype=torch.double)

    opt = torch.optim.Adam(diff.parameters(), lr=args.lr)
    loss_values = []

    for epoch in range(args.epochs):
        epoch_loss = torch.tensor(0.0, dtype=torch.double, device=args.device)
        for batch in ds:
            x = batch[0].to(args.device, dtype=torch.double)
            opt.zero_grad()
            result = diff(x=x, T=args.tau, verbose=False)
            if isinstance(result, tuple) and len(result) >= 1:
                batch_loss = result[0]  # 只取第一个值
            else:
                batch_loss = result  # 假设结果是单一值
            epoch_loss += batch_loss.mean()
            opt.step()
        loss_value = epoch_loss.item()
        loss_values.append(loss_value)


    # 测试模型
    first_x = torch.rand(15, 1, args.img_size, args.img_size, dtype=torch.double).to(args.device) * 0.75 + 0.5
    ssim_values = test(diff, x_test, first_x)

    # 保存模型
    sp = pathlib.Path(args.save_path) / f"{diff.save_name()}_{loss_values[-1]}_{ssim_values[-1]}.pt"
    if not sp.parent.exists():
        sp.parent.mkdir(parents=True)

    try:
        torch.save({
            'model_state_dict': diff.state_dict(),
            'loss_values': loss_values,
            'epochs': args.epochs
        }, sp)
        print(f"Model saved to {sp}")
    except Exception as e:
        print(f"Error saving model: {e}")


    tune.report(loss=loss_values[-1], ssim=ssim_values[-1])

    return {"loss": loss_values[-1], "ssim": ssim_values[-1]}


def test(diff, x_test, first_x):  # [0,255]

    diff.eval()
    tau_test = 5

    outp = diff.sample(first_x=first_x, n_iters=tau_test, show_progress=True, only_last=False)

    # 对输出进行处理以确保在 [0, 1] 的范围内
    outp = torch.clamp(outp, 0.0, 1)

    # 将输出放大到 [0, 255] 范围内
    outp = outp * 255.0
    outp = torch.clamp(outp, 0.0, 255.0)

    generated_images = einops.rearrange(
        outp, "(iters height) (batch width) -> iters batch 1 height width",
        iters=tau_test + 1, height=args.img_size, width=args.img_size
    )

    real_images = x_test.view(len(x_test), 1, args.img_size, args.img_size).double()

    # 对 real_images 进行等比例放大到 [0, 255]
    real_images_min = real_images.view(real_images.size(0), -1).min(dim=1, keepdim=True)[0]
    real_images_max = real_images.view(real_images.size(0), -1).max(dim=1, keepdim=True)[0]

    # 调整 real_images_min 和 real_images_max 的形状，使其与 real_images 兼容
    real_images_min = real_images_min.view(-1, 1, 1, 1)
    real_images_max = real_images_max.view(-1, 1, 1, 1)

    real_images = (real_images - real_images_min) / (real_images_max - real_images_min + 1e-7)  # 归一化到 [0, 1]
    real_images = real_images * 255.0
    real_images = torch.clamp(real_images, 0.0, 255.0)

    # 计算并保存 SSIM，使用最多 10 张生成图像和 4 张真实图像
    ssim_values = get_ssim_single(generated_images, real_images, args, gen_img_count=1, real_img_count=20)

    return ssim_values


if __name__ == "__main__":
    args = parse_args([])


    search_space = {
        "lr": tune.loguniform(1e-3, 1e-1),
        "batch_size": tune.choice([1]),  # 4 8 16
        "hidden_features": tune.choice([8]),  # 6-10
        "L": tune.choice([6]),  # 6-10
        "N": tune.choice([2]),
        "epochs": tune.choice([20])
    }
    shed = AsyncHyperBandScheduler(metric="ssim", mode="max", max_t=50)

    analysis = tune.run(
        train_fmnist28,
        config=search_space,
        num_samples=50,
        max_concurrent_trials=1,  # 同时运行的最大 trial 数量
        scheduler=shed,
        local_dir="tune_results",
        # verbose=1  # 打印
    )

    best_trial = analysis.get_best_trial("loss", "min", "last")  # 筛选指标为loss，最小的最好，最后一次迭代的loss
    best_ssim_trial = analysis.get_best_trial("ssim", "max", "last")

    print(f"Best loss_trial id: {best_trial}")

    print(f"Best loss_trial config: {best_trial.config}")
    print(f"Best loss_trial final validation loss: {best_trial.last_result['loss']}\n")

    print(f"Best ssim_trial id: {best_ssim_trial}")
    print(f"Best ssim_trial config: {best_ssim_trial.config}")
    print(f"Best loss_trial final validation ssim: {best_ssim_trial.last_result['ssim']}")
