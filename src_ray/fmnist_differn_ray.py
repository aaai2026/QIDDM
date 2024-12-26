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
from nn import QDenseUndirected_old_noise, QNN_A, QNN, differN_noise, QIDDM_PL_noise, UNetUndirected, QIDDM_PL

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
    parser.add_argument("--data", type=str, default="fashion_28x28",
    # parser.add_argument("--data", type=str, default="mnist_8x8",
    # parser.add_argument("--data", type=str, default="lfw_64x64",
    # parser.add_argument("--data", type=str, default="emnist_28x28",
                        help=f"Dataset to use. Available datasets: {', '.join(all_ds)}.")
    parser.add_argument("--img_size", type=int, default=28, help="Image size.")
    parser.add_argument("--label", type=int, default=4, help="Specify the label to be used for training.")
    parser.add_argument("--reduced_size", type=float, default=1, help="Reduced dataset size ratio.")
    parser.add_argument("--load-path", type=str, default="results/ray_", help="Load model path.")
    parser.add_argument("--save-path", type=str, default="results/ray_", help="Path to save results.")
    parser.add_argument("--n_classes", type=int, default=10, help="Number of label classes to use.")
    parser.add_argument("--target", type=str, default="data", help="Generate noise or data.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device to use.")
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

def train_with_tune(config):
    args.lr = config["lr"]
    args.batch_size = config["batch_size"]
    args.epochs = config["epochs"]
    # print(f"**************{args.batch_size}")
    # hidden_features = config["hidden_features"]
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
    # torch.cuda.set_device(0)
    # args.device = "cuda:0"
    # Load dataset
    x_train, y_train, height, width = data.fashion_28x28(n_classes=args.n_classes, ds_size=args.ds_size)
    if args.label is not None:
        mask = y_train == args.label
        x_train = x_train[mask]

    reduced_size = int(len(x_train) * args.reduced_size)
    x_train = x_train[:reduced_size]
    x_train = x_train.to(args.device, dtype=torch.double)  # Use double for better precision

    train_cutoff = int(len(x_train) * 0.8)
    x_train, x_test = x_train[:train_cutoff], x_train[train_cutoff:]

    if args.batch_size > len(x_train):
        args.batch_size = len(x_train)

    ds = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train), batch_size=args.batch_size, shuffle=True)

    # Ensure the model is on the correct device and use double precision
    # net = QIDDM_PL_noise(28* 28, hidden_features=hidden_features, spectrum_layer=L, N=N).to(args.device, dtype=torch.double)
    # # net = UNetUndirected(3, 8, 0).to(args.device, dtype=torch.double)
    # net =  QDenseUndirected_old_noise(60, 28).to(args.device, dtype=torch.double)
    net = differN_noise(28, spectrum_layer=L, N=N).to(args.device, dtype=torch.double)
    # net = QNN(28*28, hidden_features, L).to(args.device, dtype=torch.double)
    diff = models.Diffusion(net=net, noise_f=noise.add_normal_noise_multiple, prediction_goal=args.target,
                            shape=(height, width), loss=torch.nn.MSELoss()).to(args.device, dtype=torch.double)

    opt = torch.optim.Adam(diff.parameters(), lr=args.lr)
    loss_values = []

    for epoch in range(args.epochs):
        epoch_loss = torch.tensor(0.0, dtype=torch.double, device=args.device)
        for batch in ds:
            x = batch[0].to(args.device, dtype=torch.double)  # Ensure the batch is also on the correct device
            opt.zero_grad()
            result = diff(x=x, T=args.tau, verbose=False)
            if isinstance(result, tuple) and len(result) >= 1:
                batch_loss = result[0]  # Only take the first value
            else:
                batch_loss = result  # Assume the result is a single value
            epoch_loss += batch_loss.mean()
            opt.step()
        loss_value = epoch_loss.item()
        loss_values.append(loss_value)

    # Save the model
    sp = pathlib.Path(args.save_path) / f"{diff.save_name()}_{loss_values[-1]}.pt"
    if not sp.parent.exists():
        sp.parent.mkdir(parents=True)

    # if isinstance(diff.net, QIDDM_L):
    #     diff.net.save_model(sp, loss_values, args.epochs)
    # else:
    torch.save({
        'model_state_dict': diff.state_dict(),
        'loss_values': loss_values,
        'epochs': args.epochs
    }, sp)

    # Test the model
    first_x = torch.rand(10, 1, args.img_size, args.img_size, dtype=torch.double).to(args.device) * 0.75 + 0.5
    ssim_values = test(diff, x_test, first_x)

    # tune.report(loss=loss_values[-1], ssim=ssim_values[-1])

    return {"loss": loss_values[-1], "ssim": ssim_values[-1]}

def test(diff, x_test, first_x):  # [0,255]
    diff.eval()
    tau_test = 5

    outp = diff.sample(first_x=first_x, n_iters=tau_test, show_progress=True, only_last=False)

    # Ensure output is in the [0, 1] range
    outp = torch.clamp(outp, 0.0, 1)

    # Scale output to [0, 255]
    outp = outp * 255.0
    outp = torch.clamp(outp, 0.0, 255.0)

    generated_images = einops.rearrange(
        outp, "(iters height) (batch width) -> iters batch 1 height width",
        iters=tau_test + 1, height=args.img_size, width=args.img_size
    )

    real_images = x_test.view(len(x_test), 1, args.img_size, args.img_size).double()

    # Scale real images to [0, 255]
    real_images_min = real_images.view(real_images.size(0), -1).min(dim=1, keepdim=True)[0]
    real_images_max = real_images.view(real_images.size(0), -1).max(dim=1, keepdim=True)[0]

    # Adjust the shape of real_images_min and real_images_max to be compatible with real_images
    real_images_min = real_images_min.view(-1, 1, 1, 1)
    real_images_max = real_images_max.view(-1, 1, 1, 1)

    real_images = (real_images - real_images_min) / (real_images_max - real_images_min + 1e-7)  # Normalize to [0, 1]
    real_images = real_images * 255.0
    real_images = torch.clamp(real_images, 0.0, 255.0)

    # Calculate SSIM, using up to 10 generated images and 20 real images
    ssim_values = get_ssim_single(generated_images, real_images, args, gen_img_count=1, real_img_count=10)

    return ssim_values

if __name__ == "__main__":
    args = parse_args([])

    args.save_path = args.save_path + str(args.label)
    args.load_path = args.load_path + str(args.label)

    search_space = {
        "lr": tune.loguniform(1e-3, 1e-1),0.001-0.099
        "batch_size": tune.choice([1]),  # 4 8 16
        # "hidden_features": tune.choice([8]),  # 6-10
        # "hidden_features": tune.choice([6,7,8,9,10]),  # 6-10
        # "L": tune.choice([15]),  # 6-10
        # "L": tune.choice([6,7,8,9,10,11,12]),  # 6-10
        "L": tune.choice([15]),  # 6-10
        "N": tune.choice([2]),
        "epochs": tune.choice([20])
    }

    analysis = tune.run(
        train_with_tune,
        config=search_space,
        num_samples=50,#50
        max_concurrent_trials=4,  # Max number of concurrent trials
        resources_per_trial={"cpu": 5, "gpu": 1},  # 每个 trial 使用 1 个 GPU
        scheduler=AsyncHyperBandScheduler(metric="loss", mode="min", max_t=50),
    )

    best_trial = analysis.get_best_trial("loss", "min", "last")  # Best trial based on min loss
    best_ssim_trial = analysis.get_best_trial("ssim", "max", "last")

    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")

    print(f"Best ssim_trial config: {best_ssim_trial.config}")
    print(f"Best trial final validation ssim: {best_ssim_trial.last_result['ssim']}")
