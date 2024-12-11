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
from nn import QDenseUndirected_old_noise, QNN_A, QNN_noise, differN_noise, QIDDM_PL_noise, UNetUndirected, QIDDM_L

all_nn = [name for name, obj in inspect.getmembers(nn) if inspect.isclass(obj)]
all_ds = [
    name
    for name, obj in inspect.getmembers(data)
    if inspect.isfunction(obj) and not name.startswith("_")
]


# "参数"：model；data; load_path; save_path; show_img；first_x
def parse_args(args):
    parser = argparse.ArgumentParser(description="Quantum Denoising Diffusion Model")
    parser.add_argument(
        "--model",
        type=str,
        default=[
            ["UNetUndirected", "3", "6", "0"],
            ["differN_noise", 28, "15", "2"],
            # ["differN_noise", 28, "9", "2"],
            ["QDenseUndirected_old_noise", "60", "28"],
            ["QIDDM_PL_noise", 28 * 28, "8", "6", "2"],
            ["QNN_noise", 28 * 28, "8", "6"],

        ],
        nargs="+",
        help=f"Model name and parameters. \
                Models are defined in the nn module, including {', '.join(all_nn)}.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="mnist_28x28",
        help=f"Dataset to use. Available datasets: {', '.join(all_ds)}.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=28,
        help=f"Dataset to use. Available datasets: {', '.join(all_ds)}.",
    )
    parser.add_argument(
        "--label",
        type=int,
        default=0,
        help="Specify the label to be used for training."
    )
    parser.add_argument(
        "--add_noise",
        type=int,
        default=0,
        help="Specify the type to be used for adding noise.(1-3)"
    )
    parser.add_argument(
        "--reduced_size",
        type=float,
        default=1.0,
        help=""
    )
    parser.add_argument(
        "--load-path",
        type=str,
        default="results/formal/mnist_",
        help="Load model from path. \
                If no path is given, train a new model.\
                The trained model will be saved in --save-path.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="results/formal/mnist_",
        help="Path to save results."
    )
    parser.add_argument(
        "--n_classes",
        type=int,
        default=10,
        help="Number of label classes to use. \
                        Smaller models perform better on a smaller number of classes.",
    )
    parser.add_argument(
        "--target", type=str, default="noise", help="Generate noise or data."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use.",
    )
    parser.add_argument(
        "--tau",
        type=int,
        default=10,
        help="Number of iterations. \
            Models perform better with more iterations on higher resolution images, \
            for low-res, tau=10 suffices.",
    )
    parser.add_argument("--ds-size", type=int, default=5000, help="Dataset size. 80% is used for training.")
    # 所有预模型学习率i
    # u-net
    parser.add_argument("--UNetUndirected_lr", type=float, default=0.00637, help="Learning rate.")
    # differn
    parser.add_argument("--differN_noise_lr", type=float, default=0.0411, help="Learning rate.") #0.04587
    # qdense
    parser.add_argument("--QDenseUndirected_old_noise_lr", type=float, default=0.00211, help="Learning rate.")
    # pl
    parser.add_argument("--QIDDM_PL_noise_lr", type=float, default=0.01116, help="Learning rate.")
    # qnn
    parser.add_argument("--QNN_noise_lr", type=float, default=0.01011, help="Learning rate.")

    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for traning.")
    return parser.parse_args(args)


def train():
    print("Training model")
    # 根据模型类型设置训练时的设备和微分方法
    # if isinstance(diff.net, (QDenseUndirected_old_noise, QNN_A, differN_noise)):
    #     diff.net.device_type = "default.qubit.torch"  # QDenseUndirected_old_noise 的训练设备
    #     diff.net.diff_method = "backprop"  # 使用 backprop
    #     # 使用 QDenseUndirected_old_noise 特定的 wires 作为量子比特数
    #     diff.net.qdev = qml.device(diff.net.device_type, wires=diff.net.wires)
    # else:
    #     diff.net.device_type = "lightning.qubit"  # 其他模型的训练设备
    #     diff.net.diff_method = "parameter-shift"  # 使用 parameter-shift
    #     # 使用其他模型的 hidden_features 作为量子比特数
    #     diff.net.qdev = qml.device(diff.net.device_type, wires=diff.net.hidden_features)
    #
    # # 设置噪声为 0
    # diff.net.add_noise = 0

    # 配置量子节点
    # diff.net.qnode = qml.QNode(diff.net._circuit, diff.net.qdev, interface="torch", diff_method=diff.net.diff_method)

    diff.train()
    pbar = tqdm.tqdm(total=args.epochs - start_epoch, ascii="⬛⬜", dynamic_ncols=True)  # ,  ncols=150, nrows=1
    opt = torch.optim.Adam(diff.parameters(), lr=args.lr)

    if args.epochs - start_epoch <= 0:
        return loss_values

    for epoch in range(args.epochs - start_epoch):
        epoch_loss = torch.tensor(0.0, dtype=torch.double, device=args.device)
        for batch in ds:
            x = batch[0].to(args.device, dtype=torch.double)
            opt.zero_grad()
            batch_loss, _ = diff(x=x, T=args.tau, verbose=True)
            epoch_loss += batch_loss.mean()
            opt.step()
        pbar.set_postfix({"loss": epoch_loss.item()})  # type: ignore
        loss_value = epoch_loss.item()
        loss_values.append(loss_value)
        pbar.update(1)
    pbar.close()

    sp = pathlib.Path(args.save_path) / f"{diff.save_name()}_{args.label}.pt"
    if not sp.parent.exists():
        sp.parent.mkdir(parents=True)

    # 保存整个 diff 模型，包括模型参数和量子电路状态
    if isinstance(diff.net, QIDDM_L):
        diff.net.save_model(sp, loss_values, args.epochs)
    else:
        torch.save({
            'model_state_dict': diff.state_dict(),
            'loss_values': loss_values,
            'epochs': args.epochs
        }, sp)

    return loss_values


def test(add_noise):  # [0,255]
    print("Testing model")
    # 测试时统一使用 default.mixed 和 backprop
    # if add_noise != 0:
    #     diff.net.device_type = "default.mixed"
    #     diff.net.diff_method = "backprop"
    #
    # # 设置噪声类型参数，1-3
    # diff.net.add_noise = add_noise
    #
    # # 根据模型类型选择适当的 wires 参数
    # if isinstance(diff.net, (QDenseUndirected_old_noise, QNN_A, differN_noise)):
    #     diff.net.qdev = qml.device(diff.net.device_type, wires=diff.net.wires)
    # else:
    #     diff.net.qdev = qml.device(diff.net.device_type, wires=diff.net.hidden_features)
    #
    # # 配置量子节点
    # diff.net.qnode = qml.QNode(diff.net._circuit, diff.net.qdev, interface="torch", diff_method=diff.net.diff_method)

    diff.eval()
    tau_test = args.tau * 2
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

    # 保存训练集图像，保存生成图像
    # image_0_path = pathlib.Path(args.save_path) / "image_0"
    # image_0_path.mkdir(parents=True, exist_ok=True)
    #
    # for i in range(x_train.size(0)):  # 遍历训练集中的每张图像
    #     img_path = image_0_path / f"train_image_{i + 1}.png"
    #     plt.imsave(img_path, x_train[i].cpu().view(args.img_size, args.img_size).numpy(), cmap="gray")
    #
    # # # 保存图像到不同的文件夹
    # for i in range(generated_images.size(1)):  # 遍历每张图像
    #     folder_path = pathlib.Path(args.save_path) / f"image_{i + 1}"
    #     folder_path.mkdir(parents=True, exist_ok=True)
    #
    #     for j in range(generated_images.size(0)):  # 遍历去噪的每一步
    #         img_path = folder_path / f"step_{j + 1}.png"
    #         plt.imsave(img_path, generated_images[j, i, 0].cpu().numpy(), cmap="gray")

    # 显示并保存全部图片
    plt.imshow(outp.cpu().double(), cmap="gray")  #
    plt.axis("off")
    sp = pathlib.Path(args.save_path) / f"{diff.save_name()}_{args.label}.png"
    if not sp.parent.exists():
        sp.parent.mkdir(parents=True)

    plt.savefig(sp)

    return generated_images, real_images


def load_model(diff, load_path, label):
    """
    加载模型的状态和 PCA 状态（如果存在）。
    :param diff: 模型对象
    :param load_path: 模型文件的路径
    :param label: 模型对应的标签
    """
    try:
        # 确定模型文件路径
        if load_path.endswith(".pt"):
            lp = load_path
        else:
            lp = pathlib.Path(load_path) / f"{diff.save_name()}_{label}.pt"

        if torch.cuda.is_available():
            checkpoint = torch.load(lp, weights_only=True)
        else:
            checkpoint = torch.load(lp, map_location=torch.device('cpu'), weights_only=True)


        # 如果是 QIDDM，加载 PCA 状态
        if isinstance(diff.net, QIDDM_L):
            diff.net.load_model(lp)
        else:
            diff.load_state_dict(checkpoint['model_state_dict'])

        # 加载成功返回
        print("Model loaded successfully.\n")
        return checkpoint['loss_values'], checkpoint['epochs']

    except FileNotFoundError:
        print("Failed to load model: File not found.\n")
        return [], 0


def initial_log():
    log_path = './Logs/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file_name = log_path + 'log-' + time.strftime("%Y%m%d-%H%M", time.localtime()) + '.log'
    sys.stdout = Logger(log_file_name)
    sys.stderr = Logger(log_file_name)


if __name__ == "__main__":
    initial_log()
    args = parse_args(sys.argv[1:])

    # 初始化柱状图分数字典
    ssim_socre_dict = {}
    psnr_socre_dict = {}
    cos_socre_dict = {}
    fid_socre_dict = {}

    for model_args in args.model:
        model_name = model_args[0]
        ssim_socre_dict[model_name] = []
        psnr_socre_dict[model_name] = []
        cos_socre_dict[model_name] = []
        fid_socre_dict[model_name] = []

    original_save_path = args.save_path
    original_load_path = args.load_path

    for label in range(0, 1):
        args.label = label
        print(args)

        # 恢复路径
        args.save_path = original_save_path
        args.load_path = original_load_path

        # 默认路径
        noise_save_path = args.save_path + str(args.label) + "/noise_"
        noise_load_path = args.load_path + str(args.label) + "/noise_"

        args.save_path = noise_save_path + str(0)
        args.load_path = noise_load_path + str(0)

        if args.seed is not None:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)

        # Load dataset
        x_train, y_train, height, width = eval(f"data.{args.data}")(
            n_classes=args.n_classes, ds_size=args.ds_size
        )

        # Filter dataset by label if specified
        if args.label is not None:
            mask = y_train == args.label
            x_train = x_train[mask]
            y_train = y_train[mask]

        # 调整数据集大小
        reduced_size = int(len(x_train) * args.reduced_size)
        x_train = x_train[:reduced_size]

        print(f"description of dataset: len of x_train: {x_train.shape}\n")

        #  参数数据类型切换
        x_train = x_train.to(args.device, dtype=torch.double)

        # 设定训练集大小，生成高斯分布
        train_cutoff = int(len(x_train) * 0.8)
        x_train, x_test = x_train[:train_cutoff], x_train[train_cutoff:]
        first_x = torch.rand(10, 1, args.img_size, args.img_size, dtype=torch.double).to(args.device) * 0.75 + 0.5

        # 自动调整 batch_size 以适应数据集大小
        if args.batch_size > len(x_train):
            print(
                f"Warning: batch size ({args.batch_size}) is bigger than the data size ({len(x_train)}). Setting batch size to data size.")
            args.batch_size = len(x_train)

        ds = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_train),
            batch_size=args.batch_size,
            shuffle=True,
        )

        for add_noise in range(0, 1):
            print(f"Test for add_noise:{add_noise}")
            args.save_path = noise_save_path + str(add_noise)

            # 结果字典
            generated_images_dict = {}
            real_images_dict = {}
            loss_dict = {}
            # 动态实例化模型
            for model_args in args.model:
                model_name = model_args[0]
                model_params = [
                    int(a) if isinstance(a, str) and a.isdigit() else a for a in model_args[1:]
                ]
                net = eval(f"nn.{model_name}")(*model_params)

                # 根据模型切换设备cuda和学习率，数据
                if isinstance(net, (QIDDM_PL_noise, QNN_noise,)):
                    args.device = "cuda"
                else:
                    args.device = "cpu"

                x_train = x_train.to(args.device)
                x_test = x_test.to(args.device)
                first_x = first_x.to(args.device)

                args.lr = getattr(args, f"{model_name}_lr")

                # 打印模型以确认
                print(f"\nInitialized {model_name} with parameters {model_params}, with lr {args.lr}")

                diff = models.Diffusion(
                    net=net,
                    noise_f=noise.add_normal_noise_multiple,
                    prediction_goal=args.target,
                    shape=(height, width),
                    loss=torch.nn.MSELoss(),
                ).to(args.device, dtype=torch.double)
                print('parameters:%d\n' % (sum(p.numel() for p in diff.parameters() if p.requires_grad)))

                start_epoch = 0
                # 加载模型
                if args.load_path is not None:
                    print("Loading model")
                    loss_values, start_epoch = load_model(diff, args.load_path, args.label)

                print(f"epoch start from {start_epoch}, left {args.epochs - start_epoch}")
                loss_values = train()

                loss_dict[model_name] = loss_values

                generated_images, real_images = test(add_noise)
                generated_images_dict[f"{diff.save_name()}"] = generated_images
                real_images_dict[f"{diff.save_name()}"] = real_images

            # 显示并保存损失曲线
            show_metrics(loss_dict, "LOSS", args, model_name=model_name, model_params=model_params)

            # 计算并保存 SSIM，使用最多 10 张生成图像和 4 张真实图像
            ssim_values_dict = get_ssim(generated_images_dict, real_images_dict, args, gen_img_count=1,
                                        real_img_count=20)

            # 计算并保存 PSNR，使用最多 10 张生成图像和 4 张真实图像
            psnr_values_dict = get_psnr(generated_images_dict, real_images_dict, args, gen_img_count=1,
                                        real_img_count=20)

            # 计算并保存余弦相似度，使用最多 10 张生成图像和 4 张真实图像
            cos_values_dict = get_cosine_similarity(generated_images_dict, real_images_dict, args, gen_img_count=1,
                                                    real_img_count=20)
            # #
            # # # 计算并保存 fid，使用最多 10 张生成图像和 4 张真实图像
            fid_values_dict = get_fid(generated_images_dict, real_images_dict, args, gen_img_count=1,
                                      real_img_count=20)

            keys1 = list(ssim_socre_dict.keys())
            keys2 = list(ssim_values_dict.keys())

            for model_name, diff_name in zip(keys1, keys2):
                ssim_socre_dict[model_name].append(ssim_values_dict[diff_name][-1])
                psnr_socre_dict[model_name].append(psnr_values_dict[diff_name][-1])
                cos_socre_dict[model_name].append(cos_values_dict[diff_name][-1])
                fid_socre_dict[model_name].append(fid_values_dict[diff_name][-1])

    show_histogram(ssim_socre_dict, "SSIM", args)
    show_histogram(psnr_socre_dict, "PSNR", args)
    show_histogram(cos_socre_dict, "COS", args)
    show_histogram(fid_socre_dict, "FID", args)

