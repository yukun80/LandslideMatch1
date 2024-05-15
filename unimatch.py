import argparse
import logging
import os
import pprint

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed

"""通过利用标记和未标记数据来处理完全监督和半监督设置。以下是其关键组件的详细分解：
该脚本首先解析命令行参数，例如配置文件路径 ( --config )、标记和未标记数据的路径以及用于保存模型和日志的输出路径。
然后，使用 yaml.load 函数加载配置文件，该配置文件包含模型架构、数据集和训练参数等信息。
接着，该脚本初始化日志记录器，并将命令行参数和配置文件信息记录到日志中。
然后，该脚本使用 torch.backends.cudnn.enabled 和 torch.backends.cudnn.benchmark 来启用 cuDNN 加速。
然后，该脚本实例化模型、优化器和损失函数。
接着，该脚本实例化训练集、验证集和数据加载器。
最后，该脚本使用 torch.nn.SyncBatchNorm.convert_sync_batchnorm 函数将 BatchNorm 转换为同步 BatchNorm，并使用 torch.nn.parallel.DistributedDataParallel 将模型分布到多个 GPU 上。
"""
parser = argparse.ArgumentParser(description="遥感半监督实验cvpr2023")
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--labeled-id-path", type=str, required=True)
parser.add_argument("--unlabeled-id-path", type=str, required=True)
parser.add_argument("--save-path", type=str, required=True)
parser.add_argument("--local_rank", default=0, type=int)  # 分布式系统中的排名
parser.add_argument("--port", default=None, type=int)


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        """
        为了确保在多个 GPU 上训练时，每个 GPU 都有自己的日志记录器和 TensorBoard 记录器，我们需要在主进程中初始化它们。
        这会检查当前进程的等级是否为 0。在分布式训练中，每个进程都会分配一个等级。 Rank 0 通常保留用于只应执行一次而不是由每个进程执行的任务，例如记录日志和写入文件。
        """
        all_args = {**cfg, **vars(args), "ngpus": world_size}
        logger.info("{}\n".format(pprint.pformat(all_args)))

        writer = SummaryWriter(args.save_path)

        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    """模型实例，特别是 DeepLabV3Plus ，是根据提供的配置创建的。模型的优化器也在这里配置，骨干网和其他参数具有不同的学习率。"""
    model = DeepLabV3Plus(cfg)
    optimizer = SGD(
        [
            {"params": model.backbone.parameters(), "lr": cfg["lr"]},
            {"params": [param for name, param in model.named_parameters() if "backbone" not in name], "lr": cfg["lr"] * cfg["lr_multi"]},
        ],
        lr=cfg["lr"],
        momentum=0.9,
        weight_decay=1e-4,
    )

    if rank == 0:
        logger.info("Total params: {:.1f}M\n".format(count_params(model)))

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    """torch.nn.parallel.DistributedDataParallel 函数将模型分布到多个 GPU 上。它需要以下参数："""
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], broadcast_buffers=False, output_device=local_rank, find_unused_parameters=False
    )
    """损失函数和数据集设置。我们使用交叉熵损失函数和 ProbOhemCrossEntropy2d 损失函数。"""
    if cfg["criterion"]["name"] == "CELoss":
        criterion_l = nn.CrossEntropyLoss(**cfg["criterion"]["kwargs"]).cuda(local_rank)
    elif cfg["criterion"]["name"] == "OHEM":
        criterion_l = ProbOhemCrossEntropy2d(**cfg["criterion"]["kwargs"]).cuda(local_rank)
    else:
        raise NotImplementedError("%s criterion is not implemented" % cfg["criterion"]["name"])

    """损失函数根据配置设置。对于标记数据，使用标准或 OHEM（在线硬示例挖掘）交叉熵损失。
    对于未标记的数据，使用不允许减少的简单交叉熵损失来单独处理每个实例。"""
    criterion_u = nn.CrossEntropyLoss(reduction="none").cuda(local_rank)
    """数据集设置。我们使用 SemiDataset 类来处理半监督学习数据集。"""
    trainset_u = SemiDataset(cfg["dataset"], cfg["data_root"], "train_u", cfg["crop_size"], args.unlabeled_id_path)
    trainset_l = SemiDataset(cfg["dataset"], cfg["data_root"], "train_l", cfg["crop_size"], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg["dataset"], cfg["data_root"], "val")

    """数据加载器设置。我们使用 torch.utils.data.DataLoader 类来加载数据集。"""
    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg["batch_size"], pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg["batch_size"], pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, sampler=valsampler)

    total_iters = len(trainloader_u) * cfg["epochs"]
    previous_best = 0.0
    epoch = -1

    """加载检查点。如果存在检查点，则加载模型和优化器状态。"""
    if os.path.exists(os.path.join(args.save_path, "latest.pth")):
        checkpoint = torch.load(os.path.join(args.save_path, "latest.pth"))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = checkpoint["epoch"]
        previous_best = checkpoint["previous_best"]

        if rank == 0:
            logger.info("************ Load from checkpoint at epoch %i\n" % epoch)

    """训练循环。在每个 epoch 中，我们遍历标记和未标记数据加载器，并在每个迭代中计算损失并执行反向传播。"""
    for epoch in range(epoch + 1, cfg["epochs"]):
        if rank == 0:
            logger.info("===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}".format(epoch, optimizer.param_groups[0]["lr"], previous_best))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_w_fp = AverageMeter()
        total_mask_ratio = AverageMeter()
        """设置每个 epoch 的随机种子。这确保了每个 epoch 的数据加载器都会生成相同的随机序列。"""
        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)
        """将标记和未标记数据加载器打包在一起。这样，我们可以在每个迭代中同时处理标记和未标记数据。"""
        loader = zip(trainloader_l, trainloader_u, trainloader_u)

        for i, (
            (img_x, mask_x),
            (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2),
            (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, ignore_mask_mix, _, _),
        ) in enumerate(loader):
            """将数据移动到 GPU 上。"""
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2, ignore_mask = img_u_s1.cuda(), img_u_s2.cuda(), ignore_mask.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()

            with torch.no_grad():
                model.eval()

                pred_u_w_mix = model(img_u_w_mix).detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

            model.train()

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            preds, preds_fp = model(torch.cat((img_x, img_u_w)), True)
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            pred_u_w_fp = preds_fp[num_lb:]

            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2))).chunk(2)

            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]

            loss_x = criterion_l(pred_x, mask_x)

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= cfg["conf_thresh"]) & (ignore_mask_cutmixed1 != 255))
            loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()

            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= cfg["conf_thresh"]) & (ignore_mask_cutmixed2 != 255))
            loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()

            loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
            loss_u_w_fp = loss_u_w_fp * ((conf_u_w >= cfg["conf_thresh"]) & (ignore_mask != 255))
            loss_u_w_fp = loss_u_w_fp.sum() / (ignore_mask != 255).sum().item()

            loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5) / 2.0

            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
            total_loss_w_fp.update(loss_u_w_fp.item())

            mask_ratio = ((conf_u_w >= cfg["conf_thresh"]) & (ignore_mask != 255)).sum().item() / (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i
            lr = cfg["lr"] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg["lr_multi"]

            if rank == 0:
                writer.add_scalar("train/loss_all", loss.item(), iters)
                writer.add_scalar("train/loss_x", loss_x.item(), iters)
                writer.add_scalar("train/loss_s", (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters)
                writer.add_scalar("train/loss_w_fp", loss_u_w_fp.item(), iters)
                writer.add_scalar("train/mask_ratio", mask_ratio, iters)

            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info(
                    "Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss w_fp: {:.3f}, Mask ratio: "
                    "{:.3f}".format(i, total_loss.avg, total_loss_x.avg, total_loss_s.avg, total_loss_w_fp.avg, total_mask_ratio.avg)
                )

        eval_mode = "sliding_window" if cfg["dataset"] == "cityscapes" else "original"
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg)

        if rank == 0:
            for cls_idx, iou in enumerate(iou_class):
                logger.info("***** Evaluation ***** >>>> Class [{:} {:}] " "IoU: {:.2f}".format(cls_idx, CLASSES[cfg["dataset"]][cls_idx], iou))
            logger.info("***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n".format(eval_mode, mIoU))

            writer.add_scalar("eval/mIoU", mIoU, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar("eval/%s_IoU" % (CLASSES[cfg["dataset"]][i]), iou, epoch)

        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)
        if rank == 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "previous_best": previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, "latest.pth"))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, "best.pth"))


if __name__ == "__main__":
    main()
