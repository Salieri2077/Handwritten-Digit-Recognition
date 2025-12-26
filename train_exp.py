import argparse
import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from CNN.densenet_mnist import DenseNetBC_MNIST
from CNN.utils import (get_dataloaders, human_time, predict_one_image,
                       save_checkpoint, set_seed, train_one_epoch,
                       evaluate)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DenseNet-MNIST Training Script")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training and testing")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--out_dir", type=str, default="./checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--save_name", type=str, default="densenet_mnist.pth", help="Checkpoint file name")
    parser.add_argument("--blocks", type=str, default="8,12,16", help="Comma-separated block configuration for DenseNet")
    parser.add_argument("--growth_rate", type=int, default=12, help="Growth rate for DenseNet")
    parser.add_argument("--init_channels", type=int, default=24, help="Initial number of channels for DenseNet")
    parser.add_argument("--bn_size", type=int, default=4, help="Bottleneck size for DenseNet")
    parser.add_argument("--theta", type=float, default=0.5, help="Compression factor for DenseNet")
    parser.add_argument("--drop_rate", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing factor")
    parser.add_argument("--no_amp", action="store_true", help="Disable automatic mixed precision training")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to a checkpoint to load")
    parser.add_argument("--predict", type=str, default="", help="Path to an image for prediction mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = (device.type == "cuda") and (not args.no_amp)

    block_config = tuple(int(x.strip()) for x in args.blocks.split(",") if x.strip())
    if len(block_config) != 3:
        raise ValueError("--blocks must have exactly 3 integers like '8,12,16' for MNIST adaptation.")

    model = DenseNetBC_MNIST(
        growth_rate=args.growth_rate,
        block_config=block_config,  # (b1,b2,b3)
        init_channels=args.init_channels,
        bn_size=args.bn_size,
        theta=args.theta,
        drop_rate=args.drop_rate,
        num_classes=10,
    ).to(device)

    # Optional checkpoint load
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state"], strict=True)

    # If predict mode
    if args.predict:
        model.eval()
        pred, probs = predict_one_image(model, args.predict, device)
        topk = torch.topk(probs, k=3)
        print(f"Prediction: {pred}")
        print("Top-3 probs:")
        for score, cls in zip(topk.values.tolist(), topk.indices.tolist()):
            print(f"  {cls}: {score:.4f}")
        return

    train_loader, test_loader = get_dataloaders(args.batch_size, args.num_workers)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_acc = -1.0
    save_path = os.path.join(args.out_dir, args.save_name)

    print("=== Config ===")
    print(f"device={device} amp={amp_enabled} seed={args.seed}")
    print(f"blocks={block_config} growth_rate={args.growth_rate} init_channels={args.init_channels} "
          f"bn_size={args.bn_size} theta={args.theta} drop_rate={args.drop_rate}")
    print(f"epochs={args.epochs} batch_size={args.batch_size} lr={args.lr} wd={args.weight_decay}")
    print(f"checkpoint_out={save_path}")
    print("==============")

    start_all = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        improved = te_acc > best_acc
        if improved:
            best_acc = te_acc
            save_checkpoint(save_path, model, optimizer, epoch, best_acc)

        flag = " (best)" if improved else ""
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"lr {lr_now:.2e} | "
            f"train loss {tr_loss:.4f} acc {tr_acc*100:.2f}% | "
            f"test loss {te_loss:.4f} acc {te_acc*100:.2f}% | "
            f"time {human_time(elapsed)}{flag}"
        )

    total = time.time() - start_all
    print(f"Done. Best test acc: {best_acc*100:.2f}% | total time: {human_time(total)}")
    print(f"Best checkpoint saved to: {save_path}")


if __name__ == "__main__":
    main()