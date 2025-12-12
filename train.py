import os
import torch
import pytorch_lightning as pl

from vit import ViT
from config import config
from dataset import ImageNetDataModule

def get_tiny_imagenet_classes(root: str):
    train_root = os.path.join(root, "train")
    classes = sorted([d for d in os.listdir(train_root)
                      if os.path.isdir(os.path.join(train_root, d))])
    return classes

def main():
    root = "./data/tiny-imagenet-200"

    classes = get_tiny_imagenet_classes(root)
    assert len(classes) == config["class_number"], \
        f"Ожидается {config['class_number']} классов, найдено {len(classes)}"

    dm = ImageNetDataModule(
        root=root,
        classes=classes,
        batch_size=128,
        num_workers=4,
    )

    model = ViT(config)

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="./checkpoints/vit-tinyimagenet-{epoch:02d}-{val_acc:.4f}",
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        callbacks=[checkpoint_cb, lr_monitor],
        log_every_n_steps=50,
    )

    trainer.fit(model, datamodule=dm)

    # best_path = checkpoint_cb.best_model_path
    # best_model = ViT.load_from_checkpoint(best_path, cfg=config)
    # trainer.validate(best_model, datamodule=dm)

if __name__ == "__main__":
    main()
