from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as T
import pytorch_lightning as pl
from PIL import Image
import torch
import os



class ImageNetDataset(Dataset):
    def __init__(self, root: str, classes: list[str], ds_type: str, transform = None) -> None:
        super().__init__()
        self.ds_type = ds_type
        self.root = root
        self.classes = classes
        if ds_type == 'val':
            self.prepareValData()
        self.transform = transform
        self.label_to_idx = {label : idx for idx, label in enumerate(classes)}
        
    def prepareValData(self) -> None:
        self.valData = []
        labelsSet = set(self.classes)
        with open(os.path.join(self.root, self.ds_type, "val_annotations.txt")) as f:
            rows = f.readlines()
            f.close()    
        for row in rows:
            data = self._parseLine(row)
            label = data[1]
            if label in labelsSet:
                self.valData.append(tuple(data))
    
    def _parseLine(self, row: str):
        els = row.split("\t")
        data = [els[k].strip("\n") for k in range(len(els))]
        if self.ds_type == 'val':
            start_idx = 2
        else:
            start_idx = 1
        for i in range(start_idx, len(data)):
            data[i] = int(data[i])
        return data
    
    def __getitem__(self, index: int):
        if self.ds_type == 'val':
            data = self.valData[index]
            img_path = os.path.join(self.root, self.ds_type, "images", data[0])
            label = data[1]
        else:
            label_idx = index // 500
            img_idx = index % 500
            label = self.classes[label_idx]
            label_root = os.path.join(self.root, self.ds_type, label)
            img_path = os.path.join(label_root, "images", f'{label}_{img_idx}.JPEG')
            with open(os.path.join(label_root, f"{label}_boxes.txt")) as f:
                rows = f.readlines()
                f.close()
            row = rows[img_idx]
            data = self._parseLine(row)
        
        img = Image.open(img_path).convert('RGB')
            
        if self.transform:
            img = self.transform(img)
        return (img, torch.tensor(self.label_to_idx[label]))
        
    def __len__(self) -> int:
        #500 examples for each class
        if self.ds_type == 'train':
            return len(self.classes) * 500
        else:
            return len(self.valData)

class ImageNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        classes: list[str],
        batch_size: int = 128,
        num_workers: int = 4,
    ):
        super().__init__()
        self.root = root
        self.classes = classes
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = T.Compose([
            T.RandomResizedCrop(64),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        self.val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def setup(self, stage=None):
        if stage in ("fit", None):
            self.train_ds = ImageNetDataset(
                root=self.root,
                classes=self.classes,
                ds_type="train",
                transform=self.train_transform,
            )
            self.val_ds = ImageNetDataset(
                root=self.root,
                classes=self.classes,
                ds_type="val",
                transform=self.val_transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
