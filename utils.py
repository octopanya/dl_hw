import os

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from PIL import Image
from matplotlib import gridspec
from torch.utils.data import Dataset
from torchvision import transforms


class WeizmannHorsesDataset(Dataset):
    """Weizmann Horses dataset."""

    def __init__(self, root, split, img_shape=(400, 500), color="images"):
        """
        Args:
            root (string): Directory with all the images.
            split - "train" or "val"
            color - "rgb" or "gray"
            
        """
        self.root = root
        self.color = color

        self.img_folder = os.path.join(root, color)
        self.mask_folder = os.path.join(root, "masks")
        self.img_list = sorted([name for name in os.listdir(self.img_folder)])
        # self.mask_list = sorted([name for name in os.listdir(self.mask_folder)])

        self.img_shape = img_shape

        split_index = int(len(self.img_list) * 0.8)

        self.split = split

        if split == "train":
            self.img_list = self.img_list[:split_index]
        if split == "val":
            self.img_list = self.img_list[split_index:]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, ind):
        # reading image
        img_name = os.path.join(self.img_folder,
                                self.img_list[ind])
        image = Image.open(img_name)

        # reading mask
        mask_name = self.img_list[ind].replace("image-", "mask-")
        mask_name = os.path.join(self.mask_folder,
                                 mask_name)
        mask = Image.open(mask_name)
        mask = mask.convert('1')

        # resizing of mask and image
        img_transforms = transforms.Compose([transforms.Resize(self.img_shape, interpolation=Image.BILINEAR),
                                             transforms.ToTensor()])

        mask_transforms = transforms.Compose([transforms.Resize(self.img_shape, interpolation=Image.NEAREST),
                                              transforms.ToTensor()])

        image = img_transforms(image)
        mask = mask_transforms(mask)

        return image, mask.long()


def show_sample(dataset, ind):
    fig, axs = plt.subplots(1, 2, figsize=(15, 4), constrained_layout=True)
    img, mask = dataset[ind]
    axs[0].imshow(img.permute(1, 2, 0))
    axs[0].axis('off')
    axs[1].imshow(mask.squeeze())
    axs[1].axis('off')
    plt.show()


def plot_batch_with_results(batch_imgs, batch_gts, results):
    """
    Plots images, GT segmentations and generated segmentations
    Input:
        batch_imgs - tensor of size (batch_size, 3, h, w)
        batch_gts - tensor of size (batch_size, 1, h, w)
        results - tensor of size (batch_size, 1, h, w)
    """
    batch_size = batch_imgs.shape[0]
    rows = 3

    fig = plt.figure(figsize=(batch_size * 5, rows * 4))
    gs = gridspec.GridSpec(rows, batch_size, wspace=0.0, hspace=0.0)

    for img_num in range(batch_size):
        ax = plt.subplot(gs[0, img_num])
        ax.axis('off')
        ax.imshow(batch_imgs[img_num].permute(1, 2, 0))

        ax = plt.subplot(gs[1, img_num])
        ax.axis('off')
        ax.imshow(batch_gts[img_num].squeeze())

        ax = plt.subplot(gs[2, img_num])
        ax.axis('off')
        ax.imshow(results[img_num].squeeze())

    fig.tight_layout()
    clear_output()
    plt.show()


def plot_history(history):
    """
    Визуализация истории обучения
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history["loss_train"], label='train', alpha=0.7)
    if "loss_val" in history:
        axes[0, 0].plot(history["loss_val"], label='val')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Iterations')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # IoU
    if "iou_val" in history:
        axes[0, 1].plot(history["iou_val"], label='val')
    if "iou_train" in history:
        axes[0, 1].plot(history["iou_train"], label='train', alpha=0.7)
    axes[0, 1].set_title('IoU')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Pixel Accuracy
    if "pixel_acc_val" in history:
        axes[1, 0].plot(history["pixel_acc_val"], label='val')
    if "pixel_acc_train" in history:
        axes[1, 0].plot(history["pixel_acc_train"], label='train', alpha=0.7)
    axes[1, 0].set_title('Pixel Accuracy')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Mean Accuracy
    if "mean_acc_val" in history:
        axes[1, 1].plot(history["mean_acc_val"], label='val')
    if "mean_acc_train" in history:
        axes[1, 1].plot(history["mean_acc_train"], label='train', alpha=0.7)
    axes[1, 1].set_title('Mean Accuracy')
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()


def pixel_accuracy(preds, masks):
    """
    preds: (H, W) - предсказанные классы (уже после argmax)
    masks: (H, W) - истинные маски
    """
    # Проверяем и выравниваем размеры
    if preds.shape != masks.shape:
        # Если размеры не совпадают, изменяем размер предсказаний под маски
        preds = torch.nn.functional.interpolate(
            preds.unsqueeze(0).unsqueeze(0).float(), 
            size=masks.shape[-2:],
            mode='nearest'
        ).squeeze().long()
    
    correct = (preds == masks).float().sum()
    total = masks.numel()
    return (correct / total).item() if total > 0 else 0.0


def mean_accuracy(preds, masks):
    """
    preds: (H, W) - предсказанные классы
    masks: (H, W) - истинные маски
    """
    if preds.shape != masks.shape:
        preds = torch.nn.functional.interpolate(
            preds.unsqueeze(0).unsqueeze(0).float(), 
            size=masks.shape[-2:],
            mode='nearest'
        ).squeeze().long()
    
    classes = torch.unique(masks)
    class_acc = []
    
    for cls in classes:
        if cls == 255:  # игнорируем класс игнора если есть
            continue
        mask_cls = masks == cls
        if mask_cls.sum() > 0:
            acc = (preds[mask_cls] == cls).float().mean()
            class_acc.append(acc.item())
    
    return np.mean(class_acc) if class_acc else 0.0


def mean_iou(preds, masks):
    """
    preds: (H, W) - предсказанные классы
    masks: (H, W) - истинные маски
    """
    if preds.shape != masks.shape:
        preds = torch.nn.functional.interpolate(
            preds.unsqueeze(0).unsqueeze(0).float(), 
            size=masks.shape[-2:],
            mode='nearest'
        ).squeeze().long()
    
    classes = torch.unique(masks)
    ious = []
    
    for cls in classes:
        if cls == 255:  # игнорируем класс игнора
            continue
            
        pred_cls = preds == cls
        mask_cls = masks == cls
        
        intersection = (pred_cls & mask_cls).sum().float()
        union = (pred_cls | mask_cls).sum().float()
        
        if union > 0:
            iou = intersection / union
            ious.append(iou.item())
    
    return np.mean(ious) if ious else 0.0
