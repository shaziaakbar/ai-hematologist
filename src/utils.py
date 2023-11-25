import pandas as pd
import glob
import os
import torch
from medpy import metric


def read_training_data_df(base_dir, img_dir="imagesTr", gt_dir="labelsTr", gt_ext=".png"):
    """ Reads the necessary filepaths for training into a DataFrame.

    Args:
        base_dir (str): path to parent directory
        img_dir (str): name of image folder
        gt_dir (str): name of ground truth folder
        gt_ext (str): extension of ground truth files

    Returns:
        Pandas Dataframe listing image and segmentation files
    """
    list_images = glob.glob(os.path.join(base_dir, img_dir, "*.tiff"))
    list_contents = []

    for image_file in list_images:
        id = os.path.basename(image_file).split(".")[0]
        gt_file = os.path.join(base_dir, gt_dir, id + gt_ext)

        # If file does not exist then skip
        if not os.path.exists(gt_file):
            continue

        list_contents.append([id, image_file, gt_file])

    return pd.DataFrame(list_contents, columns=["id", "image_path", "seg_path"])


def read_val_data_df(base_dir, img_dir="imagesTr"):
    """ Reads the necessary filepaths for validation into a DataFrame.

    Args:
        base_dir (str): path to parent directory
        img_dir (str): name of image folder

    Returns:
        Pandas Dataframe listing path to images
    """
    list_images = glob.glob(os.path.join(base_dir, img_dir, "*.tiff"))
    return pd.DataFrame(list_images, columns=["image_path"])


def get_optimizer(model, optimizer_name="adam"):
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters())
    elif optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    else:
        raise NotImplementedError("optimizer {} not implemented".format(optimizer_name))


def get_criterion():
    """ Get dice loss function that operated on tensors. """
    def DiceLoss(predict, targets, smooth=1, p=2):
        predict = torch.max(predict, dim=1, keepdim=True)[0]
        predict = predict.contiguous().view(predict.shape[0], -1)
        targets = targets.contiguous().view(targets.shape[0], -1)

        num = torch.sum(torch.mul(predict, targets), dim=1) + smooth
        den = torch.sum(predict.pow(p) + targets.pow(p), dim=1) + smooth

        loss = 1 - num / den
        return loss.mean()
    return DiceLoss