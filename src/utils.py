import pandas as pd
import glob
import torch
from PIL import Image
import os
import src.mask_operations as mask_op
import numpy as np


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
        return torch.optim.Adam(model.parameters(), lr=0.00001)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=0.001)
    else:
        raise NotImplementedError("optimizer {} not implemented".format(optimizer_name))


class DiceLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.nn.functional.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


def get_criterion(criterion_name="cross_entropy"):
    """ Get dice loss function that operated on tensors. """
    if criterion_name == "dice":
        fn = DiceLoss()
    elif criterion_name == "cross_entropy":
        fn = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError("{} not implemented".format(criterion_name))
    return fn


def medpy_dc(result, reference):
    r"""
    Same as the medpy dice function but supported for numpy v23.0.0 and greater.
    """
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))

    intersection = np.count_nonzero(result & reference)

    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)

    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0

    return dc


def save_binary_masks(dataloader, model, save_dir, original_size=[400, 400], device="cpu"):
    """ Save outputs from model into images containing masks.

    Args:
        dataloader (torch Dataloader): initialized dataloader containing test data
        model (torch Model): torch model with weights
        save_dir (str): path to output directory
        original_size (list[int]): size of saved image
        device (str): device that model resides in
    """
    model.eval()

    gt = False
    dice_scores = []
    if "seg_path" in dataloader.dataset.dataframe.columns:
        gt = True

    for i, data in enumerate(dataloader, 0):
        if gt:
            img_id, inputs, labels = data
            labels = labels.to(device)
        else:
            img_id, inputs = data
        inputs = inputs.to(device)

        outputs = model(inputs)

        for j in range(inputs.size(0)):
            current_img = torch.argmax(outputs["out"][j], dim=0).detach().cpu().numpy().astype("bool")
            img = Image.fromarray(current_img).resize(original_size)  # resize back to original shape

            filename = os.path.join(save_dir, str(img_id[j]) + "_label.png")
            img.save(filename)

            if gt:
                current_label = labels[j].detach().cpu().numpy().astype("bool")
                dice_scores.append(medpy_dc(current_img, current_label))

    pd.DataFrame(dice_scores, columns=["dice"]).to_csv(os.path.join(save_dir, "dice_scores.csv"))
    print("Average dice: {:.6f}".format(np.mean(np.array(dice_scores))))


def save_patch_binary_masks(dataloader, model, save_dir, device="cpu",
                            image_shape=(65,65), original_size=(400,400)):
    """ Save outputs from model into images containing masks.

    Args:
        dataloader (torch Dataloader): initialized dataloader containing test data
        model (torch Model): torch model with weights
        save_dir (str): path to output directory
        device (str): device that model resides in
    """
    model.eval()

    gt = False
    dice_scores = []
    if "seg_path" in dataloader.dataset.dataframe.columns:
        gt = True

    for i, data in enumerate(dataloader, 0):
        if gt:
            img_id, inputs, labels = data
            labels = labels.to(device)
        else:
            img_id, inputs = data
        inputs = inputs.to(device)

        outputs = model(inputs)

        # reshape patched to image again
        predicted_image = torch.argmax(outputs, dim=1).detach().cpu().numpy().astype("bool")
        predicted_image_reshaped = predicted_image.reshape(image_shape)
        img = Image.fromarray(predicted_image_reshaped).resize(original_size)

        filename = os.path.join(save_dir, str(img_id[0]) + "_label.png")
        img.save(filename)

        if gt:
            current_label = labels.detach().cpu().numpy().astype("bool")
            dice_scores.append(medpy_dc(predicted_image.flatten(), current_label))

    pd.DataFrame(dice_scores, columns=["dice"]).to_csv(os.path.join(save_dir, "dice_scores.csv"))
    print("Average dice: {:.6f}".format(np.mean(np.array(dice_scores))))