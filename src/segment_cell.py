import argparse
import torch
import albumentations as A
import torchvision

from src.datasets import SegmentationDataset
from src.utils import read_data_df


def get_dataloader(data_df, shuffle=False, bs=24, num_workers=0):
    """ Get Torch dataloader for training/validation.

    Args:
        data_df (pandas DataFrame): contains paths to image and GT
        shuffle (bool): whether to shuffle data
        bs (int): batch size
        num_workers (int): number of workers

    Returns:
        Pytorch dataloader
    """
    if not shuffle:
        crop = [A.Resize(224, 224), A.CenterCrop(224, 224)]
    else:
        # augment data during training
        crop = [
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.4),
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
            A.Resize(224, 224),
            A.CenterCrop(224, 224)
        ]

    dataset = SegmentationDataset(data_df,
                                  transform=A.Compose(
                                      crop +
                                      [A.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])]
                                  ))
    return torch.utils.data.DataLoader(dataset, shuffle=shuffle, batch_size=bs, num_workers=num_workers)


def build_model(model_name="alexnet", pretrain=None):
    """ Get torch model for training.

    Args:
        model_name (str): type of CNN model

    Returns:
        PyTorch model pretrained with specified weights
    """
    if model_name == "alexnet":
        model = torchvision.models.alexnet(weights=pretrain)
    else:
        raise NotImplementedError("model {} not implemented".format(model_name))
    return model


def train_model(df, model):
    model.train()
    return


def validate_model(df, model):
    model.eval()
    return


def main(dir, epochs):
    """ Main function for training and validating cell segmentation model.

    Args:
        dir (str): path to parent directory containing all data
        epochs (int): number of training epochs
    """
    train_df = read_data_df(dir, img_dir="imagesTr", gt_dir="labelsTr", gt_ext=".png")
    val_df = read_data_df(dir, img_dir="imagesTs-Internal", gt_dir="imagesTs-Internal", gt_ext=".jpg")

    train_dataloader = get_dataloader(train_df, shuffle=True)
    val_dataloader = get_dataloader(val_df, shuffle=False)

    model = build_model()

    for _e in range(epochs):
        model = train_model(train_dataloader, model)
        validate_model(model, val_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', metavar='path', required=True,
                        help='path to ground truth evaluation data')
    parser.add_argument('--num_epochs', default=24, required=False,
                        help='number of training epochs')
    args = parser.parse_args()

    main(args.data_dir, args.num_epochs)
