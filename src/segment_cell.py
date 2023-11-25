import argparse
import torch
import albumentations as A
import torchvision
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

from src.datasets import SegmentationDataset
from src.utils import read_training_data_df, get_optimizer, get_criterion, save_masks

# Require for Mac download
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

DEVICE = "mps"


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
        crop = [A.Resize(256, 256), A.CenterCrop(224, 224)]
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


def build_model(model_name="fcn", pretrain=None, num_classes=2):
    """ Get torch model for training.

    Args:
        model_name (str): type of CNN model
        pretrain (str): pretrained weights to load
        num_classes (int): number of output classes

    Returns:
        PyTorch model pretrained with specified weights
    """
    if model_name == "fcn":
        model = torchvision.models.segmentation.fcn_resnet101(weights=pretrain, num_classes=num_classes)
    else:
        raise NotImplementedError("model {} not implemented".format(model_name))
    return model


def train_model(dataloader, model, epoch, optimizer, criterion, tb_writer=None):
    """ Train a single epoch.

    Args:
        dataloader (torch Dataloader): initialized dataloader containing training data
        model (torch Model): torch model with weights
        epoch (int): current epoch
        optimizer: torch optimizer
        criterion: function for assessing loss
        tb_writer (optional): Tensorboard writer
    """
    model.train()

    running_correct = 0
    loss_idx_value = (epoch * len(dataloader))
    for i, data in enumerate(dataloader, 0):
        _, inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs["out"], labels)
        loss.backward()
        optimizer.step()

        # Get dice performance for current batch
        correct = torch.abs(torch.Tensor([1]) - loss.item())
        running_correct += correct

        # print statistics
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss:.6f}; acc: {float(correct.detach().cpu()):.6f}')

        if tb_writer is not None:
            tb_writer.add_scalar("Loss/Minibatches", loss, loss_idx_value)
            loss_idx_value += 1

    if tb_writer is not None:
        tb_writer.add_scalar("Loss/Epochs", loss, epoch)
        tb_writer.add_scalar("Accuracy/Epochs", running_correct / len(dataloader), epoch)


def validate_model(dataloader, model, epoch, criterion, tb_writer=None):
    """ Evaluate dice on validation set.

    Args:
        dataloader (torch Dataloader): initialized dataloader containing validation data
        model (torch Model): torch model with weights
        epoch (int): current epoch
        criterion: function for assessing dice loss
        tb_writer (optional): Tensorboard writer
    """
    model.eval()
    correct = 0.0
    for i, data in enumerate(dataloader, 0):
        _, inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        outputs = model(inputs)
        correct += torch.abs(torch.Tensor([1]) - criterion(outputs["out"], labels).detach().cpu())

    if tb_writer is not None:
        tb_writer.add_scalar("Accuracy/Validation", correct / len(dataloader), epoch)


def main(dir, save_dir, epochs, model_name, optimizer, batch_size):
    """ Main function for training and validating cell segmentation model.

    Args:
        dir (str): path to parent directory containing all data
        save_dir (str): path to output directory
        epochs (int): number of training epochs
        model_name(str): type of model to train
        optimizer (str): name of optimizer
    """
    df = read_training_data_df(dir, img_dir="imagesTr", gt_dir="labelsTr", gt_ext="_label.png")
    train_df, val_df = train_test_split(df, test_size=0.1)

    train_dataloader = get_dataloader(train_df, shuffle=True, bs=batch_size)
    val_dataloader = get_dataloader(val_df, shuffle=False, bs=batch_size)

    print("Building model...")
    model = build_model(model_name=model_name)
    model.to(DEVICE)

    optimizer = get_optimizer(model, optimizer)
    criterion = get_criterion()
    writer = SummaryWriter()

    print("Training...")
    for _e in range(epochs):
        train_model(train_dataloader, model, _e, optimizer, criterion, tb_writer=writer)
        validate_model(val_dataloader, model, _e, criterion, tb_writer=writer)

    writer.close()
    save_masks(val_dataloader, model, save_dir, device=DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', metavar='path', required=True, help='path to ground truth evaluation data')
    parser.add_argument('--output_dir', metavar='path', required=True, help='path where results will be saved')
    parser.add_argument('--num_epochs', default=200, required=False, help='number of training epochs')
    parser.add_argument('--model_name', default="fcn", required=False, help='type of model')
    parser.add_argument('--optimizer_name', default="adam", required=False, help='type of optimizer')
    parser.add_argument('--batch_size', default=12, required=False)
    args = parser.parse_args()

    main(args.data_dir, args.output_dir, args.num_epochs, args.model_name, args.optimizer_name, args.batch_size)
