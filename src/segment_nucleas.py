import argparse

import torch
import torchvision
import os
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import albumentations as A

from src.trainer import Trainer
import src.utils as utils
from src.datasets import PatchSegmentationDataset

# Require for Mac download
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


PATCH_SIZE = 64


class NucleasTrainer(Trainer):
    def get_dataloader(self, data_df, shuffle=False, bs=24, num_workers=0,
                       label_idx=1, dataset_type=None, collate_fn=None,
                       patch_size=[32, 32], stride=[16, 16]):
        """ Get Torch dataloader for training/validation.

        Args:
            data_df (pandas DataFrame): contains paths to image and GT
            shuffle (bool): whether to shuffle data
            bs (int): batch size
            num_workers (int): number of workers
            label_idx (int): threshold to apply to cell images
            dataset_type (torch Dataset): define dataset to be used for dataloader
            collate_fn: optional function for collating patches

        Returns:
            Pytorch dataloader
        """
        init_transform = []
        if shuffle:
            # augment data during training
            init_transform = [
                A.ShiftScaleRotate(shift_limit=0.4, scale_limit=0.2, rotate_limit=30, p=0.8),
                A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.6),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            ]

        if dataset_type is None:
            dataset_type = PatchSegmentationDataset

        dataset = dataset_type(data_df,
                               transform=A.Compose(
                                   init_transform +
                                   [A.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])]
                               ),
                               label_idx=label_idx,
                               patch_size=patch_size,
                               stride=stride)

        return torch.utils.data.DataLoader(dataset, shuffle=shuffle, batch_size=bs,
                                           num_workers=num_workers, collate_fn=collate_fn)

    def build_model(self, model_name, pretrain=None, num_classes=2):
        if model_name == "vit":
            model = torchvision.models.vit_b_16(image_size=PATCH_SIZE, num_classes=2)
        else:
            raise NotImplementedError("model {} not implemented".format(model_name))

        return model


def collate_fn(data):
    im = torch.concat([x[1] for x in data], axis=1).transpose(1, 0)
    pat_id = tuple(sum([list(x[0]) for x in data], []))

    if len(data[0]) > 2:
        labels = torch.concat([x[2] for x in data])
        return pat_id, im, labels
    else:
        return pat_id, im


def main(_args):
    """ Main function for training and validating cell segmentation model.

    Args:
        _args (dict): arguments from user
    """
    df = utils.read_training_data_df(args.data_dir, img_dir="imagesTr", gt_dir="labelsTr", gt_ext="_label.png")
    train_df, val_df = train_test_split(df, test_size=0.1)
    train_df.to_csv(os.path.join(args.output_dir, "train.csv"))
    val_df.to_csv(os.path.join(args.output_dir, "val.csv"))

    writer = SummaryWriter()
    trainer = NucleasTrainer(args, tb_writer=writer)

    train_dataloader = trainer.get_dataloader(train_df, shuffle=True, bs=args.batch_size,
                                              collate_fn=collate_fn, patch_size=[PATCH_SIZE, PATCH_SIZE],
                                              stride=[PATCH_SIZE, PATCH_SIZE])
    val_dataloader = trainer.get_dataloader(val_df, shuffle=False, bs=args.batch_size,
                                            collate_fn=collate_fn, patch_size=[PATCH_SIZE, PATCH_SIZE],
                                            stride=[PATCH_SIZE, PATCH_SIZE])

    trainer.run(train_dataloader, val_dataloader)
    writer.close()

    print("Saving outputs...")
    torch.save(trainer.model, os.path.join(args.output_dir, "model.pt"))
    val__full_dataloader = trainer.get_dataloader(val_df, shuffle=False, bs=args.batch_size,
                                                  collate_fn=collate_fn, patch_size=[PATCH_SIZE, PATCH_SIZE],
                                                  stride=[8, 8])
    utils.save_patch_binary_masks(val__full_dataloader, trainer.model, args.output_dir,
                                  device=args.device, image_shape=(51, 51))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', metavar='path', required=True, help='path to ground truth evaluation data')
    parser.add_argument('--output_dir', metavar='path', required=True, help='path where results will be saved')
    parser.add_argument('--num_epochs', default=50, required=False, help='number of training epochs')
    parser.add_argument('--model_name', default="vit", required=False, help='type of model')
    parser.add_argument('--model_weights', default=None, required=False, help='type of model weights')
    parser.add_argument('--optimizer_name', default="adam", required=False, help='type of optimizer')
    parser.add_argument('--batch_size', default=12, required=False)
    parser.add_argument('--device', default="cpu", required=False)
    args = parser.parse_args()

    main(args)
