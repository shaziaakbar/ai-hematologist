import argparse
import torch
import torchvision
import os
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

from src.trainer import Trainer
import src.utils as utils

# Require for Mac download
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class CustomViT(torchvision.models.vit.ViT):
    def __init__(self, *args, **kwargs):
        super(CustomViT, self).__init__(*args, **kwargs)

        # Modify the architecture to have a decoder


class NucleasTrainer(Trainer):
    def build_model(self, model_name, pretrain=None, num_classes=2):
        if model_name == "vit":
            model = torchvision.models.vit_b_16(weights=pretrain)
        else:
            raise NotImplementedError("model {} not implemented".format(model_name))

        classifier = list(model.classifier.children())
        model.classifier = torch.nn.Sequential(*classifier[:-1])
        model.classifier.add_module('last_conv', torch.nn.Conv2d(classifier[-1].in_channels, num_classes,
                                                                 kernel_size=classifier[-1].kernel_size,
                                                                 stride=classifier[-1].stride)
                                    )
        return model


def main(_args):
    """ Main function for training and validating cell segmentation model.

    Args:
        _args (dict): arguments from user
    """
    df = utils.read_training_data_df(args.data_dir, img_dir="imagesTr", gt_dir="labelsTr", gt_ext="_label.png")
    train_df, val_df = train_test_split(df, test_size=0.1)
    train_df.to_csv(os.path.join(args.output_dir, "train.csv"))
    val_df.to_csv(os.path.join(args.output_dir, "val.csv"))

    train_dataloader = utils.get_dataloader(train_df, shuffle=True, bs=args.batch_size, label_idx=1)
    val_dataloader = utils.get_dataloader(val_df, shuffle=False, bs=args.batch_size, label_idx=1)

    writer = SummaryWriter()
    trainer = NucleasTrainer(args, tb_writer=writer)
    trainer.run(train_dataloader, val_dataloader)
    writer.close()

    print("Saving outputs...")
    torch.save(trainer.model, os.path.join(args.output_dir, "model.pt"))
    utils.save_binary_masks(val_dataloader, trainer.model, args.output_dir, device=args.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', metavar='path', required=True, help='path to ground truth evaluation data')
    parser.add_argument('--output_dir', metavar='path', required=True, help='path where results will be saved')
    parser.add_argument('--num_epochs', default=100, required=False, help='number of training epochs')
    parser.add_argument('--model_name', default="vit", required=False, help='type of model')
    parser.add_argument('--model_weights', default="ViT_B_16_Weights.DEFAULT", required=False, help='type of model weights')
    parser.add_argument('--optimizer_name', default="adam", required=False, help='type of optimizer')
    parser.add_argument('--batch_size', default=24, required=False)
    parser.add_argument('--device', default="cpu", required=False)
    args = parser.parse_args()

    main(args)
