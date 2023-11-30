import numpy as np
import pandas as pd
import argparse
import torch
import os
import glob

from src.train_vit_nucleas import NucleasTrainer, PATCH_SIZE
import src.utils as utils
from src.datasets import NumpyPatchInferenceDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', metavar='path', required=True, help='path to ground truth evaluation data')
    parser.add_argument('--output_dir', metavar='path', required=True, help='path where results will be saved')
    parser.add_argument('--model_path', default="cpu", required=False)
    parser.add_argument('--patch_dir', default="cpu", required=False)
    parser.add_argument('--device', default="cpu", required=False)
    args = parser.parse_args()

    if not os.path.exists(args.patch_dir):
        os.mkdir(args.patch_dir)
        print("Extracting patches...")
        utils.create_patches_from_directory(args.data_dir, args.patch_dir, patch_size=[PATCH_SIZE, PATCH_SIZE])

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    prediction_path = os.path.join(args.output_dir, "predictions.npz")
    df = pd.DataFrame(glob.glob(os.path.join(args.patch_dir, "*.npz")), columns=["image_path"])
    if not os.path.exists(prediction_path):
        trainer = NucleasTrainer(args)
        val_full_dataloader = trainer.get_dataloader(df, shuffle=False, bs=1, label_idx=1,
                                                     dataset_type=NumpyPatchInferenceDataset)
        model = torch.load(args.model_path)
        model.to(args.device)

        print("Getting patch predictions...")
        scores = utils.save_patch_binary_masks(val_full_dataloader, model, args.output_dir,
                                               device=args.device,
                                               save_images=False)

        np.savez(prediction_path, y=np.stack(scores))

    print("Saving predictions to image files...")
    predictions = np.load(prediction_path)["y"]
    utils.group_numpys_and_save_masks(predictions, df, args.output_dir, original_size=[360, 363])

