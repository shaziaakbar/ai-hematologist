import os.path
import glob
import argparse
from src.utils import medpy_dc
from skimage import io
import pandas as pd


def get_dice_scores(label_seg, label_gt):
    """ Compute dice scores for cell and nucleas.

    Args:
        label_seg (numpy): array containing segmentation
        label_gt (numpy): array containing ground truth

    Returns:
        Whole cell dice and nucleas dice scores.
    """
    whole_cell_dc = medpy_dc(label_seg>0, label_gt>0)
    nucleus_dc = medpy_dc(label_seg==2, label_gt==2)

    return whole_cell_dc, nucleus_dc


def main(segmentation_directory, gt_directory):
    """ Main function to evaluate results.

    Args:
        segmentation_directory (str): path to segmentation results
        gt_directory (str): path to ground truth data
    """
    list_gt = glob.glob(os.path.join(gt_directory, "*.tiff"))
    results = []
    for file_gt in list_gt:
        name = file_gt.split(".")[0]
        file_seg = os.path.join(segmentation_directory, name + "_label.png")
        if not os.path.exists(file_seg):
            raise FileNotFoundError("Segmentation file {} not found".format(file_seg))

        label_seg = io.imread('path to segmentation mask file')
        label_gt = io.imread('path to ground-truth mask file')
        results.append([name, file_gt, file_seg] + [get_dice_scores(label_seg, label_gt)])

    df = pd.DataFrame(results, columns=["id", "path_to_seg", "path_to_gt","dice_cell", "dice_nucleas"])
    df.to_csv(os.path.join(segmentation_directory, "results.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seg_dir', metavar='path', required=True,
                        help='the path to segmentation results')
    parser.add_argument('--gt_dir', metavar='path', required=True,
                        help='path to ground truth evaluation data')
    args = parser.parse_args()

    main(args.seg_dir, args.gt_dir)
