import torch
from skimage import io


class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform=None, label_idx=1):
        self.dataframe = dataframe
        self.transform = transform
        self.label_idx = label_idx

    def __len__(self):
        return len(self.dataframe)

    def binarize_segmentation(self, mask):
        return mask > self.label_idx

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image = io.imread(row["image_path"])
        label = self.binarize_segmentation(io.imread(row["seg_path"]))

        if self.transform is not None:
            if "seg_path" in row:
                transformed = self.transform(image=image, mask=label)
                img = transformed["image"]
                seg = transformed["mask"]
            else:
                img = self.transform(image=image)["image"]

        return img.transpose(2, 0, 1), seg.transpose(2, 0, 1)