import torch
import torchvision.transforms
from skimage import io
import numpy as np


class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform=None, label_idx=0):
        self.dataframe = dataframe
        self.transform = transform
        self.label_idx = label_idx

    def __len__(self):
        return len(self.dataframe)

    def binarize_segmentation(self, mask):
        return (mask > self.label_idx)[..., np.newaxis].astype('int')

    def read_image(self, row):
        image = io.imread(row["image_path"])[:, :, :3].astype("float32")
        return image

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image = self.read_image(row)

        if self.transform is not None:
            if "seg_path" in row:
                label = self.binarize_segmentation(io.imread(row["seg_path"]))
                transformed = self.transform(image=image, mask=label)
                img = transformed["image"]
                seg = transformed["mask"]
                return row["id"], img.transpose(2, 0, 1), seg.transpose(2, 0, 1)
            else:
                img = self.transform(image=image)["image"]
                return row["id"], img.transpose(2, 0, 1)


class PatchSegmentationDataset(SegmentationDataset):
    def __init__(self, dataframe, patch_size=[32, 32], stride=[16, 16], *args, **kwargs):
        super(PatchSegmentationDataset, self).__init__(dataframe, *args, **kwargs)
        self.patch_size = patch_size
        self.stride = stride

    def extract_patches(self, image):
        """ Extract patches from a whole image.

        Args:
            image (numpy): array containing images to be windowed

        Returns:
            Torch tensor containing patches of size self.patch_size and stride self.stride
        """

        # pad the edges to prevent the final image from being cropped
        image = np.pad(image, ((0, 0), (self.patch_size[0] // 2, self.patch_size[0] // 2),
                               (self.patch_size[1] // 2, self.patch_size[1] // 2)))

        image = torch.from_numpy(image)
        patches = image.unfold(1, self.patch_size[0], self.stride[0]).unfold(2, self.patch_size[1], self.stride[1])
        return patches

    def __getitem__(self, index):
        data = super(PatchSegmentationDataset, self).__getitem__(index)

        label = None
        if len(data) == 3:
            pat_id, image, label = data
        else:
            pat_id, image = data

        image = self.extract_patches(image)
        num_rows, num_cols = image.shape[1], image.shape[2]
        patches = image.flatten(start_dim=1, end_dim=2)
        pat_id = tuple(np.repeat(pat_id, num_rows * num_cols))

        if label is None:
            return pat_id, patches
        else:
            label = self.extract_patches(label)
            mid_x, mid_y = label.size(-2) // 2, label.size(-1) // 2
            label = label[:, :, :, mid_x, mid_y].flatten()
            return pat_id, patches, label

