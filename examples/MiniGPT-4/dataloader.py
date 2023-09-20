from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class Image_text_set(Dataset):
    def __init__(self, input_path) -> None:
        super().__init__()
        self.path = Path(input_path)
        self.data_list = list(self.path.glob("*.npz"))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        npzfile = np.load(self.data_list[index])
        image = npzfile['arr_0']
        text = npzfile['arr_1'][0]
        image = torch.from_numpy(image).cuda()

        return image, text
