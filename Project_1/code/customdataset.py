from torch.utils.data import Dataset
import os
# from PIL import Image
import glob
import cv2

class CustomDataset(Dataset):
    def __init__(self, path, transform=None):
        self.all_path = glob.glob(os.path.join(path, "*", "*.jpg"))
        self.transform = transform
        self.label_dict = {}
        for i, category in enumerate(sorted(os.listdir(path))):
            self.label_dict[category] = i

    def __getitem__(self, item):
        # 1. Reading image
        image_path = self.all_path[item]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. class label
        label_path = self.all_path[item].split("/")[-2]
        label = self.label_dict[label_path]
        # 3. Applying transform on images
        if self.transform is not None:
            image = self.transform(image=image)["image"]

        # 4. return
        return image, label

    def __len__(self):
        return len(self.all_path)

# test = CustomDataset("./dataset/test")
# for i in test:
#     pass
