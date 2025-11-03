import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

IMAGE_SIZE = 32
IMAGES_DIR = os.path.join('..', 'data', 'cifar10', 'data', 'img')

class ClientDataset(Dataset):
    """ CIFAR100 Dataset """

    def __init__(self, data, train=True, loading='training_time', cutout=None):
        """
        Args:
            data: dictionary in the form {'x': list of imgs ids, 'y': list of correspondings labels}
            train (bool, optional): boolean for distinguishing between client's train and test data
        """
        self.root_dir = os.path.join(IMAGES_DIR, "train" if train else "test")
        self.imgs = []
        self.labels = []
        self.loading = loading
        self.image_cache = {}  # Add cache dictionary


        if data is None:
            return

        if loading == 'init':
            # Parallel image loading using ThreadPoolExecutor
            def load_image(img_name):
                img_path = os.path.join(self.root_dir, img_name)
                image = Image.open(img_path).convert('RGB')
                image.load()  # Force load image into memory
                return image

            with ThreadPoolExecutor(max_workers=min(12, len(data['x']))) as executor:
                future_to_idx = {
                    executor.submit(load_image, img_name): i 
                    for i, img_name in enumerate(data['x'])
                }
                
                # Store images in correct order
                self.imgs = [None] * len(data['x'])
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    self.imgs[idx] = future.result()
        else:
            self.imgs = data['x']
            
        self.labels = data['y']

        if train:
            self.train_transform = transforms.Compose([
                                        transforms.RandomCrop(IMAGE_SIZE, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                    ])
            self.test_transform = None
            if cutout is not None:
                self.train_transform.transforms.append(cutout(n_holes=1, length=16))
        else:
            self.train_transform = None
            self.test_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                    ])


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.loading == 'training_time':
            img_name = self.imgs[idx]
            if img_name not in self.image_cache:
                img_path = os.path.join(self.root_dir, img_name)
                self.image_cache[img_name] = Image.open(img_path).convert('RGB')
            image = self.image_cache[img_name]
        else:
            image = self.imgs[idx]
        label = self.labels[idx]

        if self.train_transform:
            image = self.train_transform(image)
        elif self.test_transform:
            image = self.test_transform(image)
        return image, label