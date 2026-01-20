import imageio
import numpy as np
import pickle
from tqdm import tqdm

DIRPATH = "./"


def unpickle(file):
    with open(file, "rb") as fo:
        res = pickle.load(fo, encoding="bytes")
    return res


print("#### Setting up CIFAR100 ####")

meta = unpickle(DIRPATH + "cifar-100-python/meta")
train = unpickle(DIRPATH + "cifar-100-python/train")

fine_labels = train[b"fine_labels"]
data = train[b"data"]

images = list()
for d in data:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[..., 0] = np.reshape(d[:1024], (32, 32))  # Red channel
    image[..., 1] = np.reshape(d[1024:2048], (32, 32))  # Green channel
    image[..., 2] = np.reshape(d[2048:], (32, 32))  # Blue channel
    images.append(image)

print("Saving train images...")
for index, image in tqdm(enumerate(images), total=len(images)):
    label = fine_labels[index]
    filename = "img_" + str(index) + "_label_" + str(label) + ".png"
    imageio.imwrite("../data/img/train/%s" % filename, image)

test = unpickle(DIRPATH + "cifar-100-python/test")
fine_labels = test[b"fine_labels"]
data = test[b"data"]

images = list()
for d in data:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[..., 0] = np.reshape(d[:1024], (32, 32))  # Red channel
    image[..., 1] = np.reshape(d[1024:2048], (32, 32))  # Green channel
    image[..., 2] = np.reshape(d[2048:], (32, 32))  # Blue channel
    images.append(image)

print("Saving test images...")
for index, image in tqdm(enumerate(images), total=len(images)):
    label = fine_labels[index]
    filename = "img_" + str(index) + "_label_" + str(label) + ".png"
    imageio.imwrite("../data/img/test/%s" % filename, image)
