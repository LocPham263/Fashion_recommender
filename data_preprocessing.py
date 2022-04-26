import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import cv2

BASE_DIR = 'D:/CV Course/FashionMNIST/data/train/'

# img = cv2.imread(BASE_DIR + '0/0.jpg')
# print(img.shape)
# cv2.imshow('img',img)
# cv2.waitKey()
# cv2.destroyAllWindows()

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])
    # ])

# Create datasets for training & validation, download if necessary
training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=False)
validation_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=False)

# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = torch.utils.data.DataLoader(training_set, batch_size=1, shuffle=True, num_workers=2)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=1, shuffle=False, num_workers=2)

# Class labels
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# # Report split sizes
# print('Training set has {} instances'.format(len(training_set)))
# print('Validation set has {} instances'.format(len(validation_set)))

# Helper function for inline image display
def matplotlib_imshow(img, one_channel=False):
    fig = plt.figure(figsize=(12,12))
    cols = 5
    rows = 5
    for i in range(1, cols*rows+1):
        img_show = img[i]    
        fig.add_subplot(rows, cols, i)
        plt.imshow(img_show)
    plt.show()

    # if one_channel:
    #     img = img.mean(dim=0)
    # img = img / 2 + 0.5     # unnormalize
    # npimg = img.numpy()
    # if one_channel:
    #     plt.imshow(npimg, cmap="Greys")
    # else:
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.show()

if __name__ == "__main__":
    for i, (inputs, labels) in tqdm(enumerate(training_loader)):
        inputs = (inputs.numpy()/2 + 0.5) * 255
        labels = labels.numpy()
        inputs = cv2.resize(inputs[0,0], (280,280), interpolation=cv2.INTER_CUBIC)
        
        # print(str(labels[0]))
        img = inputs.astype(np.uint8)
        img = Image.fromarray(img)
        # img = img.convert("RGB")
        img.save(BASE_DIR + str(labels[0]) + '/' + str(len(os.listdir(BASE_DIR + str(labels[0]))) + 1) + '.jpg')
     
            
        # print(inputs.shape, labels)
        # plt.imshow(inputs[0,0], cmap='gray')
        # plt.show()

    # dataiter = iter(training_loader)
    # images, labels = dataiter.next()

    # # Create a grid from the images and show them
    # img_grid = torchvision.utils.make_grid(images)
    # matplotlib_imshow(img_grid, one_channel=True)
    # print('  '.join(classes[labels[j]] for j in range(4)))