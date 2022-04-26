import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import scipy.ndimage as ndimage

BASE_DIR = 'D:/CV Course/FashionMNIST/data/'

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

train_image_list = []
train_id = []
train_img_bow = []
train_img_bow_label = []

test_image_list = []
test_id = []
test_img_bow = []
test_img_bow_label = []

All_vector = []

word_num = 300
batch = 30000



if __name__ == "__main__":
    # Decode images into vectors
    akaze = cv2.KAZE_create()
    for i, (inputs, labels) in tqdm(enumerate(training_loader)):
        inputs = cv2.resize(inputs.numpy()[0,0], (280,280), interpolation=cv2.INTER_CUBIC)

        # print(inputs.shape)
        kp, des = akaze.detectAndCompute(inputs, None)
        for d in des:
            All_vector.append(d)     
        # print(des.shape)   
    # Clustering vectors 
    kmeans = MiniBatchKMeans(n_clusters=word_num, batch_size=batch, verbose=1).fit(All_vector)
    

    # Create histogram of features for each training image
    for i, (inputs, labels) in tqdm(enumerate(training_loader)):
        inputs = cv2.resize(inputs.numpy()[0,0], (280,280), interpolation=cv2.INTER_CUBIC)
        labels = labels.numpy()
        kp, des = akaze.detectAndCompute(inputs, None)

        histo = np.zeros(word_num) 
        nkp = np.size(kp)  #nkp: number of keypoints

        for d in des:
            idx = kmeans.predict([d])
            histo[idx] += 1
            
        histo=np.array(histo) # chuyen ve dang vector numpy
        histo=histo/nkp # normalized

        train_img_bow.append(histo)
        train_img_bow_label.append(labels)


    # Create histogram of features for each validation image
    kmeans.verbose = False
    for i, (inputs, labels) in tqdm(enumerate(validation_loader)):
        inputs = cv2.resize(inputs.numpy()[0,0], (280,280), interpolation=cv2.INTER_CUBIC)
        labels = labels.numpy()
        kp, des = akaze.detectAndCompute(inputs, None)

        histo = np.zeros(word_num) 
        nkp = np.size(kp)  #nkp: number of keypoints

        for d in des:
            idx = kmeans.predict([d])
            histo[idx] += 1
            
        histo=np.array(histo) # chuyen ve dang vector numpy
        histo=histo/nkp # normalized

        test_img_bow.append(histo)
        test_img_bow_label.append(labels)
    

    # Classification using SVM
    X = np.array(train_img_bow)
    Y = np.array(train_img_bow_label)

    X_test = np.array(test_img_bow)
    Y_test = np.array(test_img_bow_label)

    classifier = SVC(C=5, kernel='rbf', gamma='scale')
    classifier.fit(X,Y)
    res = classifier.predict(X_test)

    accuracy = sum(res==Y_test)/len(Y_test)
    print(accuracy)





# img1 = cv2.imread(BASE_DIR + '0/1.jpg')
# img2 = cv2.imread(BASE_DIR + '0/3.jpg')

# img1 = cv2.resize(img1, (280,280), interpolation=cv2.INTER_CUBIC)
# img2 = cv2.resize(img2, (280,280), interpolation=cv2.INTER_CUBIC)

# akaze = cv2.KAZE_create()
# kpts1, desc1 = akaze.detectAndCompute(img1, None)
# kpts2, desc2 = akaze.detectAndCompute(img2, None)

# print(len(kpts1), len(kpts2))

# bf = cv2.BFMatcher()
# matches = bf.knnMatch(desc1, desc2, k=2)

# # Apply ratio test
# good_and_second_good_match_list = []
# for m in matches:
#     if m[0].distance/m[1].distance < 0.5:
#         good_and_second_good_match_list.append(m)
# good_match_arr = np.asarray(good_and_second_good_match_list)[:,0]

# # show only 30 matches
# im_matches = cv2.drawMatchesKnn(img1, kpts1, img2, kpts2,
#                                 good_and_second_good_match_list[0:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# plt.figure(figsize=(20, 20))
# plt.imshow(im_matches)
# plt.title("keypoints matches")
# plt.show()

# test = cv2.drawKeypoints(img1, kpts1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# plt.figure(figsize=(10,10))
# plt.imshow(test)
# plt.title("keypoints")
# plt.show()


# matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
# nn_matches = matcher.knnMatch(desc1, desc2, 2)

# matched1 = []
# matched2 = []
# nn_match_ratio = 0.8 # Nearest neighbor matching ratio
# for m, n in nn_matches:
#     if m.distance < nn_match_ratio * n.distance:
#         matched1.append(kpts1[m.queryIdx])
#         matched2.append(kpts2[m.trainIdx])


# imgTr_names= os.listdir(classTr_path)
# imgTest_names= os.listdir(classTest_path)

# imgTr_path=[]
# imgTest_path=[]
# imgTr_id=[]
# imgTest_id=[]

# for image in imgTr_names:
#     imgTr_path.append(classTr_path+'/'+ image)
#     imgTr_id.append(int(path))

# for image in imgTest_names:
#     imgTest_path.append(classTest_path+'/'+ image)
#     imgTest_id.append(int(path))

# # Chon ngau nhien 60 anh lam train,40 anh moi class lam anh test
# train_idx = np.arange(6000)
# np.random.shuffle(train_idx)
# test_idx = np.arange(1000)
# np.random.shuffle(test_idx)

# train_image_list += [imgTr_path[i] for i in train_idx]
# test_image_list += [imgTest_path[i] for i in test_idx]
# train_id += [imgTr_id[i] for i in train_idx]
# test_id += [imgTest_id[i] for i in test_idx]