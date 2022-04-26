# Fashion_recommender
The Fashion_recommender (FR) system recommends similar items to the buyers based on the input item.

The dataset used in this project is [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist), an extremely popular dataset to verify machine learning algorithms. The methodology proposed in this work includes hand-crafted features extraction based on traditional image processing algorithms, along with machine learning algorithms to classify the obtained features. The final results of the proposed method reach a promising accuracy of 81.3%, which is acceptable with hand-crafted feature extrator. The details regarding dataset, methodology, results/dicussion and conclusion are as follows:

# Table of Contents
1. [Usage](#para1)
    1. [Quick Trial](#sub1.1)
    2. [Installation](#sub1.2)
3. [Dataset](#para2)
4. [Methodology](#para3)
    1. [Hand-crafted feature extraction](#sub2.1)
    2. [Machine learning algorithm](#sub2.2)
5. [Results and Discussion](#para4)
6. [Conclusion](#para5)

# Usage <a name="para1"></a>
## Quick Trial <a name="sub1.1"></a>

For a quick trial of the proposed algorithm, you could access to [this](https://colab.research.google.com/drive/1XfNOKrnBJGANySVfm6XVybkOMG7aW6cr?authuser=1#scrollTo=gzTlv1ZTKYPW) Google Colab link. All required libraries and environments has been pre-installed. You could run 1 cell at a time to keep track of the flow of the algorithm, or Run-All to run all cells and wait until it outputs the final results.

This quick trial setup is highly recommended for those, who couldn't afford strong computer as running the algorithm is rather resource-consuming and time-consuming. Google Colab provides users with decent CPU and even GPU for free, the only drawback is it has quite strict limitation on GPU use time, but it is still sufficient for this task. 

## Installation <a name="sub1.2"></a>

The other way to try on this algorithm is to install it in your computer. The setps are simple, you only have to clone/download this git, install required libraries, extract `data.rar` and run the `main.py` file

# Dataset <a name="para2"></a>

[<img src="git_Img/fashion-mnist-sprite.png"/>](git_img/fashion-mnist-sprite.png)

The dataset used in this project is [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist). The initial `Fashion-MNIST` dataset has `idx-ubyte.gz` format, so I used `data_preprocessing.py` file to extract and rearrange the dataset in `jpg` format (You can find it in the `./data/test_original_size` and `./data/test_original_size` folder. The original size of each image is 28x28 pixels. Each pixel value is in the uint8 format and has the intensity range of 0-1.

At first, I tried feature extractor algorithms on the default Fashion-MNIST dataset but it failed because of the size and intensity range. Therefore, I have to scale-up the images to 10 times (280x280 pixels) and intensity range to 0-255.

In addition, the label of the dataset is as the table below:
|Label  | Description|
|-------|------------|
|0      |T-Shirt/Top |
|1      |Trouser     |
|2      |Pullover    |
|3      |Dress       |
|4      |Coat        |
|5      |Sandal      |
|6      |Shirt       |
|7      |Sneaker     |
|8      |Bag         |
|9      |Ankle boot  |

# Methodology <a name="para2"></a>
## Hand-crafted feature extraction <a name="sub2.1"></a>

There are a bunch of Hand-crafted feature extraction in the image processing world, and the most popular is `SIFT` and `SURF` algorithm. Sadly, both the authors of the algorithms claim license on their product, therefore these algorithms are no longer available for free now. Fortunely, some state-of-the-art algorithms came to the rescue as an equivalent to the licensed `SIFT` and `SURF` algorithms such as ORB, BRISK, KAZE and so on. The reference is from [this link](https://dsp.stackexchange.com/questions/1288/what-are-some-free-alternatives-to-sift-surf-that-can-be-used-in-commercial-app#:~:text=BRISK%20(includes%20a%20descriptor),descriptor%20(modified%20fast%20binary%20descriptor)). 

With the fundamental idea that an image is actually a matrix of pixels, and that matrix is constructed from a number of features, we could therefore represent an image by its set of features, instead of looking at the whole image. Thus, the ultimate goal of Hand-crafted feature extraction algorithm is to encode an image into a set of vectors (or features). In order to do that, most algorithms folow 2 phases: `Keypoint detector` and `Keypoint descriptor`. 

The `Keypoint detector` phase ouputs a number of points, which is considered an outstanding point or keypoint to represent the whole image. In addition, the `Keypoint descriptor` phase outputs a histogram of directions for each keypoint. Based on the histogram, we know the direction and magnitude specific to that keypoint.

In this project, I decided to use the `A-KAZE` algorithm, which is currently the fastest and most effective hand-crafted feature extrator available free of charge. In short, `A-KAZE` basically follows the 2 phases above with some improvements to increase speed and accuracy. It's the accelerated version of KAZE (which uses M-SURF feature descriptor and modified for nonlinear space), free of charge and uses M-LDB descriptor (modifed fast binary descriptor). An example of the ouput of Hand-crafted feature extraction using `AKAZE` on our data is as below:

This is a piece of code for feature extraction
```bash
# Decode images into vectors
akaze = cv2.KAZE_create()
for image in tqdm(train_image_list):
    img = cv2.imread(image)
    
    kp, des = akaze.detectAndCompute(img, None)
    for d in des:
        All_vector.append(d)
```

And this is some of the results

[<img src="git_Img/Figure_1.png" height="280" width="400"/>](git_img/Figure_1.png)
[<img src="git_Img/Figure_2.png" height="280" width="400"/>](git_img/Figure_2.png)
[<img src="git_Img/Figure_3.png" height="280" width="400"/>](git_img/Figure_3.png)
[<img src="git_Img/Figure_4.png" height="280" width="400"/>](git_img/Figure_4.png)

## Machine learning algorithm <a name="sub2.2"></a>

After obtaning the vectors of feature of each image in the dataset, the next step is to use `K-nn clustering` algorithm to divide the feature into many clusters. The line of code of the task is as follows:

```bash
# Clustering vectors 
kmeans = MiniBatchKMeans(n_clusters=word_num, batch_size=batch, verbose=0).fit(All_vector)
```
The next step is to assign appropriate clusters to features of each train and test image. And the final step is use `SVM algorithm` to perform the classification task. Note that there might be several kinds of featues, such as color, shape, texture, ... but in this project, we use only keypoint as texture featureand the result actually proves that it is sufficient. The SVM parameters are set as follows (regularization parameter = 5, kernel = 'rgf', gamma = 'scale'):

```bash
classifier = SVC(C=5, kernel='rbf', gamma='scale')
classifier.fit(X,Y)
```

# Results and Discussion <a name="para4"></a>

I ran this project using my PC with the configuration as follows:
|CPU    |GPU    |VRAM   |RAM    |
|-------|-------|-------|-------|
|Dual Xeon E5-2678v3|RTX 3060|12GB|32GB|

It takse roughly 1 hour to finish the running and output an accuracy of 81,3%

# Conclusion <a name="para5"></a>

We are well aware that our approach is still far from perfect, and the accuracy is still have a major distance as compared with Deeplearning-based methods. However, our methodology has promising selling points such as light-weight, fast, need only 1 input item from the customers (no need to track shopping habit) with decent accuracy of 81,3%. Our next work would be dealing with larger dataset of fashion items and find a way to improve the feature extraction efficiency.
