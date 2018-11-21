# Building a Dog Breed Dectector Bot on Facebook Workplace, with Keras

![](images/BotImage.jpg)

Here, at Vonage, we use Facebook Workplace as one of our many channels for communication. If you haven't used it or heard about it, it's just like regular Facebook, but for companies and teams. All of us here at Vonage have an account, and we are able to view and join different groups throughout the organization.

A few months ago, one of our coworkers created a group for showing our pets, and it was a great idea, and a lot of members on the team post photos of their pets. I check the group almost everyday, and its a good way to enjoy the finer things in life (puppies!)

![](images/wp-group.jpg)

So after looking at everyone's photo of their dog, cat, and even bunny, some people asked, "What breed is that?". Once I saw that, i had an idea, to build a machine learning algorithm to figure out what dog breed was in the photo. And since we were all using Workplace, it made sense to build a bot that allowed any user in a post, to '@' the bot and ask what dog breed was in the photo.


### Where do I start?

In order to tackle many machine learning problems, you need data, and lots of it. Specifically, we need photos of a lot of dogs, and what kind of breed there are. That seems like a big undertaking right there. How are we going to find images of every dog breed? We *could* do this by hand: finding a bunch of images of dogs on Google, then label every photo of what kind of dog this is. Luckly, we don't have to do this. This is called Reinforcement Learning, where we training a model on images with labels (This is a image of a shitsu, this other image is a Bulldog etc..)
Lucklly, we won't have to get this data ourselves. Places like [Kaggle](https://kaggle.com), [Google Datasets](https://toolbox.google.com/datasetsearch) and  the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/index.php) already have this kind of data. And for this example, we are going to use the dataset from the [Dog Breed Identification Challenge](https://www.kaggle.com/c/dog-breed-identification) on Kaggle

This dataset contains over 10,000 images of dogs, categorized by breed. Even though 10k *seems* like alot of images, its not. Machine learning models that have been trained well use way more images than that, so we need to be sure we can train our model well, even with this low amount of images.

### Enough chat, lets build.
We'll be splitting this post into 3 sections
* [Training the model
* Serving the model
* Building the Workplace bot


### Training the Model {#training}
First, lets start with building the model. I'll be using [Google Colab](https://colab.research.google.com/) to build my [Jupyter Notebook](jupyter.org/), in Python. A Jupyter Notebook is a open sourced web app that lets you write code, as well as text and images. Its a great way to get started. Google Colab is a free service that will host your notebooks. Also, you can run your code using GPU's and [TPU's](https://cloud.google.com/blog/products/gcp/an-in-depth-look-at-googles-first-tensor-processing-unit-tpu). Running most machine learning algorithms on a GPU/TPU makes training much faster, since GPU's/TPU's are better suited for matrix operations
 
 Note: If you just want to see how the model is built, you can view my notebook [here](https://colab.research.google.com/drive/1Y1hPUXaOAhSJv93rvXZ6p27tbBUvn0zN)

 Before building the model, we need to get the data, which is hosted on Kaggle. To load the data, we need to use a package to download the data to our notebook, using the [Kaggle API](https://github.com/Kaggle/kaggle-api).
 This will allow us to download the dataset for the Dog Breed Competition. Before we can download the dataset, we need to create an account on Kaggle, and get your Kaggle API key and secret. 

 ![](images/kaggle-create-api-token.jpg)
 Go to "Create New API Token", and save the file to your machine.
 Then

 To download the data, we'll run this [cell](https://colab.research.google.com/drive/1Y1hPUXaOAhSJv93rvXZ6p27tbBUvn0zN#scrollTo=AbUjzDJBYVLq&line=24&uniqifier=1)
 ```python
 # Run this cell and select the kaggle.json file downloaded
# from the Kaggle account settings page.
from google.colab import files
files.upload()
# Let's make sure the kaggle.json file is present.
!ls -lha kaggle.json
# Next, install the Kaggle API client.
!pip install -q kaggle

# The Kaggle API client expects this file to be in ~/.kaggle,
# so move it there.
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

# This permissions change avoids a warning on Kaggle tool startup.
!chmod 600 ~/.kaggle/kaggle.json

#download the dataset for the dog-breed identification challenge https://www.kaggle.com/c/dog-breed-identification
!kaggle competitions download -c dog-breed-identification

#unzip the downloaded files
!unzip labels.csv.zip
!unzip test.zip
!unzip train.zip
```
When you run this cell, it will prompt to enter a file. Find the JSON file that was downloaded from Kaggle, and upload to the cell. You will then be able to run the Kaggle API and download the dataset into the notebook. The dataset contains the following:
*Training Images.
*Test Images.
*CSV containing the breed name and the filename, which points to the image in the training folder.

Now, we can load our data into a Dataframe, using Pandas, which is another python package that allows you to easily load [DataFrames](https://www.tutorialspoint.com/python_pandas/python_pandas_dataframe.htm), which is a simple data structure that contains rows and columns, kind of like a CSV. 

[Source](https://colab.research.google.com/drive/1Y1hPUXaOAhSJv93rvXZ6p27tbBUvn0zN#scrollTo=k59JDzJ-Ysl-&line=8&uniqifier=1)
```python
#import the neccesary packages
import pandas as pd
import numpy as np

#constants
num_classes = 12 # the number of breeds we want to classify
seed = 42 # makes the random numbers in numpy predictable
im_size = 299 # This size of the images
batch_size = 32

#read the csv into a dataframe, group the breeds and 
df = pd.read_csv('labels.csv')
selected_breed_list = list(df.groupby('breed').count().sort_values(by='id', ascending=False).head(num_classes).index)
df = df[df['breed'].isin(selected_breed_list)]
df['filename'] = df.apply(lambda x: ('train/' + x['id'] + '.jpg'), axis=1)


breeds = pd.Series(df['breed'])
print("total number of breeds to classify",len(breeds.unique()))

df.head()
```

This prints out the first 10 rows in the dataset that we created. 

![](images/df_head.png)

Note, for this training, we are only going to train the 12 most popular breeds. The reason for this is because training for all the breeds(120), takes up a lot of memory, which actually crashes Google Colab. In order to get around this, I've trained the same model on a Google Cloud instance. Check out https://towardsdatascience.com/running-jupyter-notebook-in-google-cloud-platform-in-15-min-61e16da34d52 for more info


Next, we need to write a function that will resize all the images to the size we need, which is 299x299px. It will be clear *why* we need to resize the image later

[Source](https://colab.research.google.com/drive/1Y1hPUXaOAhSJv93rvXZ6p27tbBUvn0zN#scrollTo=LuG1lYfswAtO)
```python
from keras.preprocessing import image

def read_img(img_id, train_or_test, size):
    """Read and resize image.
    # Arguments
        img_id: string
        train_or_test: string 'train' or 'test'.
        size: resize the original image.
    # Returns
        Image as numpy array.
    """
    path =  train_or_test + "/" + img_id + ".jpg"
    img = image.load_img(path, target_size=size)
    return image.img_to_array(img)
```

This uses a function inside Keras to load the image at the size we need(299x299) and converts it to a multi-dimensional numpy array (matrix)

Next, we need to convert the labels("basenji", "scottish_deerhound") into vectors, since our machine learning model can only deal with numbers. To do this, we'll use Scikit-Learn's [LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)

Then we'll split the dataset into 2 vectors, one for training and the other for testing. When we train our model, we'll use the data from the training set to train the model, then when we need to see how well it did, we'll test the model on the test set

[Source](https://colab.research.google.com/drive/1Y1hPUXaOAhSJv93rvXZ6p27tbBUvn0zN#scrollTo=Oc74pGmVvQKx)
```python
from sklearn.preprocessing import LabelEncoder
label_enc = LabelEncoder()
np.random.seed(seed=seed)
rnd = np.random.random(len(df))
train_idx = rnd < 0.9
valid_idx = rnd >= 0.9
y_train = label_enc.fit_transform(df["breed"].values)
ytr = y_train[train_idx]
yv = y_train[valid_idx]
```

Finally, we'll take all the images in the training set, and resize them using the `read_img` function. Then we need to process each image to put it in the correct format that our Model is expecting using [xception.preprocess_input](https://stackoverflow.com/a/47556342/457901)

[Source](https://colab.research.google.com/drive/1Y1hPUXaOAhSJv93rvXZ6p27tbBUvn0zN#scrollTo=Oc74pGmVvQKx&line=9&uniqifier=1)
```python
from sklearn.preprocessing import LabelEncoder
label_enc = LabelEncoder()
np.random.seed(seed=seed)
rnd = np.random.random(len(df))
train_idx = rnd < 0.9
valid_idx = rnd >= 0.9
y_train = label_enc.fit_transform(df["breed"].values)
ytr = y_train[train_idx]
yv = y_train[valid_idx]
```

[Source](https://colab.research.google.com/drive/1Y1hPUXaOAhSJv93rvXZ6p27tbBUvn0zN#scrollTo=pu1lsrkVu336&line=10&uniqifier=1)
```python
from tqdm import tqdm
from keras.applications import xception

x_train = np.zeros((train_idx.sum(), im_size, im_size, 3), dtype='float32')
x_valid = np.zeros((valid_idx.sum(), im_size, im_size, 3), dtype='float32')
train_i = 0
valid_i = 0
for i, img_id in tqdm(enumerate(df['id'])):
    img = read_img(img_id, 'train', (im_size, im_size))
    x = xception.preprocess_input(np.expand_dims(img.copy(), axis=0))
    if train_idx[i]:
        x_train[train_i] = x
        train_i += 1
    elif valid_idx[i]:
        x_valid[valid_i] = x
        valid_i += 1
print('Train Images shape: {} size: {:,}'.format(x_train.shape, x_train.size))

[00:06, 201.73it/s]Train Images shape: (1218, 299, 299, 3) size: 326,671,254
```

Lets go over what `xception.preprocess_input` is. For our model to get good results, we can't just give it all the images, and expect our model to learn. We don't have enough images and the patience to train this. It would like lots of images and lots of compute time to train. Luckily, we can use [Transfer Learning](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html). For Transfer learning, we can use the features from a model that has been previously trained on another dataset, like [Imagenet](www.image-net.org/) and use this for our training.

![](https://cdn-images-1.medium.com/max/2000/1*L8NWufrce1Bt9aDIN7Tu4A.png)
source: https://medium.com/@14prakash/transfer-learning-using-keras-d804b2e04ef8

From the photo, you can see that `Edges`, `Shapes` and `High Level features` have been already trained using lots of images, and compute time. What we will do is use all these layers, except for the `Classifiers`. We will then add our own layers to train on, using the images from our dataset

For our model, we'll be using [Xception](https://keras.io/applications/#xception) as our base model. In my experiments, i've seen that this model gives us really good results. For any other datasets, there may be other models that are more suited.  