# Summary
- 1 Image data collection
- 2 Create data set (convert image data)
- 3 Model creation & learning
- 4 Execution (Terminal)

# Operating environment
- macOS Catalina 10.15 beta
- Python 3.6.8
- flickapi 2.4
- pillow 6.0.0
- scikit-learn 0.20.3
- google colaboratory

# Implementation procedure
## 1. Image data collection
- Image file is acquired from flickr to carry out image classification of 3 types (apple, tomato, strawberry)
- How to get the image file by flickr was written in the previous article [here] (https://qiita.com/hiraku00/items/dbdaad45ea54a35e51a4)
- Get 300 image files each
- Specify “apple”, “tomato”, “strawberry” as search keyword
- Unwanted data (image files unrelated to search keywords) downloaded from flickr are visually excluded

```python:download.py
from flickrapi import FlickrAPI
from urllib.request import urlretrieve
import os, time, sys

# Set your own API Key and Secret Key
key = "XXXXXXXXXX"
secret = "XXXXXXXXXX"
wait_time = 0.5

keyword = sys.argv[1]
savedir = "./data/" + keyword

flickr = FlickrAPI(key, secret, format='parsed-json')
result = flickr.photos.search(
    text = keyword,
    per_page = 300,
    media = 'photos',
    sort = 'relevance',
    safe_search = 1,
    extras = 'url_q, license'
)

photos = result['photos']

for i, photo in enumerate(photos['photo']):
    url_q = photo['url_q']
    filepath = savedir + '/' + photo['id'] + '.jpg'
    if os.path.exists(filepath): continue
    urlretrieve(url_q,filepath)
    time.sleep(wait_time)
```

## 2. Create data set (convert image data)
- Save the acquired image file in numpy format (binary file -> .npy)
- Resize to 224 of default size of VGG16

```python:generate_data.py
from PIL import Image
import os, glob
import numpy as np
from sklearn import model_selection

classes = ['apple', 'tomato', 'strawberry']
num_classes = len(classes)
IMAGE_SIZE = 224 # Specified size of VGG16 Default input size in VGG16

X = [] # image file
Y = [] # correct label

for index, classlabel in enumerate(classes):
    photo_dir = './data/' + classlabel
    files = glob.glob(photo_dir + '/*.jpg')
    for i, file in enumerate(files):
        image = Image.open(file)
        # standardize to 'RGB'
        image = image.convert('RGB')
        # to make image file all the same size
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        data = np.asarray(image)
        X.append(data)
        Y.append(index)

X = np.array(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)
np.save('./image_files.npy', xy)
```

## 3. Model creation & learning
### 1). Using Google Colaboratory
- Use Google Colaboratory, which can be used for free because it takes time for training processing (Environment construction unnecessary, Python execution environment on browser that can be used for free)
- This time, store "image_files.npy" created in "2." in Google Drive to Google Drive, and read the file from Google Colab

```python:mounting method
from google.colab import drive
drive.mount('/content/gdrive')
# Storage destination of image_files.npy (Create 'hoge' folder under My Drive and store it there)
PATH = '/content/gdrive/My Drive/hoge/'
```

### 2). Data loading & data conversion
- Load "image_files.npy" stored in google drive and divide it into training data and test data
- Convert the correct answer label to a one-hot vector (Ex: 0-> [1, 0, 0], 1-> [0, 1, 0])
- Standardize the data (convert the image data into the range of 0 to 1. Divide it by 255 because it is the range of (0, 0, 0) to (255, 255, 255) because it is RGB format)

```python:
X_train, X_test, y_train, y_test = np.load(PATH + 'image_files.npy', allow_pickle=True)

# convert one-hot vector
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

# normalization
X_train = X_train.astype('float') / 255.0
X_test = X_test.astype('float') / 255.0
```

### 3). Create a model
- Use the VGG16
- The three parameters are as follows.
  - <code>include_top</code> : Whether to include the three fully connected layers (Fully Connected Layer) on the output layer side of the network. This time, specify False to calculate the FC layer independently.
  - <code>weights</code> : Specify the type of weight of VGG16. Either None (random initialization) or 'imagenet' (weight learned by ImageNet)
  - <code>input_shape</code> : Optional shape tuple. Can only be specified if include_top is False (otherwise the shape of the input is (224, 224, 3). It must have exactly 3 input channels, width and height must be 48 or more)

```python:Create a model
vgg16_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)   
)
```

- Build an FC layer
- Specify the 1st and subsequent ones in the output shape of the above model in input_shape (the 0th contains the number)

```python:Build an FC layer
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(num_classes, activation='softmax'))
```

- Combine vgg16_model and top_model to create a model

```python:Combine a model
# combine models
model = Model(
    inputs=vgg16_model.input,
    outputs=top_model(vgg16_model.output)
)
model.summary()
```

```terminal:Output result of model.summary
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 224, 224, 3)       0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
_________________________________________________________________
sequential_1 (Sequential)    (None, 2)                 6423298   
=================================================================
Total params: 21,137,986
Trainable params: 21,137,986
Non-trainable params: 0
_________________________________________________________________

```

### 4). Fixed weight
- The model created above is a combination of the following two
 - vgg16_model: VGG16 excluding FC layer
 - top_model: Multilayer Perceptron
- Of these, fix the weight up to 'block4_pool' (see model.summary) of vgg16_model (in order to inherit high feature value extraction of VGG 16)

```python:fixed weight
for layer in model.layers[:15]:
    layer.trainable = False
```

### 5). Learning model
- optimizer specifies SGD
- Specify multiclass classification

```python:Learning model
opt = SGD(lr=1e-4, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, epochs=10)
```

### 6).Evaluation by test data

```python:Evaluation by test data
score = model.evaluate(X_test, y_test, batch_size=32)
print('loss: {0} - acc: {1}'.format(score[0], score[1]))
```

### 7). Save the model

```python:Save the model
model.save(PATH + 'vgg16_transfer.h5')
```

## 4. Execution (command line)
- Perform image file estimation using the created model (vgg16_transfer.h5)

```python:predict.py
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model, load_model
from PIL import Image
import sys

classes = ['apple', 'tomato', 'strawberry']
num_classes = len(classes)
IMAGE_SIZE = 224

# convert data by specifying file from terminal
image = Image.open(sys.argv[1])
image = image.convert('RGB')
image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
data = np.asarray(image)
X = []
X.append(data)
X = np.array(X)

# load model
model = load_model('./vgg16_transfer.h5')

# estimated result of the first data (multiple scores will be returned)
result = model.predict([X])[0]
predicted = result.argmax()
percentage = int(result[predicted] * 100)

print(classes[predicted], percentage)
```

- Execution is as follows (specify the image file name to be estimated in the argument)

```terminal:Execution
$ python predict.py XXXX.jpeg
```

```terminal:Result example
strawberry 100
```

# in Japanese
- [Qiita](https://qiita.com/hiraku00/items/66a3606af3b2eed57778)

