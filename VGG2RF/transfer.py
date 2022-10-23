
from google.colab import drive
drive.mount('/content/drive')
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import glob
import cv2
import pickle
from keras.models import Sequential, Model
from keras.layers import Conv2D
import os
from keras.applications.vgg16 import VGG16

SIZE_Y = 2048 
SIZE_X = 1024

train_images = []

for directory_path in glob.glob("/content/drive/MyDrive/segment/U-net segment/JPEGImages"):
    for img_path in glob.glob(os.path.join(directory_path, "*.PNG")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)       
train_images = np.array(train_images)

train_masks = [] 
for directory_path in glob.glob("/content/drive/MyDrive/segment/U-net segment/SegmentationClass"):
    for mask_path in glob.glob(os.path.join(directory_path, "*.png")):
        mask = cv2.imread(mask_path, 0)       
        mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)
        train_masks.append(mask)          
train_masks = np.array(train_masks)

X_train = train_images[:7]
y_train = train_masks[:7]
# y_train = np.expand_dims(y_train, axis=0)
#y_train.shape

VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE_X, SIZE_Y, 3))

for layer in VGG_model.layers:
	layer.trainable = False
VGG_model.summary()

new_model = Model(inputs=VGG_model.input, outputs=VGG_model.get_layer('block1_conv2').output)
new_model.summary()

features=new_model.predict(X_train)

X=features
X = X.reshape(-1, X.shape[3])
Y = y_train.reshape(-1)

dataset = pd.DataFrame(X)
dataset['Label'] = Y
print(dataset['Label'].unique())
print(dataset['Label'].value_counts())

dataset = dataset[dataset['Label'] != 0]
X_for_RF = dataset.drop(labels = ['Label'], axis=1)
Y_for_RF = dataset['Label']

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 50, random_state = 42)

model.fit(X_for_RF, Y_for_RF)

filename = 'RF_model.sav'
pickle.dump(model, open(filename, 'wb'))

train_images[8].shape

loaded_model = pickle.load(open(filename, 'rb'))
test_img = train_images[9]
test_img = np.expand_dims(test_img, axis=0)

test_img.shape

X_test_feature = new_model.predict(test_img)
X_test_feature = X_test_feature.reshape(-1, X_test_feature.shape[3])
prediction = loaded_model.predict(X_test_feature)

prediction_image = prediction.reshape(mask.shape)
plt.imshow(prediction_image, cmap='gray')
plt.imsave('/content/test.jpg', prediction_image, cmap='gray')

