# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 15:34:44 2023

@author: Arvind
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 14:08:07 2023

@author: Arvind
"""

#Importing Required DL Libraries
import tensorflow as tf
from tensorflow import keras
from keras_preprocessing import image
from keras.layers import GlobalMaxPool2D
from keras.applications.resnet import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

#Model Creation
model=ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False

model=keras.Sequential([
    model,
    GlobalMaxPool2D()
])

print(model.summary())

#Feature Extraction Function
def extract_features(img_path,model):
   img=image.load_img(img_path,target_size=(224,224))
   img_array=image.img_to_array(img)
   expanded_img_array=np.expand_dims(img_array,axis=0)
   preprocessed_img=preprocess_input(expanded_img_array)
   result=model.predict(preprocessed_img).flatten()
   normalized_result=result/norm(result)
   return normalized_result

#working 
img_path="D:/3rd Year/Project/Dataset/images/-original-imafdfvvr8hqdu65.jpeg"
img=image.load_img(img_path,target_size=(224,224))
img_array=image.img_to_array(img)
expanded_img_array=np.expand_dims(img_array,axis=0)
preprocessed_img=preprocess_input(expanded_img_array)
result=model.predict(preprocessed_img).flatten()
normalized_result=result/norm(result)
return normalized_result



filename=[]

for file in os.listdir('images'):
    filename.append(os.path.join('images',file))

feature_list=[]

for file in tqdm(filename):
    feature_list.append(extract_features(file,model))

print(np.array(feature_list).shape)

pickle.dump(filename,open('filename.pkl','wb'))
pickle.dump(feature_list,open('feature_list.pkl','wb'))
pickle.dump(feature_list,open('feature_list.csv','wt'))


#Saving Feature Map in CSV Format
import pandas as pd
fl=pd.DataFrame(feature_list)
fl.to_csv('new.csv')



