import os

import pandas as pd
from tqdm import tqdm

# Importing Required Deep Learning Libraries

from tensorflow import keras
from keras_preprocessing import image
from keras.layers import GlobalMaxPool2D
from keras.applications.resnet import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
from PIL import Image


# Importing Other Required Libraries
import pandas as pd
import os


# Pinecone setup
from pinecone import Pinecone
pc = Pinecone(api_key="pcsk_5hckh7_4LCqvYcZuiZNK4SzStDrieiDBJ1gmb7LRkJqYnGd8AQCUiNz5AYY421GffPVWRG")
index = pc.Index("fashion-recommendation")


# Importing ResNet50
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = keras.Sequential([
    model,
    GlobalMaxPool2D()
])




# Assuming model is loaded globally
# from keras.applications.resnet50 import ResNet50
# model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_features(file_stream):
    """
    Extract features from an image using a pretrained model.

    Args:
        file_stream: BytesIO object containing image data.

    Returns:
        Normalized feature vector as a NumPy array.
    """
    img = Image.open(file_stream).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommender(features):
    """
    Query Pinecone index with the given features and return a DataFrame with results.
    """
    response = index.query(
        namespace="ns1",
        vector=features.tolist(),
        top_k=10,
        include_metadata=True,
        include_values=True,
    )

    data = []
    for match in response['matches']:
        meta = match['metadata']
        data.append({
            "ID": match['id'],
            "Score": match['score'],
            "Product_Image": meta.get("Product_Image"),
            "Product_Link": meta.get("Product_Link"),
            "Embedding": match['values']
        })

    df = pd.DataFrame(data)
    return df

def prs_recommender(features, namespace):
    """
    Query Pinecone index with the given features within a specific namespace.
    Returns a DataFrame with the top 5 matches.
    """
    index = pc.Index("virtualwardrobe")

    response = index.query(
        vector=features.tolist(),
        top_k=5,
        include_metadata=True,
        include_values=True,
        namespace=f"{namespace}_wardrobe"  # <-- search only in this namespace
    )


    data = []
    for match in response['matches']:
        meta = match['metadata']
        data.append({
            "ID": match['id'],
            "Score": match['score'],
            "filename": meta.get("filename"),
            "username": meta.get("username"),
            "embedding": match['values']
        })

    df = pd.DataFrame(data)
    return df



def extract_multiple_features(image_folder, model):
    """
    Extract features for all images in a folder.
    """
    filenames = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    feature_list = [extract_features(file, model) for file in tqdm(filenames)]
    relative_paths = [os.path.join('/ps_uploads/', os.path.basename(file)) for file in filenames]
    return feature_list, relative_paths


