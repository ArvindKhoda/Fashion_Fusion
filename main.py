#Importing Required Libraries
from flask import Flask, render_template, request, url_for
import os

#Importing required dl libraries
import tensorflow as tf
from tensorflow import keras


#Loading Model
model = tf.keras.models.load_model("logo_clf.keras")

#Model Prediction
import cv2
import numpy as np
def predict_img(image,model):
    test_img=cv2.imread(image)
    test_img=cv2.resize(test_img,(224,224))
    test_img=np.expand_dims(test_img, axis=0)
    result=model.predict(test_img)
    if(result[0][0]>0.5):
        return "Fake"
    else:
        return "Genuine"


#App Creation
app=Flask(__name__)

#Routinng HomePage
@app.route('/')
def home():
    return render_template('home.html')

#Result Routing
@app.route("/result",methods=['POST'])
def result():
    f = request.files['Image']
    # "D:\3rd Year/Project/application/static/uploads/"
    file_path = os.path.join("static/upload/", f.filename)
    f.save(file_path)

    result=predict_img(file_path, model)
    return render_template('home.html',r=result)



#Running App
app.run(port=3000,debug=True)