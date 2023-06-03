
import base64
import streamlit as st
from PIL import ImageOps,Image 
import numpy as np

def classify_image(image,model, class_names):
    
    #convert the image to size
    image = ImageOps.fit(image,(64,64),Image.Resampling.LANCZOS)
    # convert the image in numpy array
    image_array = np.asarray(image)
    

    # normalize image
    noramlize_image_array = image_array.astype(np.float32)/255.0


    # set model input
    data = np.ndarray(shape=(1,64,64,3))
    data[0] = noramlize_image_array

    # make prediction
    prediction = model.predict(data)
    print(f"predictions : {prediction}")
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    print(f'Confidence_score: {prediction[0]}')
    return class_name, confidence_score