#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
#get_ipython().system('pip install streamlit')
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input


# In[2]:


model = tf.keras.models.load_model("D:/Sarang/Project_final/Pothole_Detection/model.h5")
### load file
uploaded_file = st.file_uploader("Choose a image file", type="jpg")


# In[3]:


map_dict = {0: 'Normal_road',
            1: 'Pothole_road'}


# In[4]:


if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(256,256))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")
    
    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Generate Prediction")    
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        st.title("Predicted Label for the image is {}".format(map_dict [prediction]))


# In[6]:




