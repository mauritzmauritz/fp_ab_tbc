import streamlit as st
import pandas as pd
import numpy as np
import mahotas as mh
import matplotlib.pyplot as plt
# import pickle


st.title('Group 2 Tuberculosis Recognition')
st.header('Tuberculosis disease diagnosis')

IMM_SIZE = 224
data = st.file_uploader("Upload the lung x-ray to be diagnosed")
lab = {'Normal': 0, 'Tuberculosis': 1}

def diagnosis(file):
    # Download image
    ##YOUR CODE GOES HERE##
    
    image = mh.imread(file)

    # Prepare image to classification
    ##YOUR CODE GOES HERE##
      
    if len(image.shape) > 2:
        image = mh.resize_to(image, [IMM_SIZE, IMM_SIZE, image.shape[2]]) # resize of RGB and png images
    else:
        image = mh.resize_to(image, [IMM_SIZE, IMM_SIZE]) # resize of grey images    
    if len(image.shape) > 2:
        image = mh.colors.rgb2grey(image[:,:,:3], dtype = np.uint8)  # change of colormap of images alpha chanel delete
    
    # Show image
    ##YOUR CODE GOES HERE##

    plt.imshow(image)
    st.image(image)

    # Load model  
    ##YOUR CODE GOES HERE##

    from keras.models import model_from_json
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    model.load_weights("model.h5")        
    # with open('history.pickle', 'rb') as f:
    #     history = pickle.load(f)
    # print("Loaded model from disk")

    # Normalize the data
    ##YOUR CODE GOES HERE##

    image = np.array(image) / 255
    
    # Reshape input images
    ##YOUR CODE GOES HERE##

    image = image.reshape(-1, IMM_SIZE, IMM_SIZE, 1)
    
    # Predict the diagnosis
    ##YOUR CODE GOES HERE##

    predict_x=model.predict(image) 
    predictions=np.argmax(predict_x,axis=1)
    predictions = predictions.reshape(1,-1)[0]
    
    # Find the name of the diagnosis  
    ##YOUR CODE GOES HERE##

    diag = {i for i in lab if lab[i] == predictions}
    
    return diag

if data is not None:
    st.text(diagnosis(data))