import os
import numpy as np
import streamlit as st
from PIL import Image
import random

from keras.models import load_model
from keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template


MODEL_PATH ='Malariya_TFL_Erlystop1(83).h5'

# Load your trained model
model = load_model(MODEL_PATH)

#prediction part
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255#part of image processing (we also did the same in data augmentation while training)
    x = np.expand_dims(x, axis=0)
    
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    #since {'Parasite': 0, 'Uninfected': 1}
    #because we are using softmax activation fn so it will give two probabilty which will be the highest that image will belongs to that class
    if preds==0:
        preds="The Person is Infected "
    else:
        preds="The Person is not Infected"
    return preds

#main part of the api
def main():
    st.title("Streamlit with Deeplearning") #title
    st.set_option('deprecation.showfileUploaderEncoding', False)#disable the warnings
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Malariya Predictor DL App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)#for html
    uploaded_file = st.file_uploader("Provide cell image...", type=["png","jpg","jpeg"])#file uploader
    
    if uploaded_file is not None:#will execute when a file is selected
        image1 = Image.open(uploaded_file)#reading the file using PIL
        #our image=image1, PIL=Image, Keras=image 
        st.image(image1, caption='Uploaded Image.', use_column_width=True)#to display the uploaded image
        st.write("")
        st.write("Classifying...")
        st.spinner(text='In progress...')
        n=random.randrange(1,900)#to give random number
        p=random.randrange(1,900)
        #we will save that file first bcz streamlit st.file_uploader takes the file as a buffer only
        path1='C:/Users/shalo/Desktop/ML stuffs/Dockers/uploads%s'%(str(n))
        os.mkdir(path1)
        path2 = '%s/%d.png'%(path1,p)
        image1.save(path2)
        #path2 is the path of uploaded saved file        
        label = model_predict(path2,model)
        st.write("Output:",label)
        
        if st.button("Check again"):
            st.text("Browse files above to check again")


if __name__ == '__main__':
    main()#when this condition is hitted then main function will be executed
    
