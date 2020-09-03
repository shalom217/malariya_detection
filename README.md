# Malariya_Detection
![alt text](https://github.com/shalom217/malariya_detection/blob/master/malariya.jpg)


Malaria is a disease of the blood that is caused by the Plasmodium parasite, which is transmitted from person to person by a particular type of mosquito.
Malaria is one of the world’s deadliest diseases, and remains one of the top child killers on the planet. Malaria also keeps children from going to school, families from investing in their future, and communities from prospering, taking a huge toll on lives, livelihoods and countries’ progress.



# Dataset
 Download dataset:https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria

# Transfer Learning techinque
This classifier is built using Transfer Learning techinque on one of the famous CNN architecture that is VGG19(using trained weights of imagenet dataset)which was itself built on imagenet dataset of 1000 differnt classes.
![alt text](https://github.com/shalom217/Transfer_learning/blob/master/transfer_l.jpeg)

# comparision between old/default VGG19model
Here a detailed comparision between old/default VGG19model(which was built to classify 1000 categories) and our custom model using trained weights of VGG19model and classifying only 2 classes('Parasite' and 'Uninfected').Have a look-----
![alt text](https://github.com/shalom217/malariya_detection/blob/master/COMPARISION.png)

Here we implementing(optimizer = 'adam') with Callbacks method having (EarlyStopping=83% accuracy and ModelCheckpoint=81% accuracy).Here Accuracy log is shown-----
![alt text](https://github.com/shalom217/malariya_detection/blob/master/accuracy_log1.png)

Model is trained over 5000 images of each categories('Parasite' and 'Uninfected')
# accuracyVSepoch and lossVSepoch curves

![alt text](https://github.com/shalom217/malariya_detection/blob/master/accVSepoch1.png)
![alt text](https://github.com/shalom217/malariya_detection/blob/master/lossVSepoch1.png)


We have also accuracyVSepoch and lossVSepoch curves of both test and train datasets. Go check it out----train the model, save it, and try predicting by your own.
# REQUIREMENTS---- 

![alt text](https://github.com/shalom217/images/blob/master/cuda_logo1.png)
![alt text](https://github.com/shalom217/images/blob/master/image_net_logo.jpg)
![alt text](https://github.com/shalom217/images/blob/master/keras_logo.png)
![alt text](https://github.com/shalom217/images/blob/master/opencv_logo.png)
![alt text](https://github.com/shalom217/images/blob/master/python_logo.jpg)
![alt text](https://github.com/shalom217/images/blob/master/tf_logo.png)

keras 2.3.1,
python 3.7, tensorflow 2.0.0, cuda installed, openCV 4.1.1.26, imagenet weights.

# Results--------------

![alt text](https://github.com/shalom217/malariya_detection/blob/master/pred1.png)


![alt text](https://github.com/shalom217/malariya_detection/blob/master/pred2.png)

# Deployment
As it is a end to end application, so i've also made an API. The front end of the API created using streamlit library.
![alt text](https://github.com/shalom217/malariya_detection/blob/master/stDL.png)




