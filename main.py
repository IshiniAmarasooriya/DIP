import streamlit as st
import tensorflow as tf
import numpy as np



#Tensorflow Model Prediction

def model_prediction(test_image):

    model= tf.keras.models.load_model("trained_model.h5")

    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(64,64))

    input_arr=tf.keras.preprocessing.image.img_to_array(image)

    input_arr = np.array([input_arr])#convert single to batch

    predictions = model.predict(input_arr)

    return np.argmax(predictions)#return index of max element


#Sidebar

st.sidebar.title("Dashbord")

app_mode = st.sidebar.selectbox("Select page",["Home","About Project","Prediction"])


#Main Page

if(app_mode=="Home"):

    st.header("VEGETABLE RECOGNITION SYSTEM")

    image_path="home_img.jpg"
    st.image(image_path)


#About Project

elif(app_mode=="About Project"):

    st.header("About Project")

    st.subheader("About Dataset")

    st.text("This dataset contains of the following food items:")

    st.text("bean, bitter gourd, bottle gourd, brinjal, broccoli, cabbage, capsicum,carrot, cauliflower, cucumber, potato, pumpkin, radish and tomato")

    st.subheader("Content")

    st.text("This dataset contains three folders:")

    st.text("1. train (40 images each)")

    st.text("2. test (20 images each)")

    st.text("3. validation (20 images each)")
    

#Prediction

elif(app_mode=="Prediction"):

    st.header("Model Prediction")

    test_image=st.file_uploader("Choose an image:")

    if(st.button("Show Image")):

        st.image(test_image,width=4,use_column_width=True)

    #Predict button

    if(st.button("Predict")):

        st.write("Our Predection")
        result_index = model_prediction(test_image)


        #Reading labels

        with open("labels.txt.txt")as f:

            content = f.readlines()

        label = []
        
        

        for i in content:

            label.append(i[:-1])
        st.success("Model is predecting It's a {} ".format(label[result_index]) )  