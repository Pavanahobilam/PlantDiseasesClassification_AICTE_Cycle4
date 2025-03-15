import streamlit as st
import tensorflow as tf
import numpy as np
import os
import cv2
from PIL import Image

# Load the model once at the start
model = tf.keras.models.load_model('cnn_model.keras')

def model_predict(image_path):
    img = cv2.imread(image_path)
    H, W, C = 224, 224, 3
    img = cv2.resize(img, (H, W))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img)
    img = img.astype('float32')
    img = img / 255.0
    img = img.reshape(1, H, W, C)

    prediction = np.argmax(model.predict(img), axis=-1)[0]
    return prediction

st.sidebar.title('Plant Disease Prediction System for Sustainable Agriculture')
app_mode = st.sidebar.selectbox('Select page', ['Home', 'Disease Recognition'])

# Display the main image
img = Image.open('Disease.png')
st.image(img)

if app_mode == 'Home':
    st.markdown("<h1 style='text-align: center;'>Plant Disease Prediction System for Sustainable Agriculture</h1>", unsafe_allow_html=True)

elif app_mode == 'Disease Recognition':
    st.header("Plant Disease Prediction System for Sustainable Agriculture")
    test_image = st.file_uploader("Choose an Image:", type=['jpg', 'jpeg', 'png'])

    if test_image is not None:
        # Save the uploaded image with a proper extension
        save_path = os.path.join(os.getcwd(), 'test_image' + os.path.splitext(test_image.name)[1])
        with open(save_path, 'wb') as f:
            f.write(test_image.getbuffer())

        # Display the uploaded image
        st.image(test_image, caption='Uploaded Image', use_column_width=True)

        if st.button("Predict"):
            with st.spinner("Predicting..."):
                result_index = model_predict(save_path)

            class_name = ['Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
                          'Blueberry__healthy', 'Cherry(including_sour)_Powdery_mildew', 
                          'Cherry_(including_sour)healthy', 'Corn(maize)_Cercospora_leaf_spot Gray_leaf_spot', 
                          'Corn_(maize)Common_rust', 'Corn_(maize)Northern_Leaf_Blight', 'Corn(maize)_healthy', 
                          'Grape__Black_rot', 'Grape_Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 
                          'Grape__healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Peach___Bacterial_spot',
                          'Peach__healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell__healthy', 
                          'Potato__Early_blight', 'Potato_Late_blight', 'Potato__healthy', 
                          'Raspberry__healthy', 'Soybean_healthy', 'Squash__Powdery_mildew', 
                          'Strawberry__Leaf_scorch', 'Strawberry_healthy', 'Tomato__Bacterial_spot', 
                          'Tomato__Early_blight', 'Tomato_Late_blight', 'Tomato__Leaf_Mold', 
                          'Tomato__Septoria_leaf_spot', 'Tomato__Spider_mites Two-spotted_spider_mite', 
                          'Tomato__Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus',
                          'Tomato___healthy']
            
            st.success("Model predicts that it is a {}".format(class_name[result_index]))
