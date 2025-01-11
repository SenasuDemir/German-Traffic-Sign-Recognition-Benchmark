import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

model = load_model('cnn_model.h5')

def process_image(img):
    img = img.convert('RGB')  
    img = img.resize((32, 32)) 
    img = np.array(img)  
    img = img / 255.0  
    img = np.expand_dims(img, axis=0) 
    return img

st.title('German Traffic Signs Detection :vertical_traffic_light:')
st.write('Upload a traffic sign image and the model will detect it.')

file = st.file_uploader('Select an image', type=['jpg', 'jpeg', 'png'])

if file is not None:
    # Display the uploaded image
    img = Image.open(file)
    st.image(img, caption='Uploaded Image')

    # Preprocess the image
    image = process_image(img)
    
    # Model prediction
    with st.spinner('Classifying the image...'):
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions)  
        predicted_prob = predictions[0][predicted_class]  

    # Class names for prediction
    class_names = ['Speed Limit 20', 'Speed Limit 30', 'Speed Limit 50', 'Speed Limit 60', 'Speed Limit 70', 'Speed Limit 80', 'End of a Speed Limit 80', 'Speed Limit 100', 'Speed Limit 120', 'Speed Limit 100', 'No overtaking by trucks', 'Crossroads', 'Priority Road', 'Give way', 'Stop', 'All vehicles prohibited in both directions', 'No trucks', 'No Entry', 'Other Hazards', 'Curve to left', 'Curve to right', 'Double curve, first to the left', 'Uneven Road', 'Slippery Road', 'Road Narrows Near Side', 'Roadworks', 'Traffic lights', 'No pedestrians', 'Children', 'Cycle Route', 'Be careful in winter', 'Wild animals', 'No parking', 'Turn right ahead', 'Turn left ahead', 'Ahead Only', 'Proceed straight or turn right', 'Proceed straight or turn left', 'Pass onto right', 'Pass onto left', 'Roundabout', 'No overtaking', 'End of Truck Overtaking Prohibition']
                      
    # Display the prediction
    st.subheader(f"Prediction: {class_names[predicted_class]}")
    st.write(f"Confidence: {predicted_prob * 100:.2f}%")

