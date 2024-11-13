# # import streamlit as st
# # from tensorflow.keras.models import load_model
# # from tensorflow.keras.preprocessing.image import img_to_array, load_img
# # import numpy as np

# # # Load the trained model
# # model = load_model('C:/Users/hp/Downloads/fomlproject/model_resnet50.h5')  # Replace with your actual model file path
# # class_names = ['Audi', 'Lmborghini','mercedes']  # Replace with your actual class labels
# # st.set_page_config(page_title="Image Classification App", layout="centered", page_icon="🔍", initial_sidebar_state="auto")
# # st.title("Image Classification App")

# # # Image uploader
# # uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# # if uploaded_file is not None:
# #     st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
# #     st.write("Classifying...")

# #     # Preprocess the image
# #     img = load_img(uploaded_file, target_size=(224, 224))
# #     img = img_to_array(img) / 255.0
# #     img = np.expand_dims(img, axis=0)

# #     # Make prediction
# #     predictions = model.predict(img)
# #     predicted_class = class_names[np.argmax(predictions)]
# #     confidence = np.max(predictions)

# #     st.write(f"Prediction: {predicted_class}")
# #     st.write(f"Confidence: {confidence:.2f}")
# import streamlit as st
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array, load_img
# import numpy as np

# # Load the trained model
# model = load_model('C:/Users/hp/Downloads/fomlproject/model_resnet50.h5')
# class_names = ['Audi', 'Lamborghini', 'Mercedes']

# # Set up the Streamlit page configuration
# st.set_page_config(page_title="Car Classification App", layout="centered", page_icon="🚗")

# # Add custom CSS to style the app
# st.markdown("""
#     <style>
#         .main {
#             background-color: #f5f5f5;
#             color: #333333;
#             font-family: Arial, sans-serif;
#         }
#         h1 {
#             color: #004d99;
#             font-size: 3em;
#             text-align: center;
#             margin-top: 0.5em;
#             margin-bottom: 0.2em;
#         }
#         .header-text {
#             color: #333;
#             font-size: 1.2em;
#             text-align: center;
#             margin-bottom: 1em;
#         }
#         .file-uploader {
#             text-align: center;
#         }
#         .stButton>button {
#             font-size: 1.1em;
#             padding: 0.5em 2em;
#             border-radius: 0.5em;
#             background-color: #004d99;
#             color: white;
#             border: none;
#         }
#         .stButton>button:hover {
#             background-color: #003366;
#             color: #e0e0e0;
#         }
#         .result-text {
#             font-size: 1.5em;
#             color: #004d99;
#             text-align: center;
#             margin-top: 1em;
#         }
#         .confidence-text {
#             font-size: 1.3em;
#             color: #888;
#             text-align: center;
#             margin-bottom: 2em;
#         }
#         input[type="text"] {
#             font-size: 1.2em;
#             color: #333333 !important; /* Ensures the text inside is visible */
#             padding: 0.5em;
#             border: 1px solid #ccc;
#             border-radius: 5px;
#             width: 100%;
#             background-color: #ffffff !important; /* Makes background white */
#             margin-top: 0.5em;
#         }
#     </style>
# """, unsafe_allow_html=True)

# # Display the app title and a brief welcome message
# st.title("Car Classification App 🚗")
# st.markdown("<p class='header-text'>Having trouble recognizing a car? We've got you covered!</p>", unsafe_allow_html=True)
# st.markdown("<p>Please enter your name:</p>", unsafe_allow_html=True)
# # Ask for user's name
# username = st.text_input("")

# # Greet the user and ask to upload a file
# if username:
#     st.write(f"Hey {username}, having trouble recognizing a car? We’ve got you covered!")

#     # Image uploader
#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="file_uploader")

#     if uploaded_file is not None:
#         st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
#         st.write("Classifying...")

#         # Preprocess the image
#         img = load_img(uploaded_file, target_size=(224, 224))
#         img = img_to_array(img) / 255.0
#         img = np.expand_dims(img, axis=0)

#         # Make prediction
#         predictions = model.predict(img)
#         predicted_class = class_names[np.argmax(predictions)]
#         confidence = np.max(predictions)

#         # Display the prediction result
#         st.markdown(f"<p class='result-text'>Prediction: <strong>{predicted_class}</strong></p>", unsafe_allow_html=True)
#         st.markdown(f"<p class='confidence-text'>Confidence: {confidence:.2f}</p>", unsafe_allow_html=True)

# import streamlit as st
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array, load_img
# import numpy as np
# import pandas as pd

# # Load the trained model and metadata
# model = load_model('C:/Users/hp/Downloads/FOMLcar/model_foml.h5')
# metadata = pd.read_csv("C:/Users/hp/Downloads/FOMLcar/companies.csv")
# class_names = [
#     'Acura', 'Alfa Romeo', 'Aston Martin', 'Audi', 'Bentley', 'BMW', 'Bugatti', 'Buick', 'Cadillac',
#     'Chevrolet', 'Chrysler', 'Citroen', 'Daewoo', 'Dodge', 'Ferrari', 'Fiat', 'Ford', 'Genesis', 'GMC', 'Honda',
#     'Hudson', 'Hyundai', 'Infiniti', 'Jaguar', 'Jeep', 'Kia', 'Land Rover', 'Lexus', 'Lincoln', 'Maserati', 'Mazda',
#     'Mercedes-Benz', 'MG', 'Mini', 'Mitsubishi', 'Nissan', 'Oldmobile', 'Peugeot', 'Pontiac', 'Porsche',
#     'Ram Trucks', 'Renault', 'Saab', 'Studebaker', 'Subaru', 'Suzuki', 'Tesla', 'Toyota', 'Volkswagen', 'Volvo'
# ]

# # Set up the Streamlit page configuration
# st.set_page_config(page_title="Car Classification App", layout="centered", page_icon="🚗")

# # Initialize session state variables
# if 'page' not in st.session_state:
#     st.session_state.page = 'main_page'
# if 'username' not in st.session_state:
#     st.session_state.username = ""

# # Home page to get the user's name
# def main_page():
#     st.title("Car Classification App 🚗")
#     st.markdown("<p class='header-text'>Having trouble recognizing a car? We've got you covered!</p>", unsafe_allow_html=True)
    
#     # Collect username and move to the classification page
#     username = st.text_input("Please enter your name:")
#     if st.button("Continue") and username:
#         st.session_state.username = username
#         st.session_state.page = "classification_page"

# # Classification page for uploading an image and displaying results
# def classification_page():
#     st.title("Car Classification App 🚗")
#     st.write(f"Hello, {st.session_state.username}! Upload an image, and we'll classify the car for you.")
    
#     # Image uploader
#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="file_uploader")
    
#     if uploaded_file is not None:
#         st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        
#         # Preprocess the image
#         img = load_img(uploaded_file, target_size=(224, 224))
#         img = img_to_array(img) / 255.0
#         img = np.expand_dims(img, axis=0)

#         # Make prediction
#         with st.spinner('Classifying...'):
#             predictions = model.predict(img)

#         # Ensure model output matches expected class count
#         if predictions.shape[1] == len(class_names):
#             predicted_index = np.argmax(predictions)
#             predicted_class = class_names[predicted_index]
#             confidence = np.max(predictions)
#         else:
#             st.error("The number of classes in the model's output does not match the class_names list.")
#             return

#         # Retrieve car metadata
#         car_metadata = metadata[metadata['name'] == predicted_class]
#         if not car_metadata.empty:
#             car_info = car_metadata.iloc[0]
#             origin = car_info['origin']
#             segment = car_info['segment']
#         else:
#             origin, segment = "Unknown", "N/A"

#         # Display results
#         st.markdown(f"### Prediction: **{predicted_class}**")
#         st.markdown(f"**Confidence**: {confidence:.2f}")
#         st.markdown(f"**Origin**: {origin}")
#         st.markdown(f"**Segment**: {segment}")

#         # Feedback section
#         feedback = st.radio("Was the prediction correct?", ["Select", "Yes", "No"])
#         if feedback == "Yes":
#             st.success("Thank you for your feedback!")
#         elif feedback == "No":
#             st.error("We'll work on improving the model.")

# # Main app flow
# if st.session_state.page == "main_page":
#     main_page()
# elif st.session_state.page == "classification_page":
#     classification_page()


import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import pandas as pd

# Load the trained model and metadata
model = load_model('C:/Users/hp/Downloads/fomlproject/model_resnet50.h5')
metadata = pd.read_excel("C:/Users/hp/Downloads/fomlproject/Book1.xlsx")
class_names = ['audi', 'lamborghini','mercedes']

# Set up the Streamlit page configuration
st.set_page_config(page_title="Brand Classification App", layout="centered", page_icon="🚗")

# Custom CSS to style the app
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        color: #f63366;
    }
    .sub-header {
        font-size: 1.2em;
        color: #999999;
        text-align: center;
    }
    .label {
        font-weight: bold;
        color: #f63366;
    }
    .result-box {
        padding: 10px;
        background-color: #2d2d2d;
        color: #f2f2f2;
        border-radius: 5px;
        margin-top: 10px;
        font-size: 1.1em;
    }
    .image-box {
        text-align: center;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'page' not in st.session_state:
    st.session_state.page = 'main_page'
if 'username' not in st.session_state:
    st.session_state.username = 'classification_page'

# Home page to get the user's name
def main_page():
    st.markdown("<div class='main-title'>Car Classification App 🚗</div>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Having trouble recognizing a car? We've got you covered!</p>", unsafe_allow_html=True)
    
    # Collect username and move to the classification page
    username = st.text_input("Please enter your name:")
    if st.button("Continue") and username:
        st.session_state.username = username
        st.session_state.page = "classification_page"

# Classification page for uploading an image and displaying results
def classification_page():
    st.markdown("<div class='main-title'>Car Classification App 🚗</div>", unsafe_allow_html=True)
    st.write(f"Hello, {st.session_state.username}! Upload an image, and we'll classify the car for you.")
    
    # Image uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="file_uploader")
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess the image
        img = load_img(uploaded_file, target_size=(224, 224))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Make prediction
        with st.spinner('Classifying...'):
            predictions = model.predict(img)

        # Ensure model output matches expected class count
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions)


        # Retrieve car metadata
        car_metadata = metadata[metadata['name'] == predicted_class]
        if not car_metadata.empty:
            car_info = car_metadata.iloc[0]
            origin = car_info['origin']
            segment = car_info['segment']
        else:
            origin, segment = "Unknown", "N/A"

        # Display results in styled boxes
        st.markdown(f"<div class='result-box'><span class='label'>Prediction:</span> {predicted_class}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-box'><span class='label'>Confidence:</span> {confidence:.2f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-box'><span class='label'>Origin:</span> {origin}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-box'><span class='label'>Segment:</span> {segment}</div>", unsafe_allow_html=True)

        # Feedback section
        feedback = st.radio("Was the prediction correct?", ["Select", "Yes", "No"])
        if feedback == "Yes":
            st.success("Thank you for your feedback!")
        elif feedback == "No":
            st.error("We'll work on improving the model.")

# Main app flow
if st.session_state.page == "main_page":
    main_page()
elif st.session_state.page == "classification_page":
    classification_page()