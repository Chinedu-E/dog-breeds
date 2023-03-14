import streamlit as st
import tensorflow as tf
from utils import make_prediction

if "submit" not in st.session_state:
    st.session_state.submit = False

@st.cache
def load_labels():
    with open("labels.txt") as f:
        labels = f.read().split("\n")

    return labels


labels = load_labels()


st.set_page_config(layout="wide", page_title="Dog breed prediction")
st.title("Dog Breed Prediction")
st.header("Identify what breed your dog is!")
st.markdown("Upload an image of the dog")
    
    
image = st.file_uploader("Choose an image...",
                                type=["png", "jpeg", "jpg"])
# Uploading the image


submit = st.button("Predict")
if submit:
    st.session_state.submit = True
# On predict button click
if st.session_state.submit:
    
    if image is not None:

        img = tf.io.decode_image(image.read(), channels=3)
        # Displaying the image
        st.image(img.numpy(), channels="RGB")

        prediction, confidence = make_prediction(img, labels)
        st.title(f"The Dog Breed is {prediction} {confidence:.2f} confident")
        
        feedback = st.radio("Is the predicted breed correct?", ('Yes', 'No'), index=1)

        if feedback == 'No':
            actual_breed = st.text_input("What is the actual breed?", placeholder="Placeholder")
            if actual_breed:
                st.write("Thank you for that, we'll use your help to make a better model"+f"{actual_breed}")
        elif feedback == 'Yes':
            st.write("Thank you for your feedback!")
        else:
            ...
