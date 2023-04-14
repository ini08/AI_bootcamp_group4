# Core Packages
import streamlit as st 
import altair as alt
import plotly.express as px 

# EDA Packages
import pandas as pd 
import numpy as np 
from datetime import datetime

# Utils
import joblib 
pipe_lr = joblib.load(open("spam_classifier.pkl","rb"))

# Fxn
def predict(docx):
	results = pipe_lr.predict([docx])
	return results[0]

def get_prediction_proba(docx):
	results = pipe_lr.predict_proba([docx])
	return results

hide_streamlit_style = """
            <style>
	    #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def create_footer():
    st.markdown("<div style='height: 7vh'></div>", unsafe_allow_html=True)
    footer_container = st.container()
    left_col, right_col = footer_container.columns(2)
    with left_col:
        st.write("")
    with right_col:
        st.write("Made with ❤️ by Group 4")

    st.markdown(
        """
        <script>
        const footer = document.getElementsByTagName('footer')[0];
        const appBody = document.getElementsByClassName('streamlit-container')[0];
        footer.style.position = 'fixed';
        footer.style.bottom = '0';
        appBody.style.paddingBottom = footer.offsetHeight + 'px';
        </script>
        """,
        unsafe_allow_html=True
    )

create_footer()


# Main Application
def main():
	st.title("Spam Classifier App")
	menu = ["Home","About"]
	choice = st.sidebar.selectbox("Menu",menu)
	if choice == "Home":
		st.subheader("Spam Classifier In Text")

		with st.form(key='spam_classifier_form'):
			raw_text = st.text_area("Type Here")
			submit_text = st.form_submit_button(label='Submit')

		if submit_text:
			col1,col2  = st.columns(2)

			# Apply Fxn Here
			prediction = predict(raw_text)
			probability = get_prediction_proba(raw_text)
			

			with col1:
				st.success("Original Text")
				st.write(raw_text)

				st.success("Prediction")
				st.write("{}".format(prediction))



			with col2:
				st.success("Prediction Probability")
				st.write(probability)

	else:
		st.subheader("About")
		st.write("Welcome to our spam classifier project! Our group has created a sophisticated machine learning model that can reliably recognise and categorize spam messages. With our spam classifier, you can say goodbye to pesky spam messages.")

		st.write("Our group is made up of five highly skilled individuals, each with a unique set of skills and expertise:")
		st.write("- **Ayodeji Adesegun**: He was in charge of preparing the data and creating the model. You can learn more about him from his [website](https://ayodejiades.vercel.app)")
		st.write("- **Abisola Lasisi**: She was in charge of writing the technical material for this project.")
		st.write("- **Inioluwa Adedapo**: In collaboration with Deborah Oladeji, she was in charge of training and deploying the model to the web using Streamlit.")
		st.write("- **Deborah Oladeji**: Along with Inioluwa Adedapo, she was in charge of training and deploying the model to the web using Streamlit.")
		st.write("- **James Sotomi**: He was in charge of preparing the data and building the model alongside Ayodeji.")

		st.write("Our spam classifier project is built using Python and several machine learning libraries, including Scikit-learn and NLTK. We have also developed this web application using Streamlit, which allows users to easily upload and classify their messages.")
		st.write("To use our spam classifier, simply upload your messages to our web application and click the 'submit' button. Our model will then analyze your messages and categorize them as either spam or legitimate (ham).")
		st.write("We hope that our spam classifier will be useful to you. Thank you for using our spam classifier!")
		
if __name__ == '__main__':
	main()
