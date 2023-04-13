# Core Pkgs
import streamlit as st 
import altair as alt
import plotly.express as px 

# EDA Pkgs
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
		st.write("My name is seyi ogunmusire ")
		footer="""<style>
		a:link , a:visited{
		color: blue;
		background-color: transparent;
		text-decoration: underline;}
		a:hover,  a:active {
		color: red;
		background-color: transparent;
		text-decoration: underline;
		}

		.footer {
		position: fixed;
		left: 0;
		bottom: 0;
		width: 100%;
		background-color: white;
		color: black;
		text-align: center;}
		</style>
		<div class="footer">
		<p>Developed with ❤ by <a style='display: block; text-align: center;' >Group 4</a></p>
		</div>
		"""
		st.markdown(footer,unsafe_allow_html=True)





if __name__ == '__main__':
	main()
