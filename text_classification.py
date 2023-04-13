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
pipe_lr = joblib.load(open("spam_classifier (1).pkl","rb"))

# Fxn
def predict(docx):
	results = pipe_lr.predict([docx])
	return results[0]

def get_prediction_proba(docx):
	results = pipe_lr.predict_proba([docx])
	return results

# Main Application
def main():
	st.title("spam classifier App")
	menu = ["Home","About"]
	choice = st.sidebar.selectbox("Menu",menu)
	if choice == "Home":
		st.subheader("spam classifier In Text")

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





if __name__ == '__main__':
	main()
