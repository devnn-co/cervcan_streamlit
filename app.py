import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from joblib import dump, load

cancer_df = pd.read_csv('kag_risk_factors_cervical_cancer.csv')

cancer_df = cancer_df.replace('?', None)
for i in cancer_df.columns:
 cancer_df['Num_' + i] = pd.to_numeric(cancer_df[i])

mid_index = len(cancer_df.columns) // 2

second_half_cols = cancer_df.columns[mid_index:]

cancer_df[second_half_cols] = cancer_df[second_half_cols].fillna(cancer_df[second_half_cols].median())

mid_index_cols = len(cancer_df.columns) // 2

cancer_df = cancer_df.iloc[:, mid_index_cols:]

for i in cancer_df['Num_Dx']:
  if i < 1 and i > 0:
    cancer_df['Num_Dx'] = cancer_df['Dx'].replace(i, 0)

X = cancer_df[['Num_Dx:CIN', 'Num_Smokes', 'Num_Dx:Cancer', 'Num_Hormonal Contraceptives (years)']]
y = cancer_df['Num_Dx']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)

import streamlit as st

dump(decision_tree, "model.joblib")

def create_header():
  st.title("Using AI to Predict the Risk of Cervical Cancer")
  st.subheader("Making a prediction of a diagnosis of cervical cancer using a decision tree model.")
  st.write("This model has above 95% accuracy, but it is not 100%, so the predictions are suggestions and not indicative of a formal diagnosis.")
 
def get_user_input():
  CIN_diagnosis = st.number_input("Have you been diagnosed with cervical intraepithelial neoplasia (CIN)? Y: (1), N: (0)")
  cancer_diagnosis = st.number_input("Have you been diagnosed with any form of cancer in the past? Y: (1), N: (0) ")
  smokes = st.number_input("Do you or have you smoke(d) regularly? Y: (1), N: (0) ")
  year_hormonal_contraceptives = st.number_input("How many years have you been using hormonal contraceptives? If you've never used any, input: 0.")

  input_features = [[CIN_diagnosis, cancer_diagnosis, smokes, year_hormonal_contraceptives]]
  return input_features

def make_prediction(decision_tree, input):
  return decision_tree.predict(input)

def get_app_response(prediction):
  if prediction == 1:
    st.write("The model predicts are likely to be at risk for cervical cancer. Consider seeing a doctor for a cervical screening.")
  elif prediction == 0:
    st.write("The model predicts that your are not as likely to be at risk for cervical cancer.")
  else:
    st.write ("No results yet")


import streamlit as st
from joblib import load


# Load our DecisionTree model into our web app
model = load("model.joblib")
st.write ("Model uploaded!") # You may remove this in your finalized web app!

create_header()
input_features = get_user_input()
prediction = make_prediction(model, input_features)
get_app_response(prediction)
