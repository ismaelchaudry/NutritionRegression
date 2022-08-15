# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 01:20:15 2022

@author: scd10
"""

import streamlit as st
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.write("""
# Association between Nutrition and COVID Prediction
""")
st.write('---')

# Loads the Boston House Price Dataset
proteins = pd.read_csv('proteins.csv')
X = proteins[['Animal_Products', 'Milk_Excluding_Butter', 'Vegetal_Products', 'Animal_fats', 'Eggs']]
y = proteins['Deaths']
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    Animal_Products = st.sidebar.slider('Animal_Products', 4.0, 36.0, 21.0)
    Milk_Excluding_Butter = st.sidebar.slider('Milk_Excluding_Butter', 0.0, 17.0, 6.0)
    Vegetal_Products = st.sidebar.slider('Vegetal_Products', 14.0, 46.0, 28.0)
    Animal_Fats = st.sidebar.slider('Animal_fats', 0.0, 1.0, 0.1)
    Eggs = st.sidebar.slider('Eggs', 0.0, 4.0, 1.2)
    data = {'Animal_Products': Animal_Products,
            'Milk_Excluding_Butter': Milk_Excluding_Butter,
            'Vegetal_Products': Vegetal_Products,
            'Animal_fats': Animal_Fats,
            'Eggs': Eggs}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()


# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Build Regression Model
model = RandomForestRegressor()
model.fit(X, y)
# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Prediction of COVID Risk')
st.write(prediction)
st.write('---')