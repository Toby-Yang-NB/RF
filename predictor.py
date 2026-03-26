import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer

model =joblib.load('RF.pkl')

X_test = pd.read_csv('X_test.csv')

feature_names = [
    'age',
    'sex',
    'cp',
    'trestbps',
    'chol',
    'fbs',
    'restecg',
    'thalach',
    'exang',
    'oldpeak',
    'slope',
    'ca',
    'thal'
]

st.tile('心脏病预测')

age= st.number_input('年龄：',min_value=0,max_value=120, value=41)

sex= st.selection('性别：', options=[0,1], format_func=lambda  x:'男' if x==1 else '女')

cp= st.selectionbox('胸痛类型（cp）：', options =[0,1,2,3])

trestbps = st.number_input ('静息血压（trestbps）:', min_value=50, max_value=200, value=120)

chol =st.number_input ('胆固醇（chol）:', min_value=100,max_value=600, value=157)

fbs = st.selectbox('空腹血糖>120 mg/dl (fbs):', option=[0,1], format_func=lambda x:'是' if x==1 else'否')

restecg = st.selectbox('静息心电图（restecg）:', options=[0,1,2])

thalach = st.number_input('最大心率（thalach）:', min_value= 60, max_value=220, value=182)

exang= st.selectbox('运动引发的心绞痛（exang）:', options=[0,1], format_func=lambda x:'是' if x==1 else'否')

oldpeak =st.number_input('运动引起的st段抑制（oldpeak）:', min_value= 0.0, max_value=10.0, value=1.0)

slope = st.selectbox('运动峰值st段的坡度（slope）:', options=[0,1,2])

ca = st.selectbox('主要血管数量（ca）:', options=[0,1,2,3,4])

thal = st.selectbox('地中海贫血（thal）:', options=[0,1,2,3])

feature_values = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
features =np.array([feature_values])

if st.button('Predict'):
    predicted_class = model.predict(features)[0]

    predicted_proba = model.predict_proba(features)[0]

    st.write(f'**Predicted class:** {predicted_class} (1:Disease, 0: No Disease)')
    st.write(f'**Prediction Probabilities:** {predicted_proba}')

    probability = predicted_proba[predicted_class]*100

    if predicted_class == 1:
    advice= (
        f'According to our model, you have a high risk of heart disease.'
        f'The model predicts that your probability of having heart disease is {probability:.1f}%.'
        "It's advised to consult with your heathcare provider for further evaluation and possible intervention."
    )

else: advice= (
        f'According to our model, you have a low risk of heart disease.'
        f'The model predicts that your probability of not having heart disease is {probability:.1f}%.'
        "However, maintaining a healthy lifestyle is important. Please continue regular check-ups with your healthcare provider."
    )
st.write(advice)

st.subheader('SHAP force plot explanation')
explainer_shap = shap.TreeExplainer(model)
shape_values = explainer_shap.shap_values(pd.DataFrame([feature_values], columns=feature_names))

if predicted_class ==1:
    shap.force_plot(explainer_shap.expected_value[1], shap_values(:,:,1), pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
else:
    shap.force_plot(explainer_shap.expected_value[0], shap_values(:,:,0), pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig('shape_force_plot.png',bbox_inches='tight',dpi=1200)
    st.image('shap_force_plot.png',caption ='SHAP force plot explanation')
    st.subheader('LIME Explanation')
    lim_explainer =LimeTabularExplainer(
        training_data=X_test.values,
        feature_names=X_test.columns.tolist(),
        class_names=['Not sick','Sick'],
        mode='classification'
    )
    lime_exp =lime_explainer.explain_instance(
        data_row =features.flatten(),
        predict.fn=model.predict_proba
    )
    lime_html = lime_exp.as_html(show_table=False)
    st.components.v1.html(lime_html, height=800, scrolling=True)