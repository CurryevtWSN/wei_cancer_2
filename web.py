
#%%load package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import shap
import sklearn
import xgboost
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib

#%%不提示warning信息
st.set_option('deprecation.showPyplotGlobalUse', False)

#%%set title
st.set_page_config(page_title='Prediction Model of Ocular Metastases in Gastric Adenocarcinoma: Machine Learning–Based Development and Interpretation Study')
st.title('Prediction Model of Ocular Metastases in Gastric Adenocarcinoma: Machine Learning–Based Development and Interpretation Study')

#%%set variables selection
st.sidebar.markdown('## Variables')
LDL = st.sidebar.slider("Low-density lipoprotein(mg/dL)", 0.00, 20.00, value=1.00, step=0.01)
CEA = st.sidebar.slider("CEA(ng/mL)", 0.00, 100.00, value=7.00, step=0.01)
CA724 = st.sidebar.slider("CA724(ng/mL)", 0.00, 300.00, value=25.00, step=0.01)
CA125 = st.sidebar.slider("CA125(ng/mL)", 0.00, 500.00, value=25.00, step=0.01)
TC = st.sidebar.slider("Total cholesterol(mmol/L)", 0.00, 20.00, value=5.00, step=0.01)
Ca = st.sidebar.slider("Calcium(mmol/L)", 0.00, 10.00, value=1.00, step=0.01)
HDL = st.sidebar.slider("High density lipoprotein(mg/dL)", 0.00, 20.00, value=5.00, step=0.01)
AFP = st.sidebar.slider('Alpha-fetoprotein(ng/ml)',0.00, 500.00, value=10.00, step=0.01)
ALP = st.sidebar.slider("Alkaline phosphatase(U/L)", 0, 400, value=100, step=1)
CA153 = st.sidebar.slider("CA153(ng/mL)", 0.00, 200.00, value=40.00, step=0.01)
CA199 = st.sidebar.slider("CA199(ng/mL)", 0.00, 300.00, value=40.00, step=0.01)
TG = st.sidebar.slider("Triglyceride(mmol/L)", 0.00, 20.00, value=5.00, step=0.01)
Hb = st.sidebar.slider("Hemoglobin(g/L)", 0, 200, value=100, step=1)

#分割符号
st.sidebar.markdown('#  ')
st.sidebar.markdown('#  ')
st.sidebar.markdown('##### All rights reserved') 
st.sidebar.markdown('##### For communication and cooperation, please contact wshinana99@163.com, Wu Shi-Nan, Xiamen university')
#%%传入数据


#%%load model
gbm_model = joblib.load('/Users/mac/Desktop/gbm_model.pkl')

#%%load data
hp_train = pd.read_csv('/Volumes/吴世楠/WIN/E盘相关文件/Spyder_2022.3.29/output/machinel/sy_output/wei_cancer_em/Gastric_cancer.csv')
features = ['LDL',
            'CEA',
            'CA724',
            'CA125',
            'TC',
            'Ca',
            'HDL',
            'AFP',
            'ALP',
            'CA153',
            'CA199',
            'TG',
            'Hb']

target = ["M"]
y = np.array(hp_train[target])
sp = 0.5

is_t = (gbm_model.predict_proba(np.array([[LDL,CEA,CA724,CA125,TC,Ca,HDL,AFP,ALP,CA153,CA199,TG,Hb]]))[0][1])> sp
prob = (gbm_model.predict_proba(np.array([[LDL,CEA,CA724,CA125,TC,Ca,HDL,AFP,ALP,CA153,CA199,TG,Hb]]))[0][1])*1000//1/10


if is_t:
    result = 'High Risk metastasis'
else:
    result = 'Low Risk metastasis'
if st.button('Predict'):
    st.markdown('## Result:  '+str(result))
    if result == '  Low Risk Ocular metastasis':
        st.balloons()
    st.markdown('## Probability of High Risk Ocular metastasis group:  '+str(prob)+'%')
    #%%cbind users data
    col_names = features
    X_last = pd.DataFrame(np.array([[LDL,CEA,CA724,CA125,TC,Ca,HDL,AFP,ALP,CA153,CA199,TG,Hb]]))
    X_last.columns = col_names
    X_raw = hp_train[features]
    X = pd.concat([X_raw,X_last],ignore_index=True)
    if is_t:
        y_last = 1
    else:
        y_last = 0
    
    y_raw = (np.array(hp_train[target]))
    y = np.append(y_raw,y_last)
    y = pd.DataFrame(y)
    model = gbm_model
    #%%calculate shap values
    sns.set()
    explainer = shap.Explainer(model, X)
    shap_values = explainer.shap_values(X)
    a = len(X)-1
    #%%SHAP Force logit plot
    st.subheader('SHAP Force logit plot of GBM model')
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    force_plot = shap.force_plot(explainer.expected_value,
                    shap_values[a, :], 
                    X.iloc[a, :], 
                    figsize=(25, 3),
                    # link = "logit",
                    matplotlib=True,
                    out_names = "Output value")
    st.pyplot(force_plot)
    #%%SHAP Water PLOT
    st.subheader('SHAP Water plot of GBM model')
    shap_values = explainer(X) # 传入特征矩阵X，计算SHAP值
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    waterfall_plot = shap.plots.waterfall(shap_values[a,:])
    st.pyplot(waterfall_plot)
    #%%ConfusionMatrix 
    st.subheader('Confusion Matrix of GBM model')
    gbm_prob = gbm_model.predict(X)
    cm = confusion_matrix(y, gbm_prob)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Metastasis', 'Metastasis'])
    sns.set_style("white")
    disp.plot(cmap='RdPu')
    plt.title("Confusion Matrix of GBM")
    disp1 = plt.show()
    st.pyplot(disp1)