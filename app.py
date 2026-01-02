import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import joblib

# Load model
@st.cache_resource
def load_model_and_scaler_encoder():
    model = joblib.load("model/bank_churn_model.pkl")
    std_scaler = joblib.load("model/standard_scaler.pkl")
    label_encoder = joblib.load("model/label_encoder.pkl")
    return model, std_scaler, label_encoder

model, std_scaler, label_encoder = load_model_and_scaler_encoder()

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(
        "Churn_Modelling.csv"
    )
    return df

df = load_data()

# UI
st.title("Bank Churn Prediction Streamlit App")

st.sidebar.header("Customer Details")

credit_score = st.sidebar.slider("Credit Score", 300, 900, 600)
geo = st.sidebar.selectbox("Geography", ['France', 'Germany', 'Spain'])
gender = st.sidebar.radio("Gender", ['Male', 'Female'])
age = st.sidebar.slider("Age", 18, 70, 35)
tenure = st.sidebar.slider("Tenure (years)", 0, 10, 5)
balance = st.sidebar.slider("Balance", 0.0, 500000.0, 50000.0)
num_products = st.sidebar.selectbox("Num of Products", [1, 2, 3, 4])
is_active_member = st.sidebar.selectbox("Is Active Member?", ['Yes', 'No'])
estimated_salary = st.sidebar.slider("Estimated Salary", 0.0, 200000.0, 100000.0)

# Convert categorical inputs
gender_encoded = 0 if gender == 'Male' else 1  # Match label_encoder (Male=0, Female=1 typically)
is_active_encoded = 1 if is_active_member == 'Yes' else 0
geo_germany = 1 if geo == 'Germany' else 0
geo_spain = 1 if geo == 'Spain' else 0

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_products],
    'EstimatedSalary': [estimated_salary],
    'Gender': [gender_encoded],
    'IsActiveMember': [is_active_encoded],
    'Geography_Germany': [geo_germany],
    'Geography_Spain': [geo_spain]
}, columns=['CreditScore', 'Age', 'Tenure', 'Balance', 
            'NumOfProducts', 'EstimatedSalary', 'Gender', 'IsActiveMember', 
            'Geography_Germany', 'Geography_Spain'])

# Scale numeric columns
num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
input_data[num_cols] = std_scaler.transform(input_data[num_cols])

input_data = input_data[model.feature_names_in_]

if st.sidebar.button("🔮 Predict Churn", type="primary"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    st.subheader("Prediction Results")
    st.metric("Churn Risk", "Yes" if prediction == 1 else "No", 
              f"{probability[1]:.1%}" if prediction == 1 else f"{probability[0]:.1%}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.success("✅ Customer likely to stay!" if prediction == 0 else "⚠️ Customer at risk!")
    with col2:
        st.info(f"Churn Probability: **{probability[1]:.1%}**")

# Optional: Dataset info
with st.expander("📈 Dataset Overview"):
    st.dataframe(df.head())
    st.write(f"Total customers: {len(df)}")

# Visualization
st.subheader("📊 Bank Customer Age Distribution")

df_vis = df.copy()
df_vis["Exited"] = df_vis["Exited"].map({0: "Stay", 1: "Churn"})

fig_age = px.histogram(
    df_vis,
    x="Age",
    color="Exited",
    nbins=30,
    title="Age Distribution vs Churn Label"
)
st.plotly_chart(fig_age)

st.subheader("📊 Churn Rate Based on Customer's Geography")

pivot_df = df.groupby(['Geography', 'Exited']).size().reset_index(name='count')
pivot_df['percentage'] = pivot_df.groupby('Geography')['count'].transform(
    lambda x: x / x.sum() * 100
)
pivot_df['status'] = pivot_df['Exited'].map({0: 'Stay', 1: 'Churn'})

fig = px.bar(
    pivot_df, 
    x='Geography', 
    y='percentage', 
    color='status',
    barmode='stack', 
    text='percentage',
    color_discrete_map={'Stay': '#4CAF50', 'Churn': '#F44336'}
)

fig.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
fig.update_layout(
    xaxis_title="Geography",
    yaxis_title="Percentage (%)",
    yaxis=dict(tickformat='.0f', range=[0, 100])
)

st.plotly_chart(fig, use_container_width=True)