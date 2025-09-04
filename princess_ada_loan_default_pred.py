import pandas as pd
import numpy as np
import streamlit as st 
import joblib
import os
import plotly.graph_objects as go

# --- Page Config ---
st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="💳",
    layout="wide"
)

# --- Base CSS ---
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stMetric {
        border-radius: 12px;
        padding: 10px;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True
)

# --- Load Model ---
model = joblib.load("loan_default.pkl")

# --- Load Image from assets folder ---
image_path = os.path.join("assets", "loan_default.jpg")
if os.path.exists(image_path):
    st.image(image_path, use_container_width=True)
else:
    st.warning("⚠️ Loan default image not found in assets folder.")

# --- App Header ---
st.title("💳 Loan Default Risk Prediction")
st.markdown(
    """
   Welcome to the **Loan Default Prediction App**.  
   Enter client details below and get an **instant risk evaluation** with a simulated credit score.
    """
)
st.markdown("---")

# --- Input Form ---
with st.form("loan_form"):

    # --- Segmentation ---
    st.subheader("💰 Loan Details")
    col1, col2 = st.columns(2)
    with col1:
        loanamount = st.number_input("Loan Amount (₦)", 100, 1_000_000, 50_000)
        termdays = st.number_input("Loan Term (days)", 10, 720, 90)
    with col2:
        repayment_curr_ratio = st.number_input("Repayment Current Ratio", 0.0, 2.0, 1.0)
        num_prev_loans = st.number_input("Number of Previous Loans", 0.00, 50.00, 3.00)

    st.subheader("📊 Payment History")
    col3, col4 = st.columns(2)
    with col3:
        avg_repay_delay_days = st.number_input("Average Repay Delay (days)", -50.00, 365.00, 10.00)
        total_firstrepaid_late = st.number_input("Total First Repaid Late", 0.00, 50.00, 2.00)
    with col4:
        avg_prev_repayment_ratio = st.number_input("Avg Previous Repayment Ratio", 0.0, 2.0, 1.0)
        avg_duration_days = st.number_input("Avg Duration of Previous Loans (days)", 0.00, 720.00, 180.00)

    st.subheader("📈 Financial History")
    col5, col6 = st.columns(2)
    with col5:
        avg_prev_interest = st.number_input("Avg Previous Interest (₦)", 0.00, 100000.00, 5000.00)
    with col6:
        age = st.number_input("Client Age", 18, 100, 30)
        age_group = st.selectbox("Age Group", ['Young adults','adults','middle-aged adults'])

    st.subheader("🏦 Banking & Employment Profile")
    col7, col8 = st.columns(2)
    with col7:
        bank_name_clients = st.selectbox("Bank Name",[
            'GT Bank','Sterling Bank','Fidelity Bank','Access Bank','EcoBank','FCMB','Skye Bank',
            'UBA','Zenith Bank','Diamond Bank','First Bank','Union Bank','Stanbic IBTC',
            'Standard Chartered','Heritage Bank','Keystone Bank','Unity Bank','Wema Bank'
        ])
    with col8:
        bank_account_type = st.selectbox("Bank Account Type", ['Other', 'Savings', 'Current'])
        employment_status_clients = st.selectbox("Employment Status", 
            ['Permanent', 'Unknown', 'Unemployed', 'Self-Employed', 'Student', 'Retired', 'Contract'])

    # --- Submit button ---
    submitted = st.form_submit_button("🚀 Predict Loan Default Risk")

if submitted:
    # --- Create dict ---
    user_input = {
        'loanamount': loanamount,
        'termdays': termdays,
        'repayment_curr_ratio': repayment_curr_ratio,
        'num_prev_loans': num_prev_loans,
        'avg_repay_delay_days': avg_repay_delay_days,
        'total_firstrepaid_late': total_firstrepaid_late,
        'avg_prev_repayment_ratio': avg_prev_repayment_ratio,
        'avg_duration_days': avg_duration_days,
        'avg_prev_interest': avg_prev_interest,
        'age': age,
        'bank_name_clients': bank_name_clients,
        'age_group': age_group,
        'bank_account_type': bank_account_type,
        'employment_status_clients': employment_status_clients
    }

    df = pd.DataFrame([user_input])

    # --- Feature Engineering ---
    df['late_payment_rate'] = df['total_firstrepaid_late'] / (df['num_prev_loans'] + 1e-6)
    df['repayment_efficiency'] = df['repayment_curr_ratio'] / (df['avg_prev_repayment_ratio'] + 1e-6)
    df['repayment_burden'] = df['loanamount'] / (df['termdays'] + 1e-6)
    
    df['sqrt_loanamount'] = np.sqrt(df['loanamount'])
    df['sqrt_termdays'] = np.sqrt(df['termdays'])
    df['sqrt_avg_prev_interest'] = np.sqrt(df['avg_prev_interest'])
    df['sqrt_repayment_burden'] = np.sqrt(df['repayment_burden'])
    df['sqrt_repayment_efficiency'] = np.sqrt(df['repayment_efficiency'])
    df['sqrt_late_payment_rate'] = np.sqrt(df['late_payment_rate'])

    features = [
        'repayment_curr_ratio', 'num_prev_loans', 'avg_repay_delay_days',
        'total_firstrepaid_late', 'avg_prev_repayment_ratio',
        'avg_duration_days', 'age','age_group',
        'sqrt_late_payment_rate', 'sqrt_termdays',
        'sqrt_loanamount', 'sqrt_avg_prev_interest','sqrt_repayment_burden',
        'sqrt_repayment_efficiency',
        'bank_account_type', 'employment_status_clients'
    ]
    
    X = df[features]

    # --- Prediction ---
    proba_good = model.predict_proba(X)[0, 1]
    min_score, max_score = 300, 850
    credit_score = min_score + (max_score - min_score) * proba_good

    # --- Classification & Dynamic Styling ---
    if credit_score < 575 or proba_good < 0.5:
        classification = "Bad"
        metric_css = """
        <style>
        .stMetric {
            background: linear-gradient(135deg, #CA0B00, #e63946); 
            color:white; border-radius:12px; padding:10px;
        }
        </style>
        """
        progress_color = "#CA0B00"
        gauge_color = "#CA0B00"
    else:
        classification = "Good"
        metric_css = """
        <style>
        .stMetric {
            background: linear-gradient(135deg, #10b981, #34d399); 
            color:white; border-radius:12px; padding:10px;
        }
        </style>
        """
        progress_color = "#10b981"
        gauge_color = "#10b981"

    st.markdown(metric_css, unsafe_allow_html=True)

    # --- Results Section ---
    st.markdown("---")
    st.subheader("📊 Risk Assessment Results")

    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("Credit Score", f"{credit_score:.0f}")
    with colB:
        st.metric("Repayment Probability", f"{proba_good:.2f}")
    with colC:
        st.metric("Loan Status", classification)

    # --- Progress Bar ---
    progress_value = int((credit_score - min_score) / (max_score - min_score) * 100)
    progress_css = f"""
    <style>
    .stProgress > div > div > div > div {{
        background-color: {progress_color} !important;
    }}
    </style>
    """
    st.markdown(progress_css, unsafe_allow_html=True)
    st.progress(progress_value)

    # --- Credit Score Gauge ---
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=credit_score,
        title={'text': "Credit Score"},
        gauge={
            'axis': {'range': [300, 850]},
            'bar': {'color': gauge_color},
            'steps': [
                {'range': [300, 575], 'color': "#CA0B00"},
                {'range': [575, 850], 'color': "#10b981"}
            ]
        }
    ))
    st.plotly_chart(gauge, use_container_width=True)

    # --- Classification Message ---
    if classification == "Good":
        st.markdown(
            '<div style="background-color:#10b981; padding:15px; border-radius:10px; color:white;">'
            '✅ Safe Loan: This client is likely to fulfill repayment (Good Loan)</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="background-color:#CA0B00; padding:15px; border-radius:10px; color:white;">'
            '❌ Risky Loan: This client is likely to default (Bad Loan)</div>',
            unsafe_allow_html=True
        )


# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        👩‍💻 <b>Built by:</b> Adaeze Princess Ekwuruke <br>
        📧 <b>Email:</b> princessada701@gmail.com <br>
        💼 <b>LinkedIn:</b> <a href="https://www.linkedin.com/in/adaeze-ekwuruke-7a4914155" target="_blank">adaeze-ekwuruke-7a4914155</a>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Disclaimer ---
st.markdown(
    """
    <div style="text-align: center; margin-top:20px; color:#555; font-size:14px;">
        ⚠️ <b>This tool provides risk assessment for informational purposes only.</b><br>
        Final lending decisions should consider additional factors and comply with applicable regulations.
    </div>
    """,
    unsafe_allow_html=True
)





