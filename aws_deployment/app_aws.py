"""
app_aws.py — Streamlit App versi AWS
Sama dengan deployment/app.py tetapi pakai CreditScorePredictorS3
yang download artifact dari bucket S3 saat first load.

Deployment:
  - Di-jalankan otomatis oleh systemd service di EC2
  - Pakai env var S3_BUCKET dan S3_ARTIFACT_PREFIX
"""
import os
import sys
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

# Pastikan bisa import dari sibling folder
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(APP_DIR.parent / 'aws_deployment'))
sys.path.insert(0, str(APP_DIR.parent / 'pipeline'))

from aws_inference import CreditScorePredictorS3  # noqa: E402


st.set_page_config(page_title="Credit Score Classifier — AWS",
                   page_icon="☁️", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.4rem; font-weight: 700; color: #1a3a5c; }
    .subtitle { color: #5a6b7d; margin-bottom: 1.5rem; }
    .result-good { background: linear-gradient(135deg, #2ecc71, #27ae60);
                    padding: 1.5rem; border-radius: 12px; color: white;
                    text-align: center; font-size: 1.6rem; font-weight: 600; }
    .result-standard { background: linear-gradient(135deg, #3498db, #2980b9);
                        padding: 1.5rem; border-radius: 12px; color: white;
                        text-align: center; font-size: 1.6rem; font-weight: 600; }
    .result-poor { background: linear-gradient(135deg, #e74c3c, #c0392b);
                    padding: 1.5rem; border-radius: 12px; color: white;
                    text-align: center; font-size: 1.6rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    return CreditScorePredictorS3()


try:
    predictor = load_predictor()
except Exception as e:
    st.error(f"Gagal load model dari S3:\n\n{e}\n\n"
             f"Pastikan env var S3_BUCKET dan IAM role EC2 sudah benar.")
    st.stop()


with st.sidebar:
    st.markdown("### ☁️ Model Deployment")
    st.markdown(f"**Source**: `s3://{os.environ.get('S3_BUCKET','?')}/"
                f"{os.environ.get('S3_ARTIFACT_PREFIX','artifacts/')}`")
    st.markdown("---")
    st.caption("Nicholas Juari · 2802413064")


st.markdown('<div class="main-header">☁️ Credit Score Classifier (AWS)</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subtitle">Sistem penilaian performa kredit nasabah berbasis machine learning</div>',
            unsafe_allow_html=True)

with st.form("predict_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Umur", 14, 100, 35)
        occupation = st.selectbox("Pekerjaan", [
            'Scientist', 'Teacher', 'Engineer', 'Entrepreneur', 'Developer',
            'Lawyer', 'Media_Manager', 'Doctor', 'Journalist', 'Manager',
            'Accountant', 'Musician', 'Mechanic', 'Writer', 'Architect'])
        annual_income = st.number_input("Pendapatan tahunan ($)", 0.0, value=50000.0)
        monthly_inhand = st.number_input("Take-home bulanan ($)", 0.0, value=3500.0)

    with col2:
        num_bank = st.number_input("Rekening bank", 0, 15, 3)
        num_cc = st.number_input("Kartu kredit", 0, 15, 3)
        interest_rate = st.number_input("Suku bunga (%)", 0, 40, 12)
        num_loan = st.number_input("Loan aktif", 0, 15, 2)
        outstanding_debt = st.number_input("Outstanding debt ($)", 0.0, value=1500.0)

    with col3:
        delay_due = st.number_input("Keterlambatan (hari)", -10, 100, 5)
        num_delayed = st.number_input("Pembayaran telat", 0, 30, 3)
        credit_history_age = st.number_input("Riwayat kredit (bulan)", 0, 600, 120)
        credit_mix = st.selectbox("Credit Mix", ['Good', 'Standard', 'Bad'])
        payment_min = st.selectbox("Payment of Min Amount", ['Yes', 'No'])

    submitted = st.form_submit_button("🎯 Prediksi", type="primary",
                                       use_container_width=True)

if submitted:
    inp = {
        'Age': age, 'Occupation': occupation, 'Annual_Income': annual_income,
        'Monthly_Inhand_Salary': monthly_inhand, 'Num_Bank_Accounts': num_bank,
        'Num_Credit_Card': num_cc, 'Interest_Rate': interest_rate,
        'Num_of_Loan': num_loan, 'Delay_from_due_date': delay_due,
        'Num_of_Delayed_Payment': num_delayed, 'Changed_Credit_Limit': 5.0,
        'Num_Credit_Inquiries': 4, 'Credit_Mix': credit_mix,
        'Outstanding_Debt': outstanding_debt,
        'Credit_Utilization_Ratio': 30.0,
        'Credit_History_Age': f"{credit_history_age // 12} Years and {credit_history_age % 12} Months",
        'Payment_of_Min_Amount': payment_min,
        'Total_EMI_per_month': 150.0, 'Amount_invested_monthly': 100.0,
        'Payment_Behaviour': 'High_spent_Medium_value_payments',
        'Monthly_Balance': 300.0, 'Num_Loan_Types': 2
    }
    r = predictor.predict(inp)
    cls = r['predicted_class']
    css = {'Good': 'result-good', 'Standard': 'result-standard',
            'Poor': 'result-poor'}[cls]
    icon = {'Good': '✅', 'Standard': '⚖️', 'Poor': '⚠️'}[cls]
    st.markdown(f'<div class="{css}">{icon} Credit Score: <b>{cls}</b><br>'
                f'<span style="font-size:1rem;">Confidence: {r["confidence"]*100:.1f}%</span></div>',
                unsafe_allow_html=True)
    probs = r['probabilities']
    fig = go.Figure(go.Bar(
        x=list(probs.values()), y=list(probs.keys()), orientation='h',
        marker=dict(color=['#2ecc71' if k == 'Good' else
                            '#3498db' if k == 'Standard' else '#e74c3c'
                            for k in probs.keys()]),
        text=[f'{v*100:.1f}%' for v in probs.values()], textposition='auto'))
    fig.update_layout(height=240, xaxis=dict(range=[0, 1], tickformat='.0%'))
    st.plotly_chart(fig, use_container_width=True)
