import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import date

st.set_page_config(
    page_title="VisaFlow",
    page_icon="🌍",
    layout="centered"
)

#Load pipeline

@st.cache_data
def load_pipeline():
    return joblib.load(r"models\gradient_boost_pipeline_tuned.joblib")
pipeline=load_pipeline()

#UI

st.title("VisaFlow: AI Enabled Visa Processing Days Estimator")
st.caption("Predict your visa application status and estimate processing time using AI.")
st.markdown("---")

st.subheader("Get started")

#User Inputs

col1,col2=st.columns(2)
delay_options={
    "Less than 30 days":"Fast",
    "31-50 days":"Normal",
    "51-90 days":"Delayed",
    "More than 90 days":"Severely Delayed"
}

with col1:
    st.markdown("### Applicant Details")
    app_date=st.date_input("Application Date")
    age=st.number_input("Age",min_value=1,max_value=100)
    gender=st.selectbox("Gender",['Female','Male','Other' ])
    app_country=st.selectbox("Applicant Country",["Australia", "UAE", "New Zealand", "UK", "Germany", "USA", "Canada",
                                                  "France", "Japan", "Netherlands", "Switzerland", "Singapore"])
    processing_center=st.selectbox("Processing Center",["Pakistan_Main_Center", "France_Main_Center", "Montreal_Center", "Beijing_Center",
                                                        "Nigeria_Main_Center", "Cebu_Center", "Iran_Main_Center", "Spain_Main_Center",
                                                        "South Korea_Main_Center", "Turkey_Main_Center", "Guadalajara_Center",
                                                        "Thailand_Main_Center", "Juarez_Center", "Peru_Main_Center", "Italy_Main_Center",
                                                        "Berlin_Center", "Kenya_Main_Center", "Manila_Center", "Toronto_Center", "Colombia_Main_Center",
                                                        "Egypt_Main_Center", "Vietnam_Main_Center", "Brasilia_Center", "Frankfurt_Center", "Sao Paulo_Center",
                                                        "Russia_Main_Center", "Vancouver_Center", "London_Center", "Chengdu_Center", "Japan_Main_Center",
                                                        "Bangladesh_Main_Center", "Guangzhou_Center", "Manchester_Center", "Mexico City_Center",
                                                        "Chennai_Center", "Mumbai_Center", "Monterrey_Center", "Kolkata_Center", "Rio de Janeiro_Center",
                                                        "Hyderabad_Center", "Shanghai_Center", "Delhi_Center"])

with col2:
    st.markdown("### Visa Details")
    destination_country=st.selectbox("Destination Country",["Pakistan", "Vietnam", "France", "Canada", "China", "Nigeria",
                                                            "Philippines", "Iran", "Spain", "South Korea", "Turkey", "Mexico",
                                                            "Thailand", "Peru", "Italy", "Germany", "Kenya", "Colombia", "Egypt",
                                                            "Brazil", "Russia", "UK", "Japan", "Bangladesh", "India"])
    visa_type=st.selectbox("Visa Type",['Diplomatic','Business','Work','Student','Tourist','Transit','Family'])
    visa_class=st.selectbox("Visa Class",['Diplomatic Visa', 'Business Visitor', 'Work Permit', 'Student Tier 4', 'Skilled Worker',
                                          'Tourist Visa', 'F1', 'C1', 'B1', 'Schengen Tourist', 'Family Reunion', 'Study Permit',
                                          'B2', 'L1', 'Schengen Business', 'CR1', 'Visitor Subclass 600', 'Partner Visa', 'M1',
                                          'Employment Pass', 'H1B', 'Unknown', 'IR1', 'A1', 'Transit Visa'])
    user_choice=st.selectbox("Current Delay Status",list(delay_options.keys()))
    delay_status=delay_options[user_choice]


#Pre-processing the input data
application_month=app_date.month

country_avg_map=joblib.load(r"models\country_avg_map.joblib")
visa_type_avg_map=joblib.load(r"models\visa_type_avg_map.joblib")
visa_class_avg_map=joblib.load(r"models\visa_class_avg_map.joblib")

input_df=pd.DataFrame([{
    "Application_Month":application_month,
    "Peak_Season":"Peak" if application_month in [5,6,7,8] else "Off-Peak",
    "countryAvg":country_avg_map.get(app_country),
    "visa_type_Avg":visa_type_avg_map.get(visa_type),
    "Delay_Status":delay_status,
    "visa_class_Avg":visa_class_avg_map.get(visa_class)
}])


#Prediction


if st.button("Predict Visa Approval"):
    prediction=pipeline.predict(input_df)[0]
    st.success(f"Approval Days: {prediction:.0f}")


#Disclaimer
st.markdown("---")
st.warning("This tool provides probabilistic estimates only and does not constitute legal advice.")
