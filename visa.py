import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta

# Settings
num_rows = 1000
start_date = datetime(2021, 1, 1)
end_date = datetime(2023, 12, 31)

# Countries
applicant_countries = [
    'India', 'China', 'Mexico', 'Philippines', 'Brazil', 
    'Vietnam', 'Nigeria', 'Pakistan', 'South Korea', 'UK', 
    'Canada', 'Germany', 'France', 'Colombia', 'Iran', 
    'Bangladesh', 'Egypt', 'Turkey', 'Russia', 'Japan'
]

visa_countries = [
    'USA', 'Canada', 'UK', 'Australia', 'Germany', 
    'New Zealand', 'France', 'Japan', 'Singapore', 'UAE'
]

# Visa Hierarchy (Type -> Classes)
visa_hierarchy = {
    'Work': ['H1B', 'L1', 'Work Permit', 'Skilled Worker', 'Employment Pass'],
    'Tourist': ['B2', 'Tourist Visa', 'Visitor Subclass 600', 'Schengen Tourist'],
    'Student': ['F1', 'M1', 'Student Tier 4', 'Study Permit'],
    'Business': ['B1', 'Business Visitor', 'Schengen Business'],
    'Family': ['IR1', 'CR1', 'Family Reunion', 'Partner Visa'],
    'Diplomatic': ['A1', 'Diplomatic Visa'],
    'Transit': ['C1', 'Transit Visa']
}

# Processing Centers
def get_center(country):
    centers = {
        'India': ['Mumbai', 'Delhi', 'Chennai', 'Hyderabad'],
        'China': ['Beijing', 'Shanghai', 'Guangzhou'],
        'Mexico': ['Mexico City', 'Guadalajara', 'Juarez'],
        'Philippines': ['Manila'],
        'Brazil': ['Brasilia', 'Rio de Janeiro', 'Sao Paulo'],
        'Canada': ['Toronto', 'Vancouver'],
        'USA': ['Washington DC'],
    }
    return f"{random.choice(centers.get(country, [country + '_Main']))}_Center"

data = []

for i in range(num_rows):
    # 1. Basic Demographics
    case_id = f"C-{100000 + i}"
    gender = random.choice(['Male', 'Female'])
    app_country = random.choice(applicant_countries)
    dest_country = random.choice([c for c in visa_countries if c != app_country])
    
    # 2. Visa Type & Class
    v_type = random.choices(list(visa_hierarchy.keys()), weights=[0.3, 0.3, 0.2, 0.1, 0.05, 0.02, 0.03])[0]
    v_class = random.choice(visa_hierarchy[v_type])
    
    # Age Logic
    if v_type == 'Student':
        age = random.randint(18, 30)
    elif v_type == 'Work':
        age = random.randint(22, 60)
    elif v_type == 'Business':
        age = random.randint(25, 65)
    elif v_type == 'Family':
        age = random.randint(1, 90)
    else:
        age = random.randint(18, 75)
        
    # 3. Dates & Processing Time
    days_range = (end_date - start_date).days
    app_date = start_date + timedelta(days=random.randint(0, days_range))
    
    # Processing Time Logic
    if v_type == 'Tourist':
        mu, sigma = 30, 10
    elif v_type == 'Business':
        mu, sigma = 20, 5
    elif v_type == 'Student':
        mu, sigma = 45, 15
    elif v_type == 'Work':
        mu, sigma = 90, 25 
    elif v_type == 'Family':
        mu, sigma = 180, 40
    else:
        mu, sigma = 60, 20
        
    proc_days = int(np.random.normal(mu, sigma))
    proc_days = max(1, proc_days)
    
    if app_date.month in [6, 7, 8]:
        proc_days += random.randint(5, 15)
        
    decision_date = app_date + timedelta(days=proc_days)
    
    # 4. Status
    status = random.choices(['Approved', 'Denied'], weights=[0.85, 0.15])[0]
    
    # 5. Center
    center = get_center(app_country)

    data.append([
        case_id, 
        app_date.strftime('%Y-%m-%d'), 
        decision_date.strftime('%Y-%m-%d'), 
        proc_days, 
        v_type, 
        v_class, 
        age, 
        gender, 
        app_country, 
        dest_country, 
        center, 
        status
    ])

columns = [
    'Case_ID', 'Application_Date', 'Decision_Date', 'Processing_Days', 
    'Visa_Type', 'Visa_Class', 'Applicant_Age', 'Gender', 
    'Applicant_Country', 'Visa_Country', 'Processing_Center', 'Case_Status'
]

df_final = pd.DataFrame(data, columns=columns)

# --- Introduce Missing Values ---
# Random NaNs
df_final.loc[df_final.sample(frac=0.05).index, 'Applicant_Age'] = np.nan
df_final.loc[df_final.sample(frac=0.03).index, 'Gender'] = np.nan
df_final.loc[df_final.sample(frac=0.02).index, 'Processing_Center'] = np.nan
df_final.loc[df_final.sample(frac=0.02).index, 'Visa_Class'] = np.nan

# "Pending" Cases (Structural NaNs)
pending_indices = df_final.sample(frac=0.05).index
df_final.loc[pending_indices, 'Decision_Date'] = np.nan
df_final.loc[pending_indices, 'Processing_Days'] = np.nan
df_final.loc[pending_indices, 'Case_Status'] = np.nan

df_final.to_csv('visa_dataset_missing.csv', index=False)
print("Dataset with missing values created!")