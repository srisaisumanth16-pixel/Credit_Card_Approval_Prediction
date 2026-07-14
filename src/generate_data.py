import os
import numpy as np
import pandas as pd

def generate_synthetic_data(num_records=5000, seed=42):
    np.random.seed(seed)
    
    # 1. Generate IDs
    start_id = 5008804
    ids = np.arange(start_id, start_id + num_records)
    
    # 2. Demographic distributions
    genders = np.random.choice(['F', 'M'], size=num_records, p=[0.67, 0.33])
    own_car = np.random.choice(['N', 'Y'], size=num_records, p=[0.63, 0.37])
    own_realty = np.random.choice(['Y', 'N'], size=num_records, p=[0.69, 0.31])
    
    # Children & Family Members
    children = np.random.choice([0, 1, 2, 3, 4], size=num_records, p=[0.70, 0.20, 0.08, 0.015, 0.005])
    
    family_status = np.random.choice(
        ['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow'],
        size=num_records,
        p=[0.69, 0.14, 0.08, 0.06, 0.03]
    )
    
    # Calculate family members based on status and children
    fam_members = []
    for i in range(num_records):
        status = family_status[i]
        kids = children[i]
        if status in ['Married', 'Civil marriage']:
            fam_members.append(kids + 2)
        else:
            fam_members.append(kids + 1)
    fam_members = np.array(fam_members, dtype=float)
    
    # Income details (log-normal distribution for realistic income)
    income_base = np.random.lognormal(mean=11.9, sigma=0.5, size=num_records)
    # Clip income to reasonable bounds
    income_total = np.clip(income_base, 30000, 1000000).round(2)
    
    income_types = np.random.choice(
        ['Working', 'Commercial associate', 'Pensioner', 'State servant', 'Student'],
        size=num_records,
        p=[0.51, 0.23, 0.17, 0.09, 0.00] # Almost zero students
    )
    # Adjust some Pensioners to be unemployed later
    
    education_types = np.random.choice(
        ['Secondary / secondary special', 'Higher education', 'Incomplete higher', 'Lower secondary', 'Academic degree'],
        size=num_records,
        p=[0.68, 0.27, 0.04, 0.009, 0.001]
    )
    
    housing_types = np.random.choice(
        ['House / apartment', 'With parents', 'Municipal apartment', 'Rented apartment', 'Office apartment', 'Co-op apartment'],
        size=num_records,
        p=[0.89, 0.05, 0.03, 0.015, 0.01, 0.005]
    )
    
    # Age in Days (negative)
    # 20 to 68 years old -> -24820 to -7300 days
    age_years = np.random.uniform(20, 68, size=num_records)
    days_birth = (-age_years * 365.25).astype(int)
    
    # Days Employed (negative or 365243 for unemployed/retired)
    days_employed = []
    for i in range(num_records):
        inc_type = income_types[i]
        age = age_years[i]
        
        if inc_type == 'Pensioner' or age > 60 and np.random.rand() > 0.3:
            # 365243 represents unemployed/retired in this dataset
            days_employed.append(365243)
            # Override income type to Pensioner if not already
            income_types[i] = 'Pensioner'
        else:
            # Employed: max tenure is (age - 18)
            max_tenure = max(1, int(age - 18))
            tenure_years = np.random.exponential(scale=6.0)
            tenure_years = min(tenure_years, max_tenure)
            days_employed.append(-int(tenure_years * 365.25))
    days_employed = np.array(days_employed)
    
    # Flags
    flag_mobil = np.ones(num_records, dtype=int)
    flag_work_phone = np.random.choice([0, 1], size=num_records, p=[0.78, 0.22])
    flag_phone = np.random.choice([0, 1], size=num_records, p=[0.71, 0.29])
    flag_email = np.random.choice([0, 1], size=num_records, p=[0.91, 0.09])
    
    # Occupation Type
    occupations_list = [
        'Laborers', 'Core staff', 'Sales staff', 'Managers', 'Drivers', 
        'High skill tech staff', 'Accountants', 'Medicine staff', 'Cooking staff', 
        'Security staff', 'Cleaning staff', 'Private service staff', 'Low-skill Laborers', 
        'Secretaries', 'Waiters/barmen staff', 'HR staff', 'Realty agents', 'IT staff'
    ]
    occupations_probs = [
        0.30, 0.16, 0.15, 0.12, 0.08, 
        0.05, 0.04, 0.03, 0.02, 
        0.015, 0.01, 0.008, 0.005, 
        0.004, 0.003, 0.002, 0.002, 0.001
    ]
    
    occupation_type = []
    for i in range(num_records):
        if days_employed[i] == 365243:
            # Unemployed/Pensioners typically have NaN occupation
            occupation_type.append(np.nan)
        else:
            # Some working people also have missing occupation in real dataset
            if np.random.rand() > 0.75:
                occupation_type.append(np.nan)
            else:
                occ = np.random.choice(occupations_list, p=occupations_probs)
                occupation_type.append(occ)
                
    # Create DataFrame for application record
    df_app = pd.DataFrame({
        'ID': ids,
        'CODE_GENDER': genders,
        'FLAG_OWN_CAR': own_car,
        'FLAG_OWN_REALTY': own_realty,
        'CNT_CHILDREN': children,
        'AMT_INCOME_TOTAL': income_total,
        'NAME_INCOME_TYPE': income_types,
        'NAME_EDUCATION_TYPE': education_types,
        'NAME_FAMILY_STATUS': family_status,
        'NAME_HOUSING_TYPE': housing_types,
        'DAYS_BIRTH': days_birth,
        'DAYS_EMPLOYED': days_employed,
        'FLAG_MOBIL': flag_mobil,
        'FLAG_WORK_PHONE': flag_work_phone,
        'FLAG_PHONE': flag_phone,
        'FLAG_EMAIL': flag_email,
        'OCCUPATION_TYPE': occupation_type,
        'CNT_FAM_MEMBERS': fam_members
    })
    
    # 3. Generate Credit Record
    # Create credit history for a subset of IDs (e.g. 90% of them)
    active_ids = np.random.choice(ids, size=int(num_records * 0.95), replace=False)
    
    credit_rows = []
    for uid in active_ids:
        # Determine credit behavior score based on features
        row_app = df_app[df_app['ID'] == uid].iloc[0]
        
        # Risk score calculation (lower is better, meaning lower chance of bad credit)
        # Base risk
        risk_score = 0.5
        
        # Risk adjustments
        # Higher income reduces risk
        income = row_app['AMT_INCOME_TOTAL']
        if income > 250000:
            risk_score -= 0.2
        elif income < 80000:
            risk_score += 0.2
            
        # Higher education reduces risk
        edu = row_app['NAME_EDUCATION_TYPE']
        if edu in ['Higher education', 'Academic degree']:
            risk_score -= 0.15
        elif edu in ['Lower secondary']:
            risk_score += 0.15
            
        # Realty/Car ownership reduces risk
        if row_app['FLAG_OWN_REALTY'] == 'Y':
            risk_score -= 0.05
        if row_app['FLAG_OWN_CAR'] == 'Y':
            risk_score -= 0.05
            
        # Unemployed/Pensioner details
        emp_days = row_app['DAYS_EMPLOYED']
        if emp_days == 365243:
            # Retired or unemployed
            risk_score += 0.1
        else:
            # Long tenure reduces risk
            tenure_yrs = -emp_days / 365.25
            if tenure_yrs > 10:
                risk_score -= 0.2
            elif tenure_yrs < 2:
                risk_score += 0.1
                
        # Age effect
        age = -row_app['DAYS_BIRTH'] / 365.25
        if age > 45:
            risk_score -= 0.1
        elif age < 28:
            risk_score += 0.15
            
        # Bound risk score between 0.05 and 0.95
        risk_score = np.clip(risk_score, 0.05, 0.95)
        
        # Generate credit history months (e.g. 1 to 60 months)
        num_months = np.random.randint(5, 61)
        months = np.sort(-np.random.choice(np.arange(61), size=num_months, replace=False))
        
        for m in months:
            # For each month, determine status
            # 'C' - paid off
            # 'X' - no loan
            # '0' - 1-29 days past due
            # '1' - 30-59 days past due
            # '2' - 60-89 days past due
            # '3' - 90-119 days past due
            # '4' - 120-149 days past due
            # '5' - Overdue/bad debt
            
            rand_val = np.random.rand()
            
            if rand_val < 0.45:
                status = 'C'
            elif rand_val < 0.75:
                status = '0'
            elif rand_val < 0.92:
                status = 'X'
            else:
                # Late payment
                # Probability of worse payment depends on risk_score
                late_rand = np.random.rand() * risk_score
                if late_rand < 0.50:
                    status = '0' # Minor delay
                elif late_rand < 0.75:
                    status = '1'
                elif late_rand < 0.88:
                    status = '2'
                elif late_rand < 0.95:
                    status = '3'
                elif late_rand < 0.98:
                    status = '4'
                else:
                    status = '5'
                    
            credit_rows.append({
                'ID': uid,
                'MONTHS_BALANCE': m,
                'STATUS': status
            })
            
    df_credit = pd.DataFrame(credit_rows)
    
    return df_app, df_credit

def save_data(df_app, df_credit, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df_app.to_csv(os.path.join(output_dir, 'application_record.csv'), index=False)
    df_credit.to_csv(os.path.join(output_dir, 'credit_record.csv'), index=False)
    print(f"Datasets generated and saved in: {output_dir}")
    print(f"application_record.csv shape: {df_app.shape}")
    print(f"credit_record.csv shape: {df_credit.shape}")

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(project_root, 'dataset')
    app_df, cred_df = generate_synthetic_data(num_records=6000)
    save_data(app_df, cred_df, dataset_dir)
