import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model, scaler, and encoder
model = pickle.load(open('motor_churn_model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Define the input fields for the Streamlit app
def user_input_features():
    region = st.selectbox('Region', ['South East England', 'London', 'East Midlands', 'West Midlands', 'North East England'], key='region_input')
    type_of_plan = st.selectbox('Type of Plan', ['Comprehensive', 'Third Party, Fire and Theft', 'Third Party Only'], key='type_of_plan_input')
    highest_level_education = st.selectbox('Highest Level of Education', ['Bachelor', 'PhD', 'Master', 'High School', 'Diploma'], key='highest_level_education_input')
    work_status = st.selectbox('Work Status', ['Employed', 'Unemployed', 'Retired', 'Self-employed'], key='work_status_input')
    sex = st.selectbox('Sex', ['male', 'female'], key='sex_input')
    relationship_status = st.selectbox('Relationship Status', ['Single', 'Married', 'Divorced', 'Widowed'], key='relationship_status_input')
    weeks_since_claim = st.slider('Weeks Since Last Claim', 0, 100, 30, key='weeks_since_claim_input')
    open_policies = st.slider('Number of Open Policies', 1, 9, 3, key='open_policies_input')
    renew_offer_type = st.slider('Renew Offer Type', 1, 3, 2, key='renew_offer_type_input')
    reachability = st.selectbox('Preferred Method of Contact', ['Online Portal', 'Phone', 'Email', 'Agent'], key='reachability_input')
    type_of_vehicle = st.selectbox('Type of Vehicle', ['Hatchback', 'Sedan', 'Truck', 'SUV', 'Convertible'], key='type_of_vehicle_input')
    age = st.slider('Age', 18, 80, 35, key='age_input')
    insurance_premium = st.slider('Insurance Premium', 0, 2000, 500, key='insurance_premium_input')
    annual_income = st.slider('Annual Income', 20000, 79000, 50000, key='annual_income_input')
    monthly_income = annual_income / 12
    
    data = {
        'region': region,
        'type_of_plan': type_of_plan,
        'highest_level_education': highest_level_education,
        'work_status': work_status,
        'sex': sex,
        'relationship_status': relationship_status,
        'weeks_since_claim': weeks_since_claim,
        'open_policies': open_policies,
        'Renew_Offer_Type': renew_offer_type,
        'reachability': reachability,
        'type_of_vehicle': type_of_vehicle,
        'age': age,
        'Insurance Premium': insurance_premium,
        'annual_income': annual_income,
        'monthly_income': monthly_income
    }
    features = pd.DataFrame(data, index=[0])
    return features

def preprocess_features(df):
    # Apply feature engineering
    # Convert categorical variables to categorical type
    categorical_cols = ['region', 'type_of_plan', 'highest_level_education', 'work_status', 'sex', 'relationship_status', 'reachability', 'type_of_vehicle']
    df[categorical_cols] = df[categorical_cols].astype('category')

    # Binning age into categories
    age_bins = [18, 30, 40, 50, 60, 80]
    age_labels = ['18-29', '30-39', '40-49', '50-59', '60-80']
    df['age_binned'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)

    # Binning annual income into categories
    income_bins = [0, 20000, 40000, 60000, 80000, 100000]
    income_labels = ['0-19k', '20k-39k', '40k-59k', '60k-79k', '80k-100k']
    df['annual_income_binned'] = pd.cut(df['annual_income'], bins=income_bins, labels=income_labels, right=False)
    df['premium_to_income_ratio'] = df['Insurance Premium'] / df['annual_income']
    
    # Drop original columns used for binning
    df = df.drop(['age', 'annual_income', 'Insurance Premium'], axis=1)
    
    cat1 = []
    for i in df.columns:
        if df[i].dtype == 'O' or df[i].dtype == 'category':
            cat1.append(i)
    
    # Encode categorical features
    encoded_cats = encoder.transform(df[cat1]).toarray()
    enc_data = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(cat1))
    df = df.join(enc_data)

    df.drop(cat1, axis=1, inplace=True)
    
    col = df.columns
    # Scale numerical features
    df = scaler.transform(df)
    df = pd.DataFrame(df, columns=col)
    
    return df

# Main function to run the Streamlit app
def main():
    st.title('Customer Churn Prediction')
    st.write('Enter customer details to predict churn probability')

    # User input features
    input_df = user_input_features()

    # Display the input features
    st.write('Customer Details:')
    st.write(input_df)

    # Preprocess the input features
    preprocessed_df = preprocess_features(input_df)

    # Ensure the number of features matches
    expected_num_features = scaler.n_features_in_
    if preprocessed_df.shape[1] != expected_num_features:
        st.error(f'Expected {expected_num_features} features, but got {preprocessed_df.shape[1]} features. Please check the preprocessing steps.')

    # Make prediction if the number of features matches
    if preprocessed_df.shape[1] == expected_num_features:
        prediction_proba = model.predict_proba(preprocessed_df)
        custom_threshold = 0.4  # Adjust the threshold here
        prediction = (prediction_proba[:, 1] >= custom_threshold).astype(int)

        # Display the prediction
        st.write('Churn Prediction:')
        churn_label = 'Yes' if prediction[0] == 1 else 'No'
        st.write(churn_label)

        st.write('Prediction Probability:')
        st.write(f'Probability of Churning: {prediction_proba[0][1]:.2f}')
        st.write(f'Probability of Not Churning: {prediction_proba[0][0]:.2f}')
    else:
        st.error('Feature mismatch. Please verify the input features.')

if __name__ == '__main__':
    main()
