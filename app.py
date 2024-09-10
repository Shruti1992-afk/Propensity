import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# Function to load data from uploaded file
def load_data(uploaded_file):
  if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)  # Use pandas.read_csv for CSV files
    return data
  else:
    st.warning("Please upload a CSV file.")
    return None

# Function to encode categorical features
def encode_features(data, categorical_cols):
  for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
  return data

mobile_options = ["S22", "Fold6"]
visual_display_options = ["Q55", "Neo QLED85", "UHD43"]
refrigerator_options = ["Bespoke", "French door", "Single door", "Double door"]
app_options = ["Healthify", "Goibibo"]
pdp_visit_options = ["TV", "Watch7"]
abandoned_cart_options = ["Yes", "No"]
hobbies_options = ["Hiking", "Gym", "Productivity"]
smart_thing_options = ["Yes", "No"]

# Main app logic
def main():
  # Upload data
  uploaded_file = st.file_uploader("Choose a CSV file")
  data = load_data(uploaded_file)

  if data is not None:
    # Define categorical features
    categorical_cols = ['Last mobile purchased', 'Last Visual Display purchased',
                        'Last Refrigrator purchased', 'Most used Application on phone',
                        'Product Display Page Visit', 'Abandoned cart','Hobbies',
                        'SmartThing App Installed']

    # Encode features
    data = encode_features(data.copy(), categorical_cols)

    # Separate features and target
    X = data.drop(['Customer Name','MX'], axis=1)
    y = data[['MX']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    y_train = y_train.values.flatten()  # Flatten the DataFrame into a 1D array

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Streamlit App Interface
    st.title("Propensity Predictor")

    # Feature selection with options
    last_mobile_purchased = st.selectbox("Last mobile purchased", options=mobile_options)
    last_visual_display_purchased = st.selectbox("Last Visual Display purchased", options=visual_display_options)
    last_refrigerator_purchased = st.selectbox("Last Refrigrator purchased", options=refrigerator_options)
    most_used_application_on_phone = st.selectbox("Most used Application on phone", options=app_options)
    product_display_page_visit = st.selectbox("Product Display Page Visit", options=pdp_visit_options)
    abandoned_cart = st.selectbox("Abandoned cart", options=abandoned_cart_options)
    hobbies = st.selectbox("Hobbies", options=hobbies_options)
    smart_thing_app_installed = st.selectbox("SmartThing App Installed", options=smart_thing_options)

    # User input dictionary
    user_input = {
        "Last mobile purchased": last_mobile_purchased,
        "Last Visual Display purchased": last_visual_display_purchased,
        "Last Refrigrator purchased": last_refrigerator_purchased,
        "Most used Application on phone": most_used_application_on_phone,
        "Product Display Page Visit": product_display_page_visit,
        "Abandoned cart": abandoned_cart,
        "Hobbies": hobbies,
        "SmartThing App Installed": smart_thing_app_installed
    }

    # Encode user input (if needed)
    encoded_user_input = user_input.copy()
    for col in categorical_cols:
      le = LabelEncoder()  # Create a new encoder for prediction (avoids modifying original encoders)
      encoded_user_input[col] = le.fit_transform([encoded_user_input[col]])[0]

    # Convert user input to DataFrame
    df_user_input = pd.DataFrame(encoded_user_input, index=[0])

    # Make prediction
    prediction = model.predict_proba(df_user_input)[0]

    # Display predicted probabilities
    st.write("Probability of using Product 1:", prediction[0])

if __name__ == "__main__":
  main()
