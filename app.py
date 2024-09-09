import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# Load data
data = pd.read_excel(r"C:\Users\Admin\Downloads\Base data.xlsx")

# Define categorical features (including Product Display Page Visit)
categorical_cols = ['Last mobile purchased', 'Last Visual Display purchased', 
                   'Last Refrigrator purchased', 'Most used Application on phone',
                   'Product Display Page Visit', 'Abandoned cart','Hobbies',
                   'SmartThing App Installed']


mobile_options = ["S22", "Fold6"]
visual_display_options = ["Q55", "Neo QLED85", "UHD43"]
refrigerator_options = ["Bespoke", "French door", "Single door", "Double door"]
app_options = ["Healthify", "Goibibo"]
pdp_visit_options = ["TV", "Watch7"]
abandoned_cart_options = ["Yes", "No"]
hobbies_options = ["Hiking", "Gym", "Productivity"]
smart_thing_options = ["Yes", "No"]

# Create and fit LabelEncoders (one per feature)
le_mobile = LabelEncoder()
le_visual_display = LabelEncoder()
le_refrigerator = LabelEncoder()
le_app = LabelEncoder()
le_pdp_visit = LabelEncoder()  # Encode Product Display Page Visit as well
le_abandoned_cart = LabelEncoder()
le_hobbies = LabelEncoder()
le_smart_thing = LabelEncoder()

# Encode all categorical features (including Product Display Page Visit)
for col in categorical_cols:
  data[col] = le_mobile.fit_transform(data[col])

# Separate features and target
X = data.drop(['Customer Name','MX'], axis=1)
y = data[['MX']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
y_train = y_train.values.flatten()  # Flattens the DataFrame into a 1D array

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Streamlit App

st.title("Propensity Predictor")

# Feature selection with options
last_mobile_purchased = st.selectbox("Last mobile purchased", options=mobile_options)
last_visual_display_purchased = st.selectbox("Last Visual Display purchased", options=visual_display_options)
last_refrigerator_purchased = st.selectbox("Last Refrigrator purchased", options=refrigerator_options)
most_used_application_on_phone = st.selectbox("Most used Application on phone", options=app_options)
product_display_page_visit = st.selectbox("Product Display Page Visit", options=pdp_visit_options)  # Use options list for product visit
abandoned_cart = st.selectbox("Abandoned cart", options=abandoned_cart_options)
hobbies = st.selectbox("Hobbies", options=hobbies_options)
smart_thing_app_installed = st.selectbox("SmartThing App Installed", options=smart_thing_options)

# User input dictionary (without transformation for numerical feature)
user_input = {
  "Last mobile purchased": le_mobile.transform([last_mobile_purchased])[0],
  "Last Visual Display purchased": le_visual_display.transform([last_visual_display_purchased])[0],
  "Last Refrigrator purchased": le_refrigerator.transform([last_refrigerator_purchased])[0],
  "Most used Application on phone": le_app.transform([most_used_application_on_phone])[0],
  "Product Display Page Visit": le_pdp_visit.transform([product_display_page_visit])[0],  # Encode product visit
  "Abandoned cart": le_abandoned_cart.transform([abandoned_cart])[0],
  "Hobbies": le_hobbies.transform([hobbies])[0],
  "SmartThing App Installed": le_smart_thing.transform([smart_thing_app_installed])[0]
}

# Encode mobile purchase
if last_mobile_purchased not in le_mobile.classes_:
    user_input["Last mobile purchased"] = -1  # Assign a special value for unseen labels
else:
    user_input["Last mobile purchased"] = le_mobile.transform([last_mobile_purchased])[0]

# Convert user input to DataFrame
df_user_input = pd.DataFrame(user_input, index=[0])

# Make prediction
prediction = model.predict_proba(df_user_input)[0]

# Display predicted probabilities
st.write("Probability of using Product 1:", prediction[0])
