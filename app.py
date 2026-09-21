import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('delivery_delay.sav')

st.title('Delivery Delay Prediction')
st.write('Enter the features to predict if there will be a delivery delay.')

# Define the expected columns in the order the model was trained
expected_columns = [
    'Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
    'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
    'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
    'Warehouse_Processing_Time'
]

# Create input widgets for each feature
delivery_distance = st.number_input('Delivery Distance (km)', min_value=0.0, max_value=100.0, value=20.0, step=0.1)
traffic_congestion = st.selectbox('Traffic Congestion (1-5)', [1, 2, 3, 4, 5], index=2) # Assuming 1-5 scale
weather_condition = st.selectbox('Weather Condition (1-5)', [1, 2, 3, 4, 5], index=2) # Assuming 1-5 scale
delivery_slot = st.selectbox('Delivery Slot (1-3)', [1, 2, 3], index=1) # Assuming 1-3 categories
driver_experience = st.number_input('Driver Experience (years)', min_value=0, max_value=30, value=10, step=1)
num_stops = st.number_input('Number of Stops', min_value=0, max_value=20, value=5, step=1)
vehicle_age = st.number_input('Vehicle Age (years)', min_value=0, max_value=15, value=5, step=1)
road_condition_score = st.selectbox('Road Condition Score (1-5)', [1, 2, 3, 4, 5], index=2) # Assuming 1-5 scale
package_weight = st.number_input('Package Weight (kg)', min_value=0.0, max_value=50.0, value=10.0, step=0.1)
fuel_efficiency = st.number_input('Fuel Efficiency (km/L)', min_value=0.0, max_value=30.0, value=15.0, step=0.1)
warehouse_processing_time = st.number_input('Warehouse Processing Time (hours)', min_value=0, max_value=100, value=48, step=1)

# When the 'Predict' button is clicked
if st.button('Predict Delivery Delay'):
    # Create a DataFrame from the input values
    input_data = pd.DataFrame([[delivery_distance,
                                  traffic_congestion,
                                  weather_condition,
                                  delivery_slot,
                                  driver_experience,
                                  num_stops,
                                  vehicle_age,
                                  road_condition_score,
                                  package_weight,
                                  fuel_efficiency,
                                  warehouse_processing_time]], columns=expected_columns)

    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error(f'The model predicts a **Delivery Delay** with a probability of {prediction_proba[0][1]*100:.2f}%')
    else:
        st.success(f'The model predicts **No Delivery Delay** with a probability of {prediction_proba[0][0]*100:.2f}%')
