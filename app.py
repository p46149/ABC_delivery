import streamlit as st
import joblib
import pandas as pd

# Load the trained model
logi = joblib.load("delivery_delay.xlsx")

st.title("Delivery Delay Prediction App")
st.write("Enter the features below to predict if there will be a delivery delay.")

# Input fields for each feature
delivery_distance = st.number_input("Delivery Distance", value=12.0)
traffic_congestion = st.number_input("Traffic Congestion (1-5)", min_value=1, max_value=5, value=3)
weather_condition = st.number_input("Weather Condition (1-3)", min_value=1, max_value=3, value=1)
delivery_slot = st.number_input("Delivery Slot (1-3)", min_value=1, max_value=3, value=2)
driver_experience = st.number_input("Driver Experience (Years)", value=4.0)
num_stops = st.number_input("Number of Stops", value=2)
vehicle_age = st.number_input("Vehicle Age (Years)", value=8.0)
road_condition_score = st.number_input("Road Condition Score (1-10)", min_value=1, max_value=10, value=2)
package_weight = st.number_input("Package Weight (kg)", value=15.0)
fuel_efficiency = st.number_input("Fuel Efficiency (km/l)", value=12.0)
warehouse_processing_time = st.number_input("Warehouse Processing Time (minutes)", value=80.0)


if st.button("Predict Delivery Delay"):
    # Create a DataFrame from the inputs
    input_data = pd.DataFrame([[
        delivery_distance,
        traffic_congestion,
        weather_condition,
        delivery_slot,
        driver_experience,
        num_stops,
        vehicle_age,
        road_condition_score,
        package_weight,
        fuel_efficiency,
        warehouse_processing_time
    ]], columns=[
        'Delivery_Distance',
        'Traffic_Congestion',
        'Weather_Condition',
        'Delivery_Slot',
        'Driver_Experience',
        'Num_Stops',
        'Vehicle_Age',
        'Road_Condition_Score',
        'Package_Weight',
        'Fuel_Efficiency',
        'Warehouse_Processing_Time'
    ])

    prediction = logi.predict(input_data)[0]

    if prediction == 1:
        st.error("Prediction: Delivery is likely to be delayed.")
    else:
        st.success("Prediction: Delivery is likely to be on time.")
"""
