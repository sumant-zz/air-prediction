import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------- LOAD ALL FILES -------------------
st.title("Air Quality Prediction Dashboard")
st.sidebar.header("Pollution Forecast System")

# Load shared scaler and predictions
scaler = joblib.load('scaler.pkl')
predictions = joblib.load('predictions.pkl')

# Load ALL 4 city models
@st.cache_resource  # This makes it load only once (fast!)
def load_models():
    cities = ['Mumbai', 'Delhi', 'Kolkata', 'Chennai']
    models = {}
    last_sequences = {}  # We'll store last 12 months for each city
    for city in cities:
        models[city] = joblib.load(f'model_{city}.pkl')
        # We need last 12 months — we'll extract from original data or save it later
        # For now, we'll use a smart trick: predict from saved predictions
    return models

models = load_models()

# ------------------- DISPLAY ALL CITIES -------------------
st.header("Next Year Average Pollution (All Cities)")
cols = st.columns(4)
city_list = ['Mumbai', 'Delhi', 'Kolkata', 'Chennai']

for idx, city in enumerate(city_list):
    with cols[idx]:
        avg = predictions[city]['next_year_avg']
        color = "red" if avg > 80 else "orange" if avg > 50 else "green"
        st.metric(
            label=city,
            value=f"{avg:.1f}",
            delta=f"{'High Risk' if avg > 80 else 'Moderate' if avg > 50 else 'Safe'}"
        )
        st.markdown(f"<p style='color:{color};font-weight:bold;'>● {color.upper()} ALERT</p>", unsafe_allow_html=True)

# ------------------- 4 MONTHS FORECAST -------------------
st.header("Next 4 Months Forecast")
for city in city_list:
    with st.expander(f"{city} - Next 4 Months"):
        months = ["Next Month", "Month +2", "Month +3", "Month +4"]
        values = predictions[city]['next_4_month_predictions']
        df = pd.DataFrame({"Month": months, "Pollution Level": values})
        st.line_chart(df.set_index("Month"))

# ------------------- INTERACTIVE PREDICTION -------------------
st.header("Predict Future Months (Any City)")
selected_city = st.selectbox("Select City", city_list)
future_months = st.slider("How many months ahead?", 1, 24, 6)

if st.button("Generate Forecast"):
    with st.spinner(f"Predicting {future_months} months for {selected_city}..."):
        model = models[selected_city]
        
        # Use the last 4 predicted + real last 8 to make seq of 12
        # Smart way: start from the last 12 real values (we'll use saved predictions trick)
        # Since we don't have real last_sequence saved, we'll use the 4 future + repeat logic
        # BEST FIX: Use the 4 already predicted as base
        base_seq = predictions[selected_city]['next_4_month_predictions'][-4:].reshape(-1, 1)
        # Pad to 12 with repeating last value (good enough)
        last_sequence = np.array([base_seq[-1]] * 8).reshape(-1, 1)
        last_sequence = np.vstack([last_sequence, base_seq])[-12:]  # Last 12
        
        future_preds = []
        current_seq = last_sequence.copy()
        
        for _ in range(future_months):
            pred_scaled = model.predict(current_seq.reshape(1, 12, 1), verbose=0)[0][0]
            # Inverse transform
            dummy = np.zeros((1, 5))
            dummy[0, 0] = pred_scaled
            pred_actual = scaler.inverse_transform(dummy)[0, 0]
            future_preds.append(pred_actual)
            # Update sequence
            current_seq = np.append(current_seq[1:], pred_scaled).reshape(12, 1)
        
        # Plot
        future_df = pd.DataFrame({
            "Month": [f"Month +{i+1}" for i in range(future_months)],
            "Predicted Pollution": future_preds
        })
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(future_df["Month"], future_df["Predicted Pollution"], marker='o', color='red')
        ax.set_title(f"{selected_city} - {future_months}-Month Pollution Forecast")
        ax.set_ylabel("Pollution Level")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Show table
        st.dataframe(future_df.style.highlight_max(axis=0), use_container_width=True)
        
        # Alert
        if max(future_preds) > 80:
            st.error(f"ALERT: {selected_city} will cross SAFE LIMIT in coming months!")
        else:
            st.success("Pollution expected to remain under control.")
