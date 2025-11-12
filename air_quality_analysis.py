import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import calendar
from sklearn.preprocessing import MinMaxScaler
from twilio.rest import Client

# Load the dataset
df = pd.read_csv('air_quality_data1.csv')

# Clean and preprocess the data
df.dropna(inplace=True)
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.day_name()
df['hour'] = df['date'].dt.hour

# Normalize the data for RNN input
scaler = MinMaxScaler()
df[['pollution_level', 'CO', 'SO2', 'NOx', 'O3']] = scaler.fit_transform(
    df[['pollution_level', 'CO', 'SO2', 'NOx', 'O3']]
)

# Function to convert hour to AM/PM format
def convert_to_ampm(hour):
    if hour == 0:
        return '12 AM'
    elif 1 <= hour <= 11:
        return f'{hour} AM'
    elif hour == 12:
        return '12 PM'
    else:
        return f'{hour - 12} PM'

# Function to plot trends for selected cities
def plot_city_trends(df, cities):
    sns.set(style="whitegrid")
    
    for city in cities:
        city_df = df[df['city'] == city]
        
        # Calculate trends
        yearly_trends = city_df.groupby('year')[['pollution_level', 'CO', 'SO2', 'NOx', 'O3']].mean()
        monthly_trends = city_df.groupby('month')[['pollution_level', 'CO', 'SO2', 'NOx', 'O3']].mean()
        monthly_trends.index = monthly_trends.index.map(lambda x: calendar.month_name[x])
        weekly_trends = city_df.groupby('day_of_week')[['pollution_level', 'CO', 'SO2', 'NOx', 'O3']].mean()
        hourly_trends = city_df.groupby(city_df['hour'].apply(convert_to_ampm))[['pollution_level', 'CO', 'SO2', 'NOx', 'O3']].mean()

        # Plotting
        plt.figure(figsize=(16, 10))

        # Plot yearly trends
        plt.subplot(231)
        sns.lineplot(data=yearly_trends, markers=True)
        plt.title(f'Yearly Air Pollution Trends - {city}')
        plt.xlabel('Year')
        plt.ylabel('Average Levels')
        plt.legend(yearly_trends.columns, title='Parameter')

        # Plot monthly trends
        plt.subplot(232)
        sns.lineplot(data=monthly_trends, markers=True)
        plt.title(f'Monthly Air Pollution Trends - {city}')
        plt.xlabel('Month')
        plt.ylabel('Average Levels')
        plt.legend(monthly_trends.columns, title='Parameter')

        # Plot weekly trends
        plt.subplot(233)
        weekly_trends = weekly_trends.reset_index().melt(id_vars='day_of_week', var_name='Parameter', value_name='Level')
        sns.barplot(data=weekly_trends, x='day_of_week', y='Level', hue='Parameter')
        plt.title(f'Weekly Air Pollution Trends - {city}')
        plt.xlabel('Day of the Week')
        plt.ylabel('Average Levels')
        plt.legend(title='Parameter')

        # Plot hourly trends
        plt.subplot(234)
        sns.lineplot(data=hourly_trends, markers=True)
        plt.title(f'Hourly Air Pollution Trends - {city}')
        plt.xlabel('Hour of the Day (AM/PM)')
        plt.ylabel('Average Levels')
        plt.legend(hourly_trends.columns, title='Parameter')

        plt.tight_layout()
        plt.show()

# Function to send an alert message to the city corporation number
def send_alert(city, pollution_level):
    # Twilio credentials (replace with your own)
    account_sid = 'AC2fb599b569a843805c64b**'
    auth_token = 'a1133d18d5ca11c18**'
    client = Client(account_sid, auth_token)

    message = f"Alert! The predicted average pollution level in {city} for next year is {pollution_level:.2f}, exceeding safe limits."

    try:
        client.messages.create(
            body=message,
            from_='+121850155**',
            to='+9170906226**'
        )
        print(f"Alert sent to {city} corporation.")
    except Exception as e:
        print(f"Failed to send alert to {city} corporation: {e}")

# Function to create sequences for RNN
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length])
    return np.array(sequences), np.array(labels)

# Function to predict future pollution levels for each city using RNN
def predict_future_pollution(df, cities, seq_length=12):
    predictions = {}
    models = {}  # To store trained models per city
    current_month = df['date'].dt.month.max()
    for city in cities:
        city_df = df[df['city'] == city]
        # Prepare data for RNN
        data = city_df[['pollution_level']].values
        sequences, labels = create_sequences(data, seq_length)
        # Split into training and test sets
        split_idx = int(0.8 * len(sequences))
        X_train, X_test = sequences[:split_idx], sequences[split_idx:]
        y_train, y_test = labels[:split_idx], labels[split_idx:]
        # Define the RNN model
        model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, 1)),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        # Train the model
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1, validation_data=(X_test, y_test))
        # Predict next year's average pollution level
        yearly_prediction = model.predict(X_test[-1].reshape(1, seq_length, 1))[0][0]
       
        # Add dummy values for the other features to match scaler input shape
        dummy_input = np.zeros((1, 5))  # Create a dummy input with 5 features
        dummy_input[0, 0] = yearly_prediction  # Set the first feature (pollution_level)
        # Inverse transform to get the actual pollution level
        inverse_transformed = scaler.inverse_transform(dummy_input)
        yearly_prediction = inverse_transformed[0, 0]  # Extract only the pollution_level
        # Predict future values
        future_predictions = []
        last_sequence = data[-seq_length:]
        for _ in range(4):  # Predicting the next 4 months
            prediction = model.predict(last_sequence.reshape(1, seq_length, 1))[0][0]
            future_predictions.append(prediction)
            last_sequence = np.append(last_sequence[1:], prediction).reshape(seq_length, 1)
        future_predictions = scaler.inverse_transform(
            np.hstack([np.array(future_predictions).reshape(-1, 1), np.zeros((4, 4))])
        )[:, 0]
        predictions[city] = {
            'next_year_avg': yearly_prediction,
            'next_4_month_predictions': future_predictions,
        }
        models[city] = model  # Save the trained model
        # Check if the next year's average pollution level exceeds 80 and send an alert
        if yearly_prediction > 80:
            send_alert(city, yearly_prediction)
        # Print predictions
        print(f"Predicted average pollution level in {city} for next year: {yearly_prediction:.2f}")
        print(f"Predicted pollution levels in {city} for the next 4 months: {future_predictions}")
    
    # After all cities, save everything to .pkl files
    joblib.dump(predictions, 'predictions.pkl')  # Save predictions dict
    joblib.dump(scaler, 'scaler.pkl')  # Save the scaler
    for city, model in models.items():
        joblib.dump(model, f'model_{city}.pkl')  # Save each model as separate .pkl
    print("Saved predictions.pkl, scaler.pkl, and model_*.pkl files!")
    
    return predictions

# Example usage
cities = ['Mumbai', 'Delhi', 'Kolkata', 'Chennai']
plot_city_trends(df, cities)
predictions = predict_future_pollution(df, cities)