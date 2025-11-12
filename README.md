# air-prediction
# Air Quality Prediction Dashboard  
**Real-time Pollution Forecasting using LSTM-RNN + Interactive Web App**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)  
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)  
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)  
![License](https://img.shields.io/badge/License-MIT-green)

**Live Demo**: [https://your-username-air-quality.streamlit.app](https://your-username-air-quality.streamlit.app) *(deploy after GitHub)*

---

### Project Overview
This is an **end-to-end AI-powered Air Quality Monitoring & Prediction System** that:
- Analyzes historical air pollution data (`CO`, `SO2`, `NOx`, `O3`, AQI)
- Uses **LSTM Deep Learning** (Recurrent Neural Network) to predict future pollution
- Sends **SMS alerts** via Twilio when pollution crosses safe limits
- Visualizes trends (Yearly, Monthly, Weekly, Hourly) for **Mumbai, Delhi, Kolkata, Chennai**
- Deploys a **beautiful interactive web dashboard** using **Streamlit**

---

### Features
| Feature | Description |
|-------|-----------|
| **Multi-City Forecasting** | Predicts for 4 major Indian cities |
| **Next-Year Average Prediction** | Forecasts annual average pollution |
| **Next 4–24 Months Forecast** | Interactive slider for custom future predictions |
| **Color-Coded Risk Alerts** | Green / Orange / Red based on pollution level |
| **Auto SMS Alert System** | Sends Twilio SMS if pollution > 80 |
| **Trend Analysis Plots** | Yearly, Monthly, Weekly, Hourly visualizations |
| **No Retraining Needed** | Uses saved `.pkl` models (fast loading) |
| **100% Local & Deployable** | Works offline + free online deployment |

---

### Project Structure
