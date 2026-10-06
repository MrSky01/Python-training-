import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Crop Recommendation", layout="centered")

saved = joblib.load("model.pkl")
model = saved["model"]

st.title("Crop Recommendation")
st.caption("Enter soil and weather values to find the most suitable crop.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    n = st.number_input("Nitrogen (N)", 0, 200, 50)
    p = st.number_input("Phosphorus (P)", 0, 200, 40)
    k = st.number_input("Potassium (K)", 0, 250, 40)
    temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)

with col2:
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 70.0)
    ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 400.0, 100.0)

if st.button("Predict", type="primary"):
    values = pd.DataFrame(
        [[n, p, k, temperature, humidity, ph, rainfall]],
        columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
    )

    probs = model.predict_proba(values)[0]
    ranked = sorted(zip(model.classes_, probs), key=lambda x: x[1], reverse=True)[:3]

    st.divider()
    st.caption("Recommended crop")
    st.header(ranked[0][0].title())
    st.caption(f"{ranked[0][1]*100:.0f}% confidence")

    st.write("Other options")
    for name, prob in ranked[1:]:
        st.progress(float(prob), text=f"{name.title()}  {prob*100:.0f}%")

with st.expander("About the model"):
    st.write(f"Best model: {saved['name']} (test accuracy {saved['accuracy']}%)")
    st.dataframe(saved["results"], hide_index=True)
    st.image("charts/feature_importance.png")
    st.image("charts/confusion_matrix.png")
    st.write("Only the 22 crops in the dataset can be predicted. For academic use only.")
