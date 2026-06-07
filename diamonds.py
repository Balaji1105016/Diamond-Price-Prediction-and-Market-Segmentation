import streamlit as st
import pandas as pd
import joblib

# Loading Files:
price_model = joblib.load("diamond_price_prediction.pkl")
cluster_model = joblib.load("diamond_clustering.pkl")
cluster_scaler = joblib.load("diamond_cluster_scaler.pkl")
encoder = joblib.load("diamond_encoder.pkl")

st.title("Diamond Analytics App")
st.subheader("Diamond Price & Market Segment Prediction")

# Inputs:
carat = st.number_input("Carat",min_value=0.0,value=0.5)
x = st.number_input("Length (x)",min_value=0.0,value=5.0)
y = st.number_input("Width (y)",min_value=0.0,value=5.0)
cut = st.selectbox("Cut",["Fair","Good","Very Good","Premium","Ideal"])
color = st.selectbox("Color",["J","I","H","G","F","E","D"])
clarity = st.selectbox("Clarity",["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"])

# Encoding:
cat_df = pd.DataFrame({"cut":[cut],"color":[color],"clarity":[clarity]})
encoded = encoder.transform(cat_df)
cut_encoded = encoded[0][0]
color_encoded = encoded[0][1]
clarity_encoded = encoded[0][2]

if st.button("Predict Price"):

    price_input = pd.DataFrame([[y,carat,x,clarity_encoded,color_encoded]],columns=["y","carat","x","clarity","color"])
    predicted_price = price_model.predict(price_input)[0]
    st.success(f"Predicted Diamond Price : ₹ {predicted_price:,.2f}")
    
if st.button("Predict Cluster"):

    # Price Prediction First

    price_input = pd.DataFrame([[y,carat,x,clarity_encoded,color_encoded]],columns=["y","carat","x","clarity","color"])
    predicted_price = price_model.predict(price_input)[0]
    # Cluster Input

    cluster_input = pd.DataFrame([[predicted_price,carat,cut_encoded,color_encoded,clarity_encoded]],columns=["price","carat","cut","color","clarity"])
    cluster_input_scaled = cluster_scaler.transform(cluster_input)

    cluster = cluster_model.predict(cluster_input_scaled)[0]

    cluster_names = {
        0: "Premium Heavy Diamonds",
        1: "Affordable Small Diamonds",
        2: "Mid-range Balanced Diamonds"}

    st.success(f"Cluster Number : {cluster}")

    st.info(f"Cluster Name : {cluster_names.get(cluster,'Unknown')}")
