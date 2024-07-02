import streamlit as st
import pickle
import numpy as np

# Charger le modèle et les données nécessaires
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Titre de l'application
st.title("Laptop Price Predictor")

# Interface utilisateur pour saisir les caractéristiques d'un ordinateur portable
company = st.selectbox('Brand', df['Company'].unique())
type = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input('Weight of the Laptop')
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS', ['No', 'Yes'])
screen_size = st.number_input('Screen Size')
resolution = st.selectbox('Screen Resolution', [
    '1920x1080', '1366x768', '1600x900', '3840x2160',
    '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
cpu = st.selectbox('CPU', df['Cpu brand'].unique())
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])
gpu = st.selectbox('GPU', df['Gpu brand'].unique())
os = st.selectbox('OS', df['os'].unique())

# Prévoir le prix quand le bouton est cliqué
if st.button('Predict Price'):
    # Prétraitement des données d'entrée
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    # Calcul du PPI
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

    # Création du vecteur de requête avec exactement 13 caractéristiques
    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

    # Vérification du nombre de caractéristiques
    if len(query) != 13:
        st.error(f"Expected 13 features, but got {len(query)} features.")
    else:
        query = query.reshape(1, -1)

        # Prédiction du prix
        try:
            predicted_price = pipe.predict(query)
            st.title("The predicted price of this configuration is $" + str(int(np.exp(predicted_price[0]))))
        except ValueError as e:
            st.error(f"Error in prediction: {e}")
