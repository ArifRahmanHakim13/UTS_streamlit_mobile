import pickle
import streamlit as st

model = pickle.load(open('estimasi_mobile.sav', 'rb'))

st.title('Klasifikasikan Kisaran Harga Ponsel')

battery_power = st.number_input('Input battery_power')
clock_speed = st.number_input('Input clock_speed')
mobile_wt = st.number_input('Input mobile_wt')
ram = st.number_input('Input ram')
int_memory = st.number_input('Input int_memory')
dual_sim = st.number_input('Input dual_sim')
touch_screen = st.number_input('Input touch_screen')
n_cores = st.number_input('Input n_cores')

if st.button('Mobile Price'):
    input_data = [[battery_power, clock_speed, mobile_wt, ram, int_memory, dual_sim, touch_screen, n_cores]]
    predicted_price_range = model.predict(input_data)[0]  

   
    price_category = ''
    if predicted_price_range == 0:
        price_category = "Murah"
    elif predicted_price_range == 1:
        price_category = "Menengah Bawah"
    elif predicted_price_range == 2:
        price_category = "Menengah"
    elif predicted_price_range == 3:
        price_category = "Mahal"
    
    st.write(f'Price Range: {int(predicted_price_range)}')
    #Price range di angka 0 mengkategorikan harga Hp tersebut "Murah"
    #Price range di angka 1 mengkatagorikan harga Hp tersebut "Menengah bawah/lumayan murah"
    #Price range di angka 2 mengkatagorikan harga Hp tersebut "Menengah/sedang"
    #Price range di angka 3 mengkatagorikan harga Hp tersebut "Mahal"