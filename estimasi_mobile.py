import pickle
import streamlit as st

model = pickle.load(open('estimasi_mobile.sav', 'rb'))

st.title('Klasifikasikan Kisaran Harga Ponsel')

battery_power = st.number_input('Input battery_power')
clock_speed = st.number_input('Input clock_speed')
mobile_wt = st.number_input('Input mobile_wt')
ram = st.number_input('Input ram')
int_memory = st.number_input('Input int_memory')
dual_sim = st.selectbox('Input Dual SIM',(0, 1))
touch_screen = st.selectbox('Input Touchscreen',(0, 1))
n_cores = st.number_input('Input n_cores')

if st.button('Mobile Price'):
    input_data = [[battery_power, clock_speed, mobile_wt, ram, int_memory, dual_sim, touch_screen, n_cores]]
    predicted_price_range = model.predict(input_data)[0]  

    predict_result = int(predicted_price_range)
    price_category = ''

    if predict_result == 0:
        price_category = "Murah"
    elif predict_result == 1:
        price_category = "Menengah Bawah"
    elif predict_result == 2:
        price_category = "Menengah"
    elif predict_result == 3:
        price_category = "Mahal"
    
    print(type(predicted_price_range))
    print(predicted_price_range)
    st.write(f'Price Range: {price_category} dengan nilai {predict_result}')
    #Price range di angka 0 mengkategorikan harga Hp tersebut "Murah"
    #Price range di angka 1 mengkatagorikan harga Hp tersebut "Menengah bawah/lumayan murah"
    #Price range di angka 2 mengkatagorikan harga Hp tersebut "Menengah/sedang"
    #Price range di angka 3 mengkatagorikan harga Hp tersebut "Mahal"