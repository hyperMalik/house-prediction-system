import streamlit as st
import pickle
import pandas as pd
import numpy as np



# model = pickle.load(open('model.sav', 'rb'))       # rb stands for ---- Read Binary Format




st.title("HOUSE PREDICTION SYSTEM")
st.sidebar.header('House data')
st.subheader("make your life easy")
# image = Image.open('download222.jpeg')
# st.image(image ,'')


st.write(""" :house:
""")


# def user_report():
#     deposit = st.sidebar.slider('Deposit', 1, 50000, 1)
#     rent = st.sidebar.slider('Rent', 1,50000,1)
#     floor =st.sidebar.slider('Floor', 1,22,1)
#     area =st.sidebar.slider('Area (sq_feet)',1, 582500,1 )
#     room =st.sidebar.slider('Room', 1,4,1)
#     elevator =st.sidebar.checkbox('Elevator', 0,1,0)
#     parking =st.sidebar.checkbox('Parking', 0,1,0)
#     warehouse =st.sidebar.checkbox('Warehouse', 0,1,0)
#     district=st.sidebar.checkbox('District', 0,23,0)
#
#
#
#
#
#
# user_report()


#
# def main():
#     st.title('House Price Prediction')
#
#     deposit = st.number_input('Deposit amount:', value=0, max_value =50001, min_value =0)
#     rent = st.number_input('Rent amount:', max_value =50001, min_value =0, value=0)
#     floor = st.number_input('Floor number:', value=0, max_value =30, min_value =0)
#     area = st.number_input('Area in square meters:', value=0, max_value =582501, min_value = 0)
#     rooms = st.number_input('Number of rooms:', value=1, max_value =4, min_value= 1)
#     elevator = st.number_input('Elevator (0 for No, 1 for Yes):', min_value=0, max_value=1, value=0)
#     parking = st.number_input('Parking (0 for No, 1 for Yes):', min_value=0, max_value=1, value=0)
#     warehouse = st.number_input('Warehouse (0 for No, 1 for Yes):', min_value=0, max_value=1, value=0)
#     district = st.number_input('District', max_value =22, min_value = 1, value=1)





# def predict_price(input_data):
#     # Placeholder function for prediction
#     # Replace this with your actual prediction function
#     # Make sure to load your trained model and preprocess the input_data
#     # Then, use the model to predict the house price
#      predicted_price = 0 # Dummy prediction
#      return predicted_price
#
# def kami():
#     st.title('House Price Prediction')
#     deposit = st.number_input('Deposit amount:', value=0, max_value=50001, min_value=0)
#     rent = st.number_input('Rent amount:', max_value=50001, min_value=0, value=0)
#     floor = st.number_input('Floor number:', value=0, max_value=30, min_value=0)
#     area = st.number_input('Area in square meters:', value=0, max_value=582501, min_value=0)
#     rooms = st.number_input('Number of rooms:', value=1, max_value=4, min_value=1)
#     elevator = st.number_input('Elevator (0 for No, 1 for Yes):', min_value=0, max_value=1, value=0)
#     parking = st.number_input('Parking (0 for No, 1 for Yes):', min_value=0, max_value=1, value=0)
#     warehouse = st.number_input('Warehouse (0 for No, 1 for Yes):', min_value=0, max_value=1, value=0)
#     district = st.number_input('District', max_value=22, min_value=1, value=1)
#
#     if st.button('Predict'):
#         input_data = {
#             'deposit': deposit,
#             'rent': rent,
#             'floor': floor,
#             'area': area,
#             'rooms': rooms,
#             'elevator': elevator,
#             'parking': parking,
#             'warehouse': warehouse,
#             'district': district
#         }
#         predicted_price = predict_price(input_data)
#         st.success(f'Predicted price: ${model.predict(kami()):,.2f}')
#
# if __name__ == '__main__':
#     kami()






model = pickle.load(open('model.sav', 'rb'))       # rb stands for ---- Read Binary Format


def load_model():
    return model  # Replace 'your_model_file.joblib' with the path to your trained model file

# Preprocess input data
def preprocess_input(input_data):
    # Implement any necessary preprocessing steps here
    # This may include data cleaning, feature engineering, encoding categorical variables, etc.
    return input_data  # For demonstration purposes, we'll return the input data as is

# Function to make predictions using the loaded model
def predict_price(model, input_data):
    input_data = preprocess_input(input_data)
    predicted_price = model.predict(pd.DataFrame(input_data, index=[0]))
    return predicted_price[0]

def main():
    # st.markdown(
    #     f"""
    #        <style>
    #            .reportview-container {{
    #                background: url('download333.jpeg');
    #                background-size: cover;
    #            }}
    #        </style>
    #        """,
    #     unsafe_allow_html=True
    # )


    st.title('House Price Prediction')

    # Load the pre-trained model
    model = load_model()

    deposit = st.number_input('Deposit amount:', value=0,  max_value=50000, min_value=0)
    rent = st.number_input('Rent amount:', value=0, max_value=50000, min_value=0)
    floor = st.number_input('Floor number:', value=0,  max_value=30, min_value=0)
    area = st.number_input('Area in square meters:', value=0,  max_value=582500, min_value=0)
    rooms = st.number_input('Number of rooms:', value=0, max_value=4, min_value=0)
    elevator = st.selectbox('Elevator (0 for No, 1 for Yes):', [0, 1])
    parking = st.selectbox('Parking (0 for No, 1 for Yes):', [0, 1])
    warehouse = st.selectbox('Warehouse (0 for No, 1 for Yes):', [0, 1])
    district = st.number_input('District',value=0, max_value=22, min_value=0)

    input_data = {
        'deposit': deposit,
        'rent': rent,
        'floor': floor,
        'area': area,
        'rooms': rooms,
        'elevator': elevator,
        'parking': parking,
        'warehouse': warehouse,
        'district': district
    }

    if st.button('Predict'):
        predicted_price = predict_price(model, input_data)
        st.success(f'Predicted price: ${predicted_price:,.2f}')

if __name__ == '__main__':
    main()


