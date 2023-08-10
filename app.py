from predictive_maintenence.utility import pre_process
import tensorflow 
import streamlit as st
from tensorflow.keras.losses import MeanSquaredError

st.title("Prediction of faliure Pressure around faliure dates :calendar:")
date = st.text_input('Give the date where another failure occurred',"2015-04-20")
# data format "2015-04-20"
def predict_data(date):
    model = tensorflow.keras.models.load_model('my_model.keras')
    st_test = pre_process.df_sel.loc[pre_process.df_sel['datetime'] == date].index.values[0]
# Then, filter the data to include approximately two-weeks window
    start_period_test = st_test - 7*24
    end_period_test = st_test + 7*24
    X_test, y_test, test_scaler = pre_process.create_feature(start_period_test, end_period_test)
# Shape the sequence
    X_test_seq = pre_process.shape_sequence(X_test, 5, 0)
    y_test_seq = pre_process.shape_sequence(y_test, 1, 0)
# Predict the testing data
    y_pred_test = model.predict(X_test_seq)
    mse = MeanSquaredError()
    test_err = mse(y_test_seq.reshape(-1,1), y_pred_test)
    test_err.numpy()
    return y_pred_test, test_err

st.button("Predict",on_click=predict_data(date))