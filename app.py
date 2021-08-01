# %%writefile app.py
# Importing necessary packages
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px
import pickle

st.set_page_config(page_title="Dashboard | Telecom User Data Analysis ", layout="wide")
st.markdown("<h1 style='color:#0b4eab;font-size:36px;border-radius:10px;'>Dashboard | Pharmaceutical Sales Prediction Model </h1>", unsafe_allow_html=True)

# loading the trained model
pickle_in = open('models/01-08-2021-02-51-29-864.pkl', 'rb') 
classifier = pickle.load(pickle_in)
 
@st.cache()

# defining the function which will make the prediction using the data which the user inputs 
def prediction(Store, DayOfWeek, Promo, StateHoliday, Year, Month, Day, 
                WeekOfYear, days_to_from_hol, Assortment, CompetitionDistance, CompetitionOpenSinceMonth,
                CompetitionOpenSinceYear, Promo2, Promo2SinceWeek, Promo2SinceYear):   
 
    # Pre-processing user input    
    if Promo == 1:
        Promo = 1
    else:
        Promo = 0

    if StateHoliday == "a":
        StateHoliday = 1
    elif StateHoliday == "b":
        StateHoliday = 2
    elif StateHoliday == "c":
        StateHoliday = 3
    else:
        StateHoliday = 0
    
    if Assortment == "a":
        Assortment = 0
    elif Assortment == "b":
        Assortment = 1
    else:
        Assortment = 2
 
    if Promo2 == 1:
        Promo2 = 1
    else:
        Promo2 = 0 
    
    if Promo2SinceWeek == 1:
        Promo2SinceWeek = 1
    else:
        Promo2SinceWeek = 0 
    
    if Promo2SinceYear == 1:
        Promo2SinceYear = 1
    else:
        Promo2SinceYear = 0 
 
    # Making predictions 
    prediction = classifier.predict( 
        [[Store, DayOfWeek, Promo, StateHoliday, Year, Month, Day, WeekOfYear, 
        days_to_from_hol, Assortment, CompetitionDistance, CompetitionOpenSinceMonth,
        CompetitionOpenSinceYear, Promo2, Promo2SinceWeek, Promo2SinceYear]])
     
    if prediction == 0:
        pred = 'Rejected'
    else:
        pred = 'Approved'
    return pred

    # this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    Store = st.number_input('Enter Store number/Id')
    DayOfWeek = st.number_input('Enter DayOfWeek')
    Promo = st.selectbox('Are you in Promo',(1,0))
    StateHoliday = st.selectbox('Is it StateHoliday',(1,0))  
    Year = st.number_input('Enter Year')
    Month = st.number_input('Enter Month')
    Day = st.number_input('Enter Day')
    WeekOfYear = st.number_input('Enter WeekOfYear')
    days_to_from_hol = st.number_input('Enter numbers of days to/from Holiday')
    Assortment = st.selectbox('Choose Assortment',("a","b","c")) 
    CompetitionDistance = st.number_input('Enter Competition distanc')
    CompetitionOpenSinceMonth = st.number_input('Enter new Competition distance of this month')
    CompetitionOpenSinceYear = st.number_input('Enter new Competition distance of this year')
    Promo2 = st.selectbox('Are you in Promo2',(1,0))
    Promo2SinceWeek = st.selectbox('Are you in Promo2 this week',(1,0))
    Promo2SinceYear = st.selectbox('Are you in Promo2 this Year',(1,0))
    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(Store, DayOfWeek, Promo, StateHoliday, Year, Month, Day, WeekOfYear, 
                            days_to_from_hol, Assortment, CompetitionDistance, CompetitionOpenSinceMonth,
                            CompetitionOpenSinceYear, Promo2, Promo2SinceWeek, Promo2SinceYear) 
        st.success('Your Sales is {}'.format(result))
        # print(LoanAmount)
     
if __name__=='__main__': 
    main()


# def loadData():
#     pd.set_option('max_column', None)
#     loaded_data = pd.read_csv('./data/train.csv')
#     return loaded_data

# def selectHandset():
#     df = loadData()
#     handset = st.multiselect("Choose Sore Type(s)", list(df['StoreType'].unique()))
#     if handset:
#         df = df[np.isin(df, handset).any(axis=1)]
#         st.write(df)



# st.markdown("<h1 style='color:#0b4eab;font-size:36px;border-radius:10px;'>Dashboard | Telecommunication Users Data Analysis </h1>", unsafe_allow_html=True)
# selectHandset()
# # st.markdown("<p style='padding:10px; background-color:#000000;color:#00ECB9;font-size:16px;border-radius:10px;'>Section Break</p>", unsafe_allow_html=True)
# st.title("Data Visualizations")
# # with st.beta_expander("Show More Graphs"):