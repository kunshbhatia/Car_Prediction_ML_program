### This data has been cleaned and all EDA etc. has been performed , if you need raw data ( Refer :- Vehicle Price.csv)

import pandas as pd
import streamlit as st

### Reading The CSV Data (Vehicle_final.csv) , (Download From The GitHub !!!) and making X and Y for machine learning 

df = pd.read_csv("Vehicle_final.csv")
df = df.drop(["exterior_color","interior_color"],axis=1)
df.reset_index(drop=True,inplace=True)
X = df[['make','model','year',"fuel","mileage","transmission","body","doors","drivetrain"]]
Y = df['price']

### Doing Train Test Split and making the data ready for Machine learning 

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=42)

### Encoading The Data

make = ["Chrysler","RAM","Jeep","Dodge","GMC","Nissan","Ford","Chevrolet","Kia","Hyundai","Subaru","Mazda","Toyota","Honda","Volkswagen","Buick","Volvo","Lincoln","Audi","Acura","INFINITI","Cadillac","Genesis","Jaguar","Land Rover","Lexus","BMW","Mercedes-Benz"]
model = ['Forte', 'Sentra', 'Versa', 'Soul', 'Trax', 'HR-V', 'Encore GX', 'Kicks', 'Impreza', 'Jetta', 'Altima', 'Corolla', 'Elantra HEV', 'Niro', 'Seltos', 'CX-70', 'CX-90', 'Legacy', 'Sonata', 'Sonata Hybrid', 'Tucson', 'Tucson Hybrid', 'Kona', 'Sportage', 'Sportage Hybrid', 'Compass', 'Equinox', 'Equinox EV', 'CR-V', 'CR-V Hybrid', 'Rogue', 'RAV4 Prime', 'Outback', 'Santa Fe', 'Taos', 'Murano', 'Bronco Sport', 'Trailblazer', 'Edge', 'Pacifica', 'Pacifica Hybrid', 'Voyager', 'CX-90 PHEV', 'ID.4', 'IONIQ 5', 'IONIQ 6', 'EV6', 'EV9', 'Prologue', 'Pathfinder', 'Palisade', 'Telluride', 'Grand Cherokee', 'Grand Cherokee L', 'Grand Cherokee 4xe', 'Wagoneer', 'Wagoneer L', 'Grand Wagoneer', 'Wrangler', 'Wrangler 4xe', 'Gladiator', 'Bronco', 'Expedition', 'Transit Connect', 'Transit-150', 'Transit-250', 'Transit-350', 'F-150', 'F-350', '2500', '3500', 'Silverado 1500', 'Silverado 2500', 'Sierra 1500', 'Sierra 2500', 'Sierra 3500', 'Titan', 'Frontier', 'Durango', 'Charger', 'Hornet', 'XT5', 'XT6', 'Enclave', 'Envista', 'Corsair', 'Nautilus', 'ZDX', 'QX50', 'QX55', 'LYRIQ', 'X3', 'X7', 'M235 Gran Coupe', '530', '740', 'i4 Gran Coupe', 'i5', 'i7', 'A3', 'A5 Sportback', 'SQ5', 'Q5 e', 'Q8 e-tron', 'SQ8 e-tron', 'RS e-tron GT', 'GLE 350', 'GLE 450', 'GLS 450', 'AMG C 43', 'AMG GLE 53', 'GLA 250', 'MX-5 Miata RF', 'Range Rover Evoque', 'Discovery Sport', 'XC90 Recharge Plug-In Hybrid', 'Electrified G80', 'Electrified GV70', 'S60 Recharge Plug-In Hybrid', 'EQE 350+', 'EQS 450', 'I-PACE', 'C40 Recharge Pure Electric', 'Envision', '300', 'Odyssey', 'Savana 2500', 'Sprinter 2500', 'Sprinter 3500', 'ProMaster 1500', 'ProMaster 2500', 'ProMaster 3500', 'Atlas', 'Atlas Cross Sport', 'Explorer', 'Defender', 'Escape', 'Santa Cruz', 'Passport', 'Yukon XL', 'Blazer EV', 'Terrain', 'Mustang Mach-E', 'Blazer', 'MDX', 'Tundra Hybrid', 'Ranger', 'Solterra', 'RX 500h', 'Sorento', 'Sorento Hybrid', 'Corolla Cross', 'Maverick']
fuel = ['Gasoline','Diesel','E85 Flex Fuel','Hybrid','Diesel (B20 capable)',"Electric",'PHEV Hybrid Fuel',]
transmission = ['8-Speed Automatic', 'Automatic', '6-Speed Automatic', 'Automatic CVT', '10-Speed Automatic', '8-Speed Automatic with Auto-Shift', '1-Speed Automatic', '7-Speed DSG Automatic with Tiptronic', '8-Speed Automatic with Tiptronic', '9-Speed Automatic', '8-speed automatic', 'CVT', '7-Speed Automatic S tronic', '7-Speed DSGA? Automatic w/ 4MO', '6-Speed Automatic Electronic with Overdrive', '9-Speed 948TE Automatic', '8-Speed A/T', 'Variable', '7-Speed Automatic with Auto-Shift', '9-Speed A/T', '9 Spd Automatic', 'Aisin 6-Speed Automatic', '6-Spd Aisin F21-250 PHEV Auto Trans', '62 kWh battery', '10-Speed Shiftable Automatic', '9-speed automatic', '6-SPEED AUTOMATIC', 'automatic w/paddle shifters', 'A/T', '8-Speed Shiftable Automatic', '6-Speed DCT Automatic', '6-Speed A/T', '(CVT) CONT VAR.', 'CVT with Xtronic', '8 Speed Dual Clutch', '10-Speed Automatic with Overdrive', '8-Speed Automatic Sport']
body = ['SUV', 'Pickup Truck', 'Sedan', 'Passenger Van', 'Cargo Van', 'Hatchback', 'Convertible', 'Minivan']
drivetrain = ['Four-wheel Drive', 'All-wheel Drive', 'Rear-wheel Drive', 'Front-wheel Drive']
year = ["2023","2024","2025"]
from sklearn.preprocessing import OrdinalEncoder
encode_make = OrdinalEncoder(categories=[make])
encode_model = OrdinalEncoder(categories=[model])
encode_fuel = OrdinalEncoder(categories=[fuel])
encode_transmission = OrdinalEncoder(categories=[transmission])
encode_body = OrdinalEncoder(categories=[body])
encode_drivetrain = OrdinalEncoder(categories=[drivetrain])
encode_year = OrdinalEncoder(categories=[year])
X_train['make'] = encode_make.fit_transform(X_train[['make']])
X_train['model'] = encode_model.fit_transform(X_train[['model']])
X_train['transmission'] = encode_transmission.fit_transform(X_train[['transmission']])
X_train['fuel'] = encode_fuel.fit_transform(X_train[['fuel']])
X_train['body'] = encode_body.fit_transform(X_train[['body']])
X_train['drivetrain'] = encode_drivetrain.fit_transform(X_train[['drivetrain']])
X_train['year'] = encode_year.fit_transform(X_train[['year']])

X_test['make'] = encode_make.transform(X_test[['make']])
X_test['model'] = encode_model.transform(X_test[['model']])
X_test['transmission'] = encode_transmission.transform(X_test[['transmission']])
X_test['fuel'] = encode_fuel.transform(X_test[['fuel']])
X_test['body'] = encode_body.transform(X_test[['body']])
X_test['drivetrain'] = encode_drivetrain.transform(X_test[['drivetrain']])
X_test['year'] = encode_year.transform(X_test[['year']])

### Using Random Forest , having a good R2 score ( accuracy )
### Hyperparameters has been check using RandomizedSearchCV (Check the other file for that :- Vehicle Train Set.ipynb)

from sklearn.ensemble import RandomForestRegressor
regg = RandomForestRegressor(n_estimators=100, max_depth=10 , criterion= 'squared_error')
regg.fit(X_train,Y_train)
Y_pred = regg.predict(X_test)

# Customizing The StreamLit Page

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered",  # or "wide"
    initial_sidebar_state="auto"
)

st.markdown("""
    <h1 style='text-align: center; color: #007acc;'>🚘 Car Price Prediction App</h1>
    <p style='text-align: center; color: gray;'>Get accurate car price predictions using machine learning!</p>
    <hr style='border: 1px solid #f2f2f2;'>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        body {
            background-color: #f9f9f9;
        }
        .stButton>button {
            background-color: #007acc;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)



### Generating a sidebar

st.sidebar.title("🔧 Options")
for i in range(len(df['make'].unique())):
    selected_make = st.sidebar.text(f"{i+1}) {df['make'].unique()[i]}")
    

### Input The Data Required for prediction 

NAME = st.selectbox(options=df['make'].unique(),label="Name Of The Car Brand")
MODEL = st.selectbox("Model Of The Car", options=df[df['make'] == NAME]['model'].unique())

with st.form("More Info"):   
    YEAR = st.selectbox(options=df['year'].unique(),label="Year Of The Car")
    FUEL = st.selectbox(options=df[df['model'] == MODEL]['fuel'].unique(),label="Fuel Type Of The Car")
    MILEAGE = st.number_input(label="Mileage Of The Car")
    TRANSMISSION = st.selectbox(options=df[df['model'] == MODEL]['transmission'].unique(),label="Transmission Of The Car")
    BODY = st.selectbox(options=df[df['model'] == MODEL]['body'].unique(),label="Body Type Of The Car")
    DOOR = st.selectbox(options=df[df['model'] == MODEL]['doors'].unique(),label="Number Of Doors In The Car")
    DATATRAIN = st.selectbox(options=df[df['model'] == MODEL]['drivetrain'].unique(),label="Drivetrain Of The Car")
    submitted = st.form_submit_button("Submit")
    if submitted:
        if not MODEL or not YEAR or not FUEL or not MILEAGE or not TRANSMISSION or not BODY or not DOOR or not DATATRAIN:
            st.text("Please Enter Valid Data")
        else:
            columns = ['make', 'model', 'year', 'fuel', 'mileage', 'transmission', 'body','doors', 'drivetrain']
            data = pd.DataFrame([[encode_make.transform([[NAME]]),encode_model.transform([[MODEL]]),encode_year.transform([[YEAR]]),encode_fuel.transform([[FUEL]]),MILEAGE,encode_transmission.transform([[TRANSMISSION]]),encode_body.transform([[BODY]]),DOOR,encode_drivetrain.transform([[DATATRAIN]])]],columns=columns)
            st.markdown(f"The Price For The Given Data : **${round(regg.predict(data)[0],1)}**")
    else:
        st.text("Please Enter The Data")


# Project Finished and now we can predict !!!

# Adding Footer (just decoration)

st.markdown("""<p style="margin-top: 60px; text-align: center; color: gray;">
  © 2025 Kunsh Bhatia | Built with ❤️ and ☕ in Streamlit
</p>
""", unsafe_allow_html=True)