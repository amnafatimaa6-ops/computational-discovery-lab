import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def train_model():

    df = pd.read_csv("House_prices.csv")


    df = df.dropna(axis=1, how='all')

    if 'property_type' not in df.columns:
        df['property_type'] = 'House'  
    if 'furnishing_status' not in df.columns:
        df['furnishing_status'] = 'Furnished'  

    df = df[['bedrooms', 'bathrooms', 'area sqft', 'location', 'price', 'property_type', 'furnishing_status']]


    location_avg = df.groupby('location')['price'].mean()
    df['location_avg_price'] = df['location'].map(location_avg)


    df['property_type_House'] = (df['property_type'] == 'House').astype(int)
    df['furnishing_status_Unfurnished'] = (df['furnishing_status'] == 'Unfurnished').astype(int)


    X = df[['bedrooms', 'bathrooms', 'area sqft', 'location_avg_price', 'property_type_House', 'furnishing_status_Unfurnished']]
    y = df['price']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

 
    model_r2 = model.score(X, y)

    
    return model, location_avg, model_r2
