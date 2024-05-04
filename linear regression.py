import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

def HK_to_USD(value):
    
    return value * 0.13

# pre-process the data
def preprocess():
    data = pd.read_csv("historic_deals.csv")
        
    print(data.shape)
    
    ######### clean data ##########
        
    # If DiscountPCT, Releasedate, or OriginalPrice are empty, delete the row
    for index, row in data.iterrows():
        DiscountPCT = row['DiscountPCT']
        Releasedate = row['ReleaseDate']
        OriginalPrice = row['OriginalPrice']
        
        # Check for nan values
        if (isinstance(DiscountPCT, float) or
            isinstance(Releasedate, float) or
            isinstance(OriginalPrice, float)):
            data = data.drop(index)
            
        elif (DiscountPCT == 'Trial'):
            data = data.drop(index)
     
    print(data.shape)
    
    ### Remove Releasedate and OriginalPrice ###
    x_dataframe = data[['OriginalPrice', 'ReleaseDate']].copy()
    
    # add 3 new columns for "day", "month" "year"
    x_dataframe.insert(0, 'Day', 0)
    x_dataframe.insert(0, 'Month', 0)
    x_dataframe.insert(0, 'Year', 0)
    
    ## clean data ##
    for index, row in x_dataframe.iterrows():
        Releasedate = row['ReleaseDate']
        OriginalPrice = row['OriginalPrice']
        
        # clean releasedate
        day_str, month_str, year_str = Releasedate.split("/")
    
        # Convert to integers
        day = int(day_str)
        month = int(month_str)
        year = int(year_str)
        
        x_dataframe.loc[index, 'Day'] = day
        x_dataframe.loc[index, 'Month'] = month
        x_dataframe.loc[index, 'Year'] = year
        
        # clean Original Price
        OriginalPrice = OriginalPrice.removeprefix('HK$')
        
        if (OriginalPrice.find(',')):
            OriginalPrice = OriginalPrice.replace(',','')
        
        OriginalPrice = float(OriginalPrice)
        
        x_dataframe.loc[index, 'OriginalPrice'] = HK_to_USD(OriginalPrice)
    
    # create single column based on OLDEST date
    x_dataframe['date'] = pd.to_datetime(x_dataframe[['Year', 'Month', 'Day']])
    reference_date = x_dataframe['date'].min()
    x_dataframe['date_numeric'] = (x_dataframe['date'] - reference_date).dt.days
    
    x_dataframe = x_dataframe.drop(['Day', 'Month', 'Year'], axis=1)
    
    
    # delete ReleaseDate column as it is no longer needed
    x_dataframe = x_dataframe.drop('ReleaseDate', axis=1)
    
    
    # Remove DiscountPCT to its own Dataframe and clean it
    y_dataframe = data[['DiscountPCT']].copy()
    
    # remove '%' and turn all values into ints
    for index, row in y_dataframe.iterrows():
        value = row['DiscountPCT']
        
        row['DiscountPCT'] = int(value.replace("%", ""))
    
    
    # convert the dataframes to numpy arrays and return them for use
    X_data = x_dataframe.to_numpy()
    y_data = y_dataframe.to_numpy()
    
    return X_data, y_data

X_data, y_data = preprocess()


