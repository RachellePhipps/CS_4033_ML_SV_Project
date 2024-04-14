import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
    
def split_dataset(X_data, y_data, train_split_perc):
    size = X_data.shape[0]
    
    X_train = X_data[:int(train_split_perc * size),:]
    X_test = X_data[int(train_split_perc * size):,:]
    y_train = y_data[:int(train_split_perc * size)]
    y_test = y_data[int(train_split_perc * size):]
    
    return X_train, X_test, y_train, y_test    
    
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

####################################################################################
"""
K-NN
"""
class KNN_regression():
    def __init__(self, K, X, y):
        self.k = K
        self.X_data = X
        self.y_data = y
        pass

    def standardize(self, data):
        # Standardize 
        mean = data.mean()
        std_dev = data.std()
        
        standardized_data = (data - mean) / std_dev
        return standardized_data
        
    def euclidean_distance(self, p, q):
        return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)    

    def predict(self, X_test):
        # standardize everything
        # self.X_data = self.standardize(self.X_data)
        # X_test = self.standardize(X_test)
        
        predictions = []
    
        for i in range(X_test.shape[0]):
            distances = []
            for j in range(self.X_data.shape[0]):
                q = list(X_test[i, :])
                p = list(self.X_data[j, :])
                # calculate euclidean distance
                ED = self.euclidean_distance(p, q)
                distances.append(ED)
            indices = np.argsort(distances)
            predicted_value = np.mean(self.X_data[indices[0:self.k], -1])
            predictions.append(predicted_value)
            
        return np.array(predictions)    

####################################################################################
# First attempt, trying predicting using Originalprice, time = X, DiscountPCT = y
# Plot the values initally as a scatterplot:
plt.scatter(X_data[:, 2], X_data[:, 0])
plt.title("Time vs Cost")
plt.xlabel("Days Since first date in dataset")
plt.ylabel("Original cost")
plt.show()   

""""
Create KNN Regression model
"""
# X = (Time, Original Price)
# y = (Discount percentage)
time = np.array(X_data[:, 2])
time = time[:, np.newaxis]

price = np.array(X_data[:, 0])
price = price[:, np.newaxis]

X = np.append(time, price, axis=1)
y = y_data

# split data into training and testing {.70/.30}
train_split = 0.70
X_train, X_test, y_train, y_test = split_dataset(X, y, train_split)   

# KNN model
K = 10
reg = KNN_regression(K, X_train, y_train)
predictions = reg.predict(X_test)

plt.scatter(predictions, y_test)
plt.title("Predictions vs True")
plt.xlabel("Prediction")
plt.ylabel("Original percentage")
plt.show()   













