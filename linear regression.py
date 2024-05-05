import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

def calc_MSE(y_true, y_predict):
    mse = 0

    for i in range(len(y_predict)):
        y_i = y_true[i, :].item()                
        hx_i = y_predict[i]                
        
        mse += (y_i - hx_i) ** 2
        
    mse = mse * (1 / y_predict.shape[0])
    return mse

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

###############################################################
###############################################################

class Linear_regression():
    def __init__(self, X, y, learning_rate):
        self.n = learning_rate      
        self.X = X
        self.training_points = np.concatenate((X, y), axis=1)
        # create weights set all of them to 1
        self.weights = np.ones(shape=(X.shape[0], 2))
                
        pass
        
    # use gradient descent
    def predict(self, x_test, num_interations):
        w = [1, 1]
        delta_w = [0, 0]
        n = 0.01
       
        for i in range(num_interations):
           delta_w = [0, 0]
           for point in self.training_points: 
               # o -> x = z
               o = w[0] + (w[1] * point[0]) 
               
               delta_w[0] = delta_w[0] + n * (point[1] - o)
               delta_w[1] = delta_w[1] + n * (point[1] - o) * point[0]
            
           w[0] += delta_w[0]
           w[1] += delta_w[1]
        
        # predict
        predictions = []
        for j in range(x_test.shape[0]):
            
            w0 = w[0]
            w1 = w[1]
            x_value = x_test[j, 0]
            
            val = w0 + w1 * x_value
            
            predictions.append(val)
            
        return predictions
            

###############################################################
###############################################################

time = np.array(X_data[:, 2])
time = time[:, np.newaxis]

price = np.array(X_data[:, 0])
price = price[:, np.newaxis]

X = np.append(time, price, axis=1)
y = np.array(y_data[:,:])

# only use original price
X = np.delete(X, 0, axis=1)


# split data into training and testing 
train_split = 0.80
X_train, X_test, y_train, y_test = split_dataset(X, y, train_split)  

############################################
#######       Linear regression    #########
############################################

n = 0.01
model = Linear_regression(X_train, y_train, n)
num_iterations = 5

pred = model.predict(X_test, num_iterations)
MSE = calc_MSE(y_test, pred)
RMSE = math.sqrt(MSE)

############################################
####### Scikit-learn Comparison ############
############################################

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

reg = LinearRegression().fit(X_train, y_train)

prediction = reg.predict(y_test)

MSE_sc = mean_squared_error(y_test, prediction)
RMSE_sc = math.sqrt(MSE_sc)


print("Scikit-learn Version")
print("MSE ", MSE_sc)
print("RMSE ",RMSE_sc)


