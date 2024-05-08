import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

def normalize(X):
    mean = np.mean(X)
    std_dev = np.std(X)
    normalized = (X - mean) / std_dev
    return normalized

def calc_MSE(y_true, y_predict):
    mse = 0

    for i in range(len(y_predict)):
        y_i = y_true[i] 
        hx_i = y_predict[i]                
        
        mse += (y_i - hx_i) ** 2
        
    mse = mse * (1 / len(y_predict))
    return mse

def split_dataset(X_data, y_data, train_split_perc):
    size = X_data.shape[0]
    
    # combine X and Y
    result = np.hstack((X_data, y_data))
    
    # shuffle
    np.random.shuffle(result)
    
    # return back to X and Y
    X_data = result[:, :1]
    y_data = result[:, 1]
        
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
        self.training_points = np.hstack((X, y))
        # create weights set all of them to 1
        self.weights = np.ones(shape=(X.shape[0], 2))
                
        pass
        
    # use gradient descent
    def predict(self, x_test, num_interations, n):
        
        # initialize weights to be random values between [-1, 1]
        w = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
        
        # Until the termination condition is met
        for _ in range(num_interations):
            # Initialize each delta_wi to zero
            delta_w = [0, 0]
            
            #For each (x, t) in training examples, Do
            for i in range(self.training_points.shape[0]):
                # (x, t) pair
                x_t = self.training_points[i]
                # Input the instance x' to the unit and compute the output o
                o = w[0] + (w[1] * x_t[0])
                # if o > 0, set to 1 otherwise set to -1
                o = 1 if o > 0 else -1
                
                # For each linear unit weight w, Do
                delta_w[0] = delta_w[0] + n * (x_t[1] - o)
                delta_w[1] = delta_w[1] + n * (x_t[1] - o) * x_t[0]
                
            # For each linear unit weight wi, Do
            w[0] += delta_w[0]
            w[1] += delta_w[1]
          
        predictions = []
        # calculate x_test data
        i = 0
        for i in range(x_test.shape[0]):
            result = w[0] + (w[1] * x_test[i])
            pred = np.asarray(result).item()
            predictions.append(pred)
        
        return predictions

###############################################################
###############################################################

time = np.array(X_data[:, 2])
time = time[:, np.newaxis]

price = np.array(X_data[:, 0])
price = price[:, np.newaxis]

X = np.append(time, price, axis=1)
y = np.array(y_data[:,:])

X = np.delete(X, 1, axis=1)

# split data into training and testing 
train_split = 0.80
X_train, X_test, y_train, y_test_ = split_dataset(X, y, train_split)  

############################################
#######       Linear regression    #########
############################################

n = 0.05

# reshape y 
y_train = y_train.reshape(-1, 1)

# Model
model = Linear_regression(X_train, y_train, n)
num_iterations = 1000

pred = model.predict(X_test, num_iterations , n)
MSE = calc_MSE(y_test_, pred)
RMSE = math.sqrt(MSE)

print(MSE)
print(RMSE)

############################################
####### Scikit-learn Comparison ############
############################################

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

reg = LinearRegression().fit(X_train, y_train)

y_test = y_test_.reshape(-1, 1)

prediction = reg.predict(y_test)

MSE_sc = mean_squared_error(y_test, prediction)
RMSE_sc = math.sqrt(MSE_sc)


print("Scikit-learn Version")
print("MSE ", MSE_sc)
print("RMSE ",RMSE_sc)


############################################
####### TensorFlow ############
############################################
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = X.astype(np.float32)
y = y.astype(np.float32)

X = X.reshape(-1, 1) 
y = y.reshape(-1, 1)

X_train, X_test, y_train, y_test_tf = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(1)  
])

model.compile(optimizer='adam', loss='mse')  

model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=0)  
mse = model.evaluate(X_test_scaled, y_test_tf, verbose=0)
print("Mean Squared Error on test set:", mse)

# Make predictions
predictions_tf = model.predict(X_test_scaled)

plt.scatter(prediction, y_test, c="orange", label = "Scikit-learn")
plt.scatter(predictions_tf, y_test_tf, c="blue", label = "TensorFlow")
plt.scatter(pred, y_test_, c="red", label = "Gradient Descent")

plt.legend(title = "") 

plt.title("Predictions vs True Percentage Values")
plt.xlabel("LR Prediction")
plt.ylabel("Original percentage")
plt.show()  
