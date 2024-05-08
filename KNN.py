import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

   
def split_dataset(X_data, y_data, train_split_perc):
    size = X_data.shape[0]
    
    # combine X and Y
    result = np.hstack((X_data, y_data))
    
    # shuffle
    np.random.shuffle(result)
    
    # return back to X and Y
    X_data = result[:, :2]
    y_data = result[:, 2]
        
    X_train = X_data[:int(train_split_perc * size),:]
    X_test = X_data[int(train_split_perc * size):,:]
    y_train = y_data[:int(train_split_perc * size)]
    y_test = y_data[int(train_split_perc * size):]
    
    return X_train, X_test, y_train, y_test    
    
def HK_to_USD(value):
    
    return value * 0.13

def calc_MSE(y_true, y_predict):
    mse = 0

    for i in range(len(y_predict)):
        y_i = y_true[i]       
        hx_i = y_predict[i]                
        
        mse += (y_i - hx_i) ** 2
        
    mse = mse * (1 / y_predict.shape[0])
    return mse

def standardize(data):
      # Standardize 
    mean = data.mean()
    std_dev = data.std()
            
    standardized_data = (data - mean) / std_dev
    return standardized_data

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
        
    def euclidean_distance(self, p, q):
        return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)    

    def predict(self, X_test):        
        predictions = []
    
        for i in range(X_test.shape[0]):
            distances = []
            for j in range(self.X_data.shape[0]):
                
                # get q and p values
                q = list(X_test[i, :])
                p = list(self.X_data[j, :])
                
                # calculate euclidean distance
                ED = self.euclidean_distance(p, q)
                distances.append(ED)
                
            
            indices = np.argsort(distances)
            predicted_value = np.mean(self.y_data[indices[0:self.k]])
            predictions.append(predicted_value)
            
        return np.array(predictions)    

####################################################################################
####################################################################################
    
colors = np.random.randint(len(X_data[:, 2]), size=(len(X_data[:, 2])))
plt.scatter(X_data[:, 2], X_data[:, 0], c=colors, cmap='Set3')
plt.title("Time vs Cost")
plt.xlabel("Days Since first date in dataset")
plt.ylabel("Original cost (in USD)")
plt.show()   

""""
Create KNN Regression model
"""

time = np.array(X_data[:, 2])
time = time[:, np.newaxis]

price = np.array(X_data[:, 0])
price = price[:, np.newaxis]

X = np.append(time, price, axis=1)
y = np.array(y_data[:,:])

# split data into training and testing 
train_split = 0.80
X_train_, X_test_, y_train_, y_test_ = split_dataset(X, y, train_split)  


K = 16
reg = KNN_regression(K, X_train_, y_train_)

print("Running KNN from scratch ....")
predictions = reg.predict(X_test_)

MSE = calc_MSE(y_test_, predictions)
RMSE = math.sqrt(MSE)


print("From Scratch")
print("MSE ", MSE)
print("RMSE ",RMSE)
print("\n")

######################################################
##    Test for comparisons of different K values    ##
######################################################


# Hypothesis: k = 15 will yield the lowest RMSE
# using the dataset split, test different k values to find the ideal k

K_values = []
MSE_values = []

print("Checking different K values....")
for K in range(5,21):
    K_values.append(K)
    train_split = 0.80
    X_train, X_test, y_train, y_test = X_train_, X_test_, y_train_, y_test_

    reg = KNN_regression(K, X_train, y_train)
    predictions = reg.predict(X_test)

    MSE = calc_MSE(y_test, predictions)
    RMSE = math.sqrt(MSE)
    
    MSE_values.append(RMSE)
    
    print("From Scratch using K=", str(K))
    print("MSE ", MSE)
    print("RMSE ",RMSE)
    print("\n")


plt.plot(K_values, MSE_values)
plt.title("K values vs RMSE values")
plt.xlabel("K")
plt.ylabel("R Mean Squared Error")
plt.show()  



##############################################
#############Scikit-learn version ############
##############################################
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize

# Use the same data shuffle as first KNN 

X_train_sc, X_test_sc, y_train_sc, y_test_sc = X_train_, X_test_, y_train_, y_test_

neigh = KNeighborsRegressor(n_neighbors=K)

neigh.fit(X_train_sc, y_train_sc)
print("Running Scikit-learn KNN...")
y_pred_sc = neigh.predict(X_test_sc)

MSE_sc = mean_squared_error(y_test_sc, y_pred_sc)
RMSE_sc = math.sqrt(MSE_sc)


print("Scikit-learn Version")
print("MSE ", MSE_sc)
print("RMSE ",RMSE_sc)


plt.scatter(y_pred_sc, y_test_sc, c="orange", label = "Scikit-learn", alpha=0.2)
plt.scatter(predictions, y_test_, c="blue", label = "KNN from Scratch", alpha=0.2)
plt.legend(title = "") 

plt.title("Predictions vs True Percentage Values")
plt.xlabel("KNN Prediction")
plt.ylabel("Original percentage")
plt.show()  










