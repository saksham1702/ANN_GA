import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import gc

def data(training):
    # Read the manufacturing data
    dataframe = pd.read_csv("manufacturing.csv", sep=',')
    dataset = dataframe.values
    
    # Select input variables (first 5 columns)
    X = dataset[:,0:5]
    # Select only Quality rating as output variable (last column)
    Y = dataset[:,-1]
    
    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=training, random_state=42)
     
    # Clean up memory
    del(dataframe, X, Y)
    gc.collect()
    
    # Scale input variables
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    
    # Scale output variable
    scaler_Y = StandardScaler()
    Y_train = scaler_Y.fit_transform(Y_train.reshape(-1, 1)).flatten()
    Y_test = scaler_Y.transform(Y_test.reshape(-1, 1)).flatten()
    
    return X_train, X_test, Y_train, Y_test
 