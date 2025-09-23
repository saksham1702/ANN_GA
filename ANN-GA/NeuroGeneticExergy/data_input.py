import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import gc

def data(training):
    dataframe = pd.read_csv("manufacturing.csv", sep=',', header=0)  # header=0 to skip header row
    dataset = dataframe.values
    
    #Locate input variables in the dataset (5 features: Temperature, Pressure, Temp*Pressure, Material Fusion, Material Transformation)
    X = dataset[:,0:5]  # First 5 columns as input
    #Locate output variables in the dataset (Quality Rating)
    Y = dataset[:,-1:] # Last column as output
    
        
    #split X, Y into a train and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=training, random_state=42)
     
    del(dataframe, X, Y)
    gc.collect()
    
    # Scale input features only (X), not the output (Y) for manufacturing quality data
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    
    # For manufacturing quality (0-100 scale), normalize Y to 0-1 range instead of standardizing
    Y_train = Y_train / 100.0
    Y_test = Y_test / 100.0
    
    return X_train, X_test, Y_train, Y_test
 