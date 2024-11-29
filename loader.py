



import pandas as pd  
import os

def load(path):
    
    """ Loads data from a csv file into a pandas dataframe, 
        input must be CSV file to have it correctly return a pd.df """

    if os.path.exists(path):
        data = pd.read_csv(path)
        print("Data load completed!")
        return data
    
    else:
        print("File doesnt seem to be in place")
        return None



### Test row for module 
data = load("data/forestfires.csv")
if data is not None:
    print(data.head())  # Display the first few rows
else:
    print("Failed to load the dataset.")