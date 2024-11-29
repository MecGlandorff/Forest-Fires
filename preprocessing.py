



import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess(data):

    """ Encodes months like "nov" to numerical values so it can be better handled in heatmaps and the likes, 
        requires pandas dataframe as input and returns pandas dataframe as well"""
    
    # Mappings for months and days
    month_mapping = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
        'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
        'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    day_mapping = {
        'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4,
        'fri': 5, 'sat': 6, 'sun': 7
    }

    # Map 'month' and 'day' to num values
    data['month'] = data['month'].str.lower().map(month_mapping)
    data['day'] = data['day'].str.lower().map(day_mapping)

    return data