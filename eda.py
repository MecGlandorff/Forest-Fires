



import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

def sum(data):
    
    """ Summarizes the data features from the data that was loaded from loader module """
    print("Dataset Info:")
    print(data.info())
    print("\nMissing Values:")
    print(data.isnull().sum())
    print("\nDescriptive Statistics:")
    print(data.describe())

def plot_distributions(data, column):
    
    """ Plots the distribution per specified column """

    plt.figure(figsize=(8, 6))
    sns.histplot(data[column], bins=30, kde=True)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

def cor_heatmap(data):

    """ Heatmap of feature correlations """
    num_data = data.select_dtypes(include=['float64', 'int64'])
    plt.figure(figsize=(12, 8))
    sns.heatmap(num_data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()

def scatter_plot(data, x, y):
    
    """ Scatter plot between two variables """

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x=x, y=y)
    plt.title(f"{x} vs {y}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()