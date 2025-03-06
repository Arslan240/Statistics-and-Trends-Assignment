"""
This is the template file for the statistics and trends assignment.
You will be expected to complete all the sections and
make this a fully working, documented file.
You should NOT change any function, file or variable names,
 if they are given to you here.
Make use of the functions presented in the lectures
and ensure your code is PEP-8 compliant, including docstrings.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns


def plot_relational_plot(df):
    """
    Create a relational plot (scatter plot) to show the relationship between two numerical columns.
    Args:
        df (pd.DataFrame): The input dataframe.
    """
    # Convert 'Size (min cm)' and 'Weight (min kg)' to numeric, handling errors
    df['Size (min cm)'] = pd.to_numeric(df['Size (min cm)'], errors='coerce')
    df['Weight (min kg)'] = pd.to_numeric(df['Weight (min kg)'], errors='coerce')

    # Drop rows with missing values after conversion
    df = df.dropna(subset=['Size (min cm)', 'Weight (min kg)'])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x='Size (min cm)', y='Weight (min kg)', hue='Name', ax=ax)
    ax.set_title('Relationship between Size and Weight of Irish Wildlife')
    ax.set_xlabel('Size (min cm)')
    ax.set_ylabel('Weight (min kg)')
    plt.savefig('relational_plot.png')
    plt.close()
    return


def plot_categorical_plot(df):
    """
    Create a categorical plot (bar plot) to show the distribution of a categorical variable.
    Args:
        df (pd.DataFrame): The input dataframe.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    df.groupby('Name')['Population'].mean().plot(kind='bar', ax=ax)
    ax.set_title('Average Population of Irish Wildlife Species')
    ax.set_xlabel('Species Name')
    ax.set_ylabel('Average Population')
    plt.savefig('categorical_plot.png')
    plt.close()
    return


def plot_statistical_plot(df):
    """
    Create a statistical plot (correlation heatmap) to show relationships between numerical columns.
    Args:
        df (pd.DataFrame): The input dataframe.
    """
    # Select only numerical columns for correlation calculation
    numerical_df = df.select_dtypes(include=np.number)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap of Irish Wildlife Statistics')
    plt.savefig('statistical_plot.png')
    plt.close()
    return


def statistical_analysis(df, col: str):
    """
    Calculate the four statistical moments for a given column.
    Args:
        df (pd.DataFrame): The input dataframe.
        col (str): The column to analyze.
    Returns:
        tuple: Mean, standard deviation, skewness, and excess kurtosis.
    """
    # Convert the column to numeric, handling errors
    df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing values in the column
    df = df.dropna(subset=[col])

    mean = df[col].mean()
    stddev = df[col].std()
    skew = df[col].skew()
    excess_kurtosis = df[col].kurtosis()
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """
    Preprocess the dataset by cleaning and exploring it.
    Args:
        df (pd.DataFrame): The input dataframe.
    Returns:
        pd.DataFrame: The cleaned and preprocessed dataframe.
    """
    # Check for missing values
    print("Missing values in each column:")
    print(df.isnull().sum())

    # Fill missing values (if any) or drop them
    df = df.dropna()  

    # Explore the data
    print("\nSummary statistics:")
    print(df.describe())

    print("\nFirst few rows of the dataset:")
    print(df.head())

    print("\nCorrelation matrix:")
    # Select only numerical columns for correlation calculation
    numerical_df = df.select_dtypes(include=np.number)
    print(numerical_df.corr())

    return df


def writing(moments, col):
    """
    Print the results of the statistical analysis.
    Args:
        moments (tuple): Mean, standard deviation, skewness, and excess kurtosis.
        col (str): The column analyzed.
    """
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')

    # Determine skewness and kurtosis type
    skew_type = "right" if moments[2] > 0 else "left" if moments[2] < 0 else "not skewed"
    kurtosis_type = "leptokurtic" if moments[3] > 0 else "platykurtic" if moments[3] < 0 else "mesokurtic"
    print(f'The data was {skew_type} skewed and {kurtosis_type}.')
    return


def main():
    df = pd.read_csv('data.csv')
    df = preprocessing(df)
    col = 'Population' 

    # Ensure the column is numeric
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=[col]) 

    plot_relational_plot(df)
    plot_categorical_plot(df)
    plot_statistical_plot(df)
    moments = statistical_analysis(df, col)
    writing(moments, col)
    return


if __name__ == '__main__':
    main()
