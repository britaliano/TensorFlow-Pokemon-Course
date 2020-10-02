import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing


df = pd.read_csv('pokemon.csv')

# You can print out all the column names
# print (df.columns)

# Or you can choose the ones you want
df = df[['isLegendary','Generation', 'Type_1', 'Type_2', 'HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed','Color','Egg_Group_1','Height_m','Weight_kg','Body_Style']]

# Now we need to convert a non-integer field into an integer
df['isLegendary'] = df['isLegendary'].astype(int)

# Then create dummy columns for each column in the source data set that has multiple possibble values
# For example, Column 'sport' with values football, swimming, tennis, would create a new column for football, swimming and tennis with either 0 or 1
# depending on the value in the source data
def dummy_creation(df, dummy_categories):
    # For each column passed when the function is called
    for i in dummy_categories:

        # Get the 'dummies' (possible values) for each column
        df_dummy = pd.get_dummies(df[i])

        # Add a new column for each dummy column to the source data
        df = pd.concat([df,df_dummy],axis=1)

        # Then drop the column from the source data as we do not need it
        df = df.drop(i, axis=1)
    return(df)


# Call the above function and pass the columns we want to dummify
df = dummy_creation(df, ['Egg_Group_1', 'Body_Style', 'Color','Type_1', 'Type_2'])

# DEBUG
print (df)

# DEBUG
full_result_dataframe = pd.DataFrame(df)
# print(full_result_dataframe)

# DEBUG
# Print the table to an Excel file
full_result_dataframe.to_excel("PokemonData.xlsx")


# Function to split the data by generation
# Note that column is the column passed when calling the function
def train_test_splitter(DataFrame, column):
    df_train = DataFrame.loc[df[column] != 1]
    df_test = DataFrame.loc[df[column] == 1]

    df_train = df_train.drop(column, axis=1)
    df_test = df_test.drop(column, axis=1)

    return(df_train, df_test)
# Call the function, passing Generation as the column with the values that decide which data set each row goes into
df_train, df_test = train_test_splitter(df, 'Generation')



# Function that separates the labels (the data that we want the model to learn from) and the values of the training and testing data. This gives us 4 columns.
def label_delineator(df_train, df_test, label):
    
    train_data = df_train.drop(label, axis=1).values
    train_labels = df_train[label].values
    test_data = df_test.drop(label,axis=1).values
    test_labels = df_test[label].values
    return(train_data, train_labels, test_data, test_labels)

# Call the function, passing in the name of the column that holds the labels we want to run against - isLegendary
train_data, train_labels, test_data, test_labels = label_delineator(df_train, df_test, 'isLegendary')


# DEBUG
train_data = pd.DataFrame(train_data)
train_data.to_excel("TrainingData.xlsx")

# DEBUG
test_data = pd.DataFrame(test_data)
test_data.to_excel("TestingData.xlsx")

""" print(test_data)
print("NEXT")
print(train_data) """


def data_normalizer(train_data, test_data):
    train_data = preprocessing.MinMaxScaler().fit_transform(train_data)
    test_data = preprocessing.MinMaxScaler().fit_transform(test_data)
    return(train_data, test_data)

train_data, test_data = data_normalizer(train_data, test_data)


length = train_data.shape[1]

model = keras.Sequential()
model.add(keras.layers.Dense(500, activation='relu', input_shape=[length,]))
model.add(keras.layers.Dense(2, activation='softmax'))


# Added at the bottom