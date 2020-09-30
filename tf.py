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
# df = df[['isLegendary','Generation', 'Type_1', 'Type_2', 'HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed','Color','Egg_Group_1','Height_m','Weight_kg','Body_Style']]

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
print (df)

full_result_dataframe = pd.DataFrame(df)
print(full_result_dataframe)
# Print the table to an Excel file
full_result_dataframe.to_excel("PokemonData.xlsx")