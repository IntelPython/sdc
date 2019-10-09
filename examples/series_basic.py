import pandas as pd
import numpy as np
import hpat

@hpat.jit
def get_mean(df):
    ser = pd.Series(df['Bonus %'])
    m = ser.mean()
    return m

@hpat.jit
def sort_name(df):
    ser = pd.Series(df['First Name'])
    m = ser.sort_values()
    return m
    

file = "employees.csv"
df = pd.read_csv(file)


#find mean of one column
print(get_mean(df))

#Sort the names in ascending order
print(sort_name(df))