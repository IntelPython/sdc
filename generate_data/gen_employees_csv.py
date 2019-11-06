import argparse
import pandas as pd
from datetime import date, time, timedelta
from random import randrange, uniform, random, choices
import numpy as np

"""
This file is used to generate file employees.csv used elsewhere in Pandas API examples for
Intel(R) Scalable Dataframe Compiler.

script gen_employees_csv.py [-h] [--nrows N] [--fname FILE_NAME] 
"""

# Constants
NROWS = 10  # Default number of rows in the generated CSV file

HELP_STR_ROWS = 'Number of rows in the generated file (default:'+str(NROWS)+')'
HELP_STR_FNAME = 'File name of the generated file (default: employees.csv)'

MIN_YEAR = 1989
MAX_YEAR = 2019
MIN_SALARY = 37000
MAX_SALARY = 150000
MIN_BONUS = 1.0
MAX_BONUS = 20.0

MANAGEMENT_FREQ = 0.3

TEAM_LIST = ['Marketing', 'Finance', 'Client Services', 'Legal', 'Product', 'Engineering', 'Business Development',
             'Human Resources', 'Sales', 'Distribution']
TEAM_WEIGHTS = [2, 1, 4, 1, 3, 6, 1, 1, 4, 3]

MISSING_NAME_PROPORTION = 0.1
MISSING_GENDER_PROPORTION = 0.1
MISSING_SENIOR_MANAGEMENT_PROPORTION = 0.1
MISSING_TEAM_PROPORTION = 0.1


def random_date():
    """
    Function generates random date for 'Start Date' column
    """
    start = date(MIN_YEAR, 1, 1)
    years = MAX_YEAR - MIN_YEAR + 1
    end = start + timedelta(days=365 * years)
    delta = end - start
    int_delta = delta.days
    random_days = randrange(int_delta)
    return start + timedelta(days=random_days)


def random_time():
    """
    Function generates random time for 'Last Login Time' column
    """
    return timedelta(seconds=randrange(60*60*24))


def random_management():
    """
    Function generates random 'true' or 'false' for 'Senior Management' column
    """
    return 'true' if random() < MANAGEMENT_FREQ else 'false'


# Argument parser
parser = argparse.ArgumentParser(description='The employee.csv data generator')
parser.add_argument('--nrows', default=NROWS, help=HELP_STR_ROWS, type=int)
parser.add_argument('--fname', default='employees.csv', help=HELP_STR_FNAME, type=argparse.FileType('w'))

args = parser.parse_args()
nrows = args.nrows
fname = args.fname

# Read required dataset for names generation
# The Top25BabyNames-By-Sex-2005-2017.csv is the public dataset from Open Government data.gov
#
# It has been downloaded from:
# https://data.chhs.ca.gov/dataset/4a8cb74f-c4fa-458a-8ab1-5f2c0b2e22e3/resource/2bb8036b-8ce5-42e2-98e0-85ee2dca4093/
# download/top25babynames-by-sex-2005-2017.csv
names_df = pd.read_csv("data/top25babynames-by-sex-2005-2017.csv")  # Reading names-gender dataset
names_df = names_df.rename(columns={'Name': 'First Name'})  # Replacing column 'Name' with 'First Name'
counts = names_df['Count']  # Will use 'Count" column as weights for sampling
names_df = names_df.drop(['YEAR', 'RANK', 'Count'], axis=1)  # These columns are not used for employees.csv generation
names_df = names_df.append({'Gender': '', 'First Name': 'Female'}, ignore_index=True)

employees_df = names_df.sample(nrows, weights=counts, replace=True)  # Sampling names to create dataframe with nrows
employees_df['Start Date'] = [random_date() for i in range(nrows)]  # Adding random start dates
employees_df['Last Login Time'] = [random_time() for i in range(nrows)]  # Adding random login times
employees_df['Salary'] = [randrange(MIN_SALARY, MAX_SALARY) for i in range(nrows)]  # Random salary
employees_df['Bonus %'] = [uniform(MIN_BONUS, MAX_BONUS) for i in range(nrows)]  # Random bonus %
employees_df = employees_df.round({'Bonus %': 3})  # Rounding 'Bonus %' column to 3 decimals
employees_df['Senior Management'] = [random_management() for i in range(nrows)]  # True or False
employees_df['Team'] = choices(TEAM_LIST, k=nrows, weights=TEAM_WEIGHTS)
employees_df.index = [i for i in range(nrows)]

# Generate missing values
employees_df['First Name'].values[np.random.choice(nrows, int(nrows*MISSING_NAME_PROPORTION))] = ''
employees_df['Gender'].values[np.random.choice(nrows, int(nrows*MISSING_GENDER_PROPORTION))] = ''
employees_df['Team'].values[np.random.choice(nrows, int(nrows*MISSING_TEAM_PROPORTION))] = ''
employees_df['Senior Management'].values[np.random.choice(nrows, int(nrows*MISSING_SENIOR_MANAGEMENT_PROPORTION))] = ''

# Writing dataframe to employee.csv file
employees_df.to_csv(fname)
