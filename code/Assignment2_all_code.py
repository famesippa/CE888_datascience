# General for data analytic
import numpy as np
import pandas as pd
import glob

# vitualization
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# Model
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, ExtraTreesClassifier


# skip warning
from sklearn import preprocessing
import warnings
warnings.filterwarnings( action= 'ignore')

#Exporing data and analysis
files = glob.glob("/home/sw22389/assignment/EyeT/EyeT_group_dataset_II_image_name_grey_blue_participant_2_trial_*.csv")
dfs = []
for file in files:
    df_sample = pd.read_csv(file,low_memory=False)
    dfs.append(df_sample)
concatenated_df = pd.concat(dfs)
df_sample = concatenated_df

# Convert the string values to float values

df['Pupil diameter left'] = df['Pupil diameter left'].apply(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)
df['Pupil diameter right'] = df['Pupil diameter right'].apply(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)
df_left_drop_nan = df.dropna(subset=['Pupil diameter left'])
df_right_drop_nan = df.dropna(subset=['Pupil diameter right'])
df_gaze_x_drop_nan = df.dropna(subset=['Gaze point X'])
df_gaze_y_drop_nan = df.dropna(subset=['Gaze point Y'])

from datetime import datetime

# start time
start_time = "1986-05-03 14:08:35.174000"
end_time = "1986-04-28 14:49:56.917000"

# convert time string to datetime
t1 = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S.%f")
print('Start time:', t1.time())

t2 = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S.%f")
print('End time:', t2.time())

# get difference
delta = t2 - t1

# time difference in seconds
print(delta)
print(f"Time difference is {delta.total_seconds()} seconds")

# time difference in milliseconds
ms = delta.total_seconds() * 1000
print(f"Time difference is {ms} milliseconds")

df_Q_A=pd.read_csv("/home/sw22389/assignment/Questionnaire_datasetIA.csv",encoding="cp1252")
df_Q_B=pd.read_csv("/home/sw22389/assignment/Questionnaire_datasetIB.csv",encoding="cp1252")


#Pre-processing

files = glob.glob("/home/sw22389/assignment/EyeT/EyeT*.csv")
dfs = []
for file in files:
    df = pd.read_csv(file,low_memory=False)
    dfs.append(df)
concatenated_df = pd.concat(dfs)
df = concatenated_df


participant_counts = df.groupby('Participant name')['Recording name'].nunique()

#convert value str number to int
df['Participant name'] = df['Participant name'].str.extract('(\d+)').astype(int)
df['Recording name'] = df['Recording name'].str.extract('(\d+)').astype(int)

# Change Recording name in order form.
for participant_name in range(1, 61):
    recording_names = df[df['Participant name'] == participant_name]['Recording name'].unique()
    new_recording_names = (recording_names - recording_names.min() + 1).tolist()

    # Update the DataFrame with the new recording names
    df.loc[df['Participant name'] == participant_name, 'Recording name'] = df[df['Participant name'] == participant_name]['Recording name'].replace(recording_names, new_recording_names)


#Cut some participates out
#For over attempt participants 1,2,3,4,5,6,7,9, For under attempt participants 26,32

df = df[~df['Participant name'].isin([1,2,3,4,5,6,7,9,26,32])]
df['Participant name'].unique()


# Group by Recording name and Participant name
grouped = df.groupby(['Recording name', 'Participant name'])

# Initialize Elapsed time column with NaN
df['Elapsed time'] = np.nan

# Iterate over groups
for (recording, participant), group in grouped:
    
    # Find start and end timestamps
    start_time = group.loc[group['Event'] == 'ImageStimulusStart', 'Recording timestamp'].iloc[0]
    end_time = group.loc[group['Event'] == 'ImageStimulusEnd', 'Recording timestamp'].iloc[0]
    
    # Calculate elapsed time in milliseconds
    elapsed_time = end_time - start_time
    
    # Fill Elapsed time column for this group
    df.loc[(df['Recording name'] == recording) & (df['Participant name'] == participant), 'Elapsed time'] = elapsed_time

#Remove MouseEvent
df = df[df['Event'] != 'MouseEvent']

#change column name
df_Q_A = df_Q_A.rename(columns={'Participant nr': 'Participant name'})
df_Q_B = df_Q_B.rename(columns={'Participant nr': 'Participant name'})


#Changed null-values and EyesNotFound to the event type Unclassified.
df['Eye movement type'] = df['Eye movement type'].fillna('Unclassified')
df['Eye movement type'] = df['Eye movement type'].replace('EyesNotFound', 'Unclassified')


#Create new dataframe to train model
# select unique values from 'Participant name' column
unique_names = df['Participant name'].unique()

# create new dataframe with unique names
new_df = pd.DataFrame({'Participant name': unique_names})
new_df = new_df.sort_values('Participant name').reset_index(drop=True)

# print new dataframe
new_df

# group by 'Participant name' and select unique 'Recording name' values for each group
unique_recordings = df.groupby('Participant name')['Recording name'].apply(lambda x: x.unique()).reset_index()

# explode the unique recordings to create multiple rows for each 'Participant name'
new_df = unique_recordings.explode('Recording name')

# sort the new dataframe by 'Participant name'
new_df = new_df.sort_values('Participant name').reset_index(drop=True)

# print new dataframe
print(new_df)

#calculate pupil mean
df['Pupil diameter mean'] = (df['Pupil diameter left'] + df['Pupil diameter right']) / 2

# group the dataframe by 'Participant name' and 'Recording name' and calculate the required statistics for 'Pupil diameter mean'
grouped_df = df.groupby(['Participant name', 'Recording name'])['Pupil diameter mean'].agg([np.mean, np.max, np.min, np.std])

# rename the columns to match the desired output
grouped_df = grouped_df.rename(columns={
    'mean': 'Pupil diameter mean',
    'amax': 'Pupil diameter max',
    'amin': 'Pupil diameter min',
    'std': 'Pupil diameter std'
})

# reset the index to make 'Participant name' and 'Recording name' regular columns
grouped_df = grouped_df.reset_index()

# display the new dataframe
print(grouped_df)
grouped_df


# group the grouped dataframe by 'Participant name' and calculate the mean and std of 'Pupil diameter mean' column for each group
participant_mean_std_df = grouped_df.groupby('Participant name')['Pupil diameter mean'].agg([np.mean, np.std])

# rename the columns to match the desired output
participant_mean_std_df = participant_mean_std_df.rename(columns={
    'mean': 'Pupil diameter mean total',
    'std': 'Pupil diameter std total'
})

# display the new dataframe
participant_mean_std_df


merged_df = pd.merge(grouped_df, participant_mean_std_df, on='Participant name')
merged_df

# group the dataframe by 'Recording name' and calculate the fraction of 'Fixation' and 'Saccade' for each group
count_df = df.groupby(['Participant name', 'Recording name'])['Eye movement type'].value_counts(normalize=True)
fractions_df = count_df.unstack()
fractions_df

fractions_merge_df = pd.merge(merged_df, fractions_df[['Fixation', 'Saccade']], on=['Participant name', 'Recording name'])
fractions_merge_df = fractions_merge_df.rename(columns={'Fixation': 'Fixation fraction', 'Saccade': 'Saccade fraction'})
fractions_merge_df

# get the unique values of 'Participant name' in fractions_merge_df
unique_participants = fractions_merge_df['Participant name'].unique()

# merge the dataframes using inner join and the subset of unique 'Participant name'
Q_A_merge_df = pd.merge(df_Q_A[df_Q_A['Participant name'].isin(unique_participants)][['Participant name', 'Total Score extended']], 
                        fractions_merge_df, on=['Participant name'])
Q_A_merge_df = Q_A_merge_df.rename(columns={'Total Score extended': 'Total Score extended after'})
                                            
                                            
final_df = pd.merge(df_Q_B[df_Q_B['Participant name'].isin(unique_participants)][['Participant name', 'Total Score extended']], 
                        Q_A_merge_df, on=['Participant name'])
final_df = final_df.rename(columns={'Total Score extended': 'Total Score extended before'})

final_df['Score diff'] = final_df['Total Score extended after'] - final_df['Total Score extended before']


final_df['Group'] = final_df['Participant name'].apply(lambda x: 0 if x % 2 == 0 else 1)

final_df

#save file to train model
final_df.to_csv('final_df.csv', index=False)


#read preprocess file
file = "/home/sw22389/assignment/final_df.csv"
df = pd.read_csv(file,low_memory=False)

# select the relevant features
#X = df[['Pupil diameter mean', 'Pupil diameter max', 'Pupil diameter min', 'Fixation fraction', 'Saccade fraction', 'Group']]
#X = df[['Pupil diameter mean total', 'Pupil diameter std total','Pupil diameter max', 'Pupil diameter min', 'Fixation fraction', 'Saccade fraction']]
X = df[['Pupil diameter mean total','Pupil diameter max', 'Pupil diameter min']]

# set the target variable
y = df['Total Score extended after']

# create the model

models = [
    DecisionTreeRegressor(),
    LinearRegression(),
    GradientBoostingRegressor(),
    ElasticNet(),
    SGDRegressor(),
    SVR(),
    BayesianRidge(),
    CatBoostRegressor(),
    KernelRidge(),
    XGBRegressor(),
    LGBMRegressor()
]

# set up the KFold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=5)

# initial parameters
x = 0
mse_dict = {}
mae_dict = {}
r2 = {}

# loop over models
for model in models:
    mse_list = []
    mae_list = []
    r2_list = []

    # loop over the folds
    for train_index, test_index in kf.split(X):
        # split data into training and testing sets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # fit the model
        model.fit(X_train, y_train)

        # make predictions on the testing set
        y_pred = model.predict(X_test)

        # evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
 
    
        # append the scores to the lists
        mse_list.append(mse)
        mae_list.append(mae)
        r2_list.append(r2)
    
        # add the list of scores to the dictionary
        if x == 0:
            model_x = "DecisionTreeRegressor"
        elif x == 1:
            model_x = "LinearRegression"
        elif x == 2:
            model_x = "GradientBoostingRegressor"
        elif x == 3:
            model_x = "ElasticNet"
        elif x == 4:
            model_x = "SGDRegressor"
        elif x == 5:
            model_x = "SVR"
        elif x == 6:
            model_x = "BayesianRidge"
        elif x == 7:
            model_x = "CatBoostRegressor"
        elif x == 8:
            model_x = "KernelRidge"
        elif x == 9:
            model_x = "XGBRegressor"
        elif x == 10:
            model_x = "LGBMRegressor"

        mse_dict.setdefault(model_x, []).append(mse)
        mae_dict.setdefault(model_x, []).append(mae)
        r2_dict.setdefault(model_x, []).append(r2)
    x += 1


# calculate the average of each list
mse_averages = []
mae_averages = []
r2_averages = []
for lst in mse_data:
    mse_avg = sum(lst) / len(lst)
    mse_averages.append(mse_avg)

for lst in mae_data:
    mae_avg = sum(lst) / len(lst)
    mae_averages.append(mae_avg)
    
for lst in r2_data:
    r2_avg = sum(lst) / len(lst)
    r2_averages.append(r2_avg)
    

print(mse_averages)
print(mae_averages)
print(r2_averages)