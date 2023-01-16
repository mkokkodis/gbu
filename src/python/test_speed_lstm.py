
############################################
##### Author: Marios Kokkodis            ###
##### email: marios.kokkodis@gmail.com   ###
############################################
##### >>  Python notes: tested on PY38   ###
############################################

import pandas as pd
import numpy as np
import time
import argparse
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.utils import np_utils
import tensorflow
from custom_util_functions import  split_sequences, get_auc_per_class, print_border_line
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore', 'Solver terminated early.*')

random_seed = 1234
np.random.seed(random_seed)

text = 'This script test the speed of the lstm'
parser = argparse.ArgumentParser(description=text)

###########################################################
## LSTM Hyperparameters      ###############################
###########################################################

parser.add_argument("--units", "-u", type=int, default=-1,
                        help="The dimensionality of the output space of the LSTM")

parser.add_argument("--batch_size", "-b", type=int, default=32,
                        help="The number of samples for gradient update. ")
parser.add_argument("--epochs", "-e", type=int, default=10,
                        help="The number of epochs for the LSTM")
parser.add_argument("--stacked", "-C", action='store_true', default=False,
                        help="stacked LSTM")

###########################################################
## Parse arguments from the command line  #################
###########################################################
args = parser.parse_args()
globals().update(vars(args))


###########################################################
# ---- End of arguments initialization -------------------#
###########################################################

###########################################################
# ---- LSTM Functions   ----------------------------------#
###########################################################



def lstm_predict(df_test_x, df_test_y,model,n_steps=55):
    X_test, _ = split_sequences(df_test_x, df_test_y, n_steps=n_steps)
    preds = model.predict(X_test, verbose=0)
    return preds


def run_lstm(df_train_x, df_train_y,df_test_x, df_test_y,   epochs, batch_size, neurons, stacked=False,n_steps=55):
    np.random.seed(1234)
    tensorflow.random.set_seed(1234)
    X, y = split_sequences(df_train_x,df_train_y, n_steps)
    y = np_utils.to_categorical(y)
    n_features = X.shape[2]
    if neurons == -1: neurons = int(0.67 * (n_features + n_steps))
    model = Sequential()
    if stacked:
        model.add(LSTM(units=neurons, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
        model.add(LSTM(units=int(neurons / 2), activation='relu', input_shape=(n_steps, n_features)))
    else:
        model.add(LSTM(units=neurons, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit(X, y, epochs=epochs, verbose=0, batch_size=batch_size)

    X_test, _ = split_sequences(df_test_x, df_test_y, n_steps=n_steps)
    preds = model.predict(X_test, verbose=0)

    return model,preds



emission_variables = [ 'curRating','numberOfReviewsRest',
                'cheap','affordable','expensive','qualityOfRestaurantsVisited','focalUserReviews']
transition_variables = ['focalUserReviews']
transition_index = transition_variables.index('focalUserReviews')
fold = 9
dataset = 'restaurant_ncv.csv'
employer_id = 'user'
task_id = 'choicesetId'
application_id = 'application'
d = pd.read_csv("../../data/"+dataset)
d = d.sort_values([employer_id, 'daysTrend', task_id])
pipe = Pipeline([('imputer', SimpleImputer(missing_values=np.nan, strategy="median",add_indicator=False)),
                 ('scaler',MinMaxScaler())])
train = d[d['set_annotation'].str.contains(pat =('train_'+str(fold))+',|train_'+str(fold)+"$", regex=True)==True].copy()
validation = d[d['set_annotation'].str.contains(pat =('validation_'+str(fold))+',|validation_'+str(fold)+"$",
                                                regex=True)==True].copy()
trainedPipeline = pipe.fit(train[emission_variables])
trainTransformed = pd.DataFrame(data=trainedPipeline.transform(train[emission_variables]),
             index = None,  columns=emission_variables)
validation_transformed = pd.DataFrame(data=trainedPipeline.transform(validation[emission_variables]),
             index = None,  columns=emission_variables)
trainTransformed['set_annotation'] = 'train'
validation_transformed['set_annotation'] = 'validation'
train_y = train['Y_it']
validation_y = validation['Y_it']

# %%
if __name__ == '__main__':
    start_time = time.time()
    print_border_line()
    predictive_variables = transition_variables + emission_variables
    _, predictions = run_lstm(trainTransformed[predictive_variables], train_y,
                              validation_transformed[predictive_variables], validation_y,
                                  epochs, batch_size, units, stacked)
    aucs = get_auc_per_class(validation_y, predictions)
    print_border_line()
    print("\t(Final AUC score on Hire-positive:", round(aucs[2], 3), ")")
    print_border_line()
    print("--- %.3f seconds ---" % (time.time() - start_time))


