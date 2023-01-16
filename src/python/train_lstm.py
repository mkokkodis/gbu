############################################
##### Author: Marios Kokkodis            ###
##### email: marios.kokkodis@gmail.com   ###
############################################
##### >>  Python notes: tested on PY36 (server) and PY38 (local)###
############################################

import sys

sys.path.insert(1, '/usr/public/keras/2.3.0t2p3.6gbu/lib/python3.6/site-packages/')
import pandas as pd
import numpy as np
from importlib import reload
import custom_util_functions
import time
import argparse
import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.utils import np_utils
import tensorflow
from lstm_sklearn import lstm_sklearn

reload(custom_util_functions)
from custom_util_functions import do_feature_selection, split_sequences, get_auc_per_class, print_border_line
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore', 'Solver terminated early.*')
from joblib import dump, load

start_time = time.time()

###########################################################
## Predictive variables sets ##############################
###########################################################
potential_transition_vars = ['focalUserReviews']
# those that are not included in the trans
emission_variables = ['curRating', 'numberOfReviewsRest',
                      'cheap', 'affordable', 'expensive', 'qualityOfRestaurantsVisited']
standard_vars = ['qualityOfRestaurantsVisited', 'focalUserReviews', 'curRating', 'numberOfReviewsRest']
predictive_variables = potential_transition_vars + emission_variables
fixed_features = [predictive_variables.index(i) for i in standard_vars]
###########################################################
##  Options                           #####################
###########################################################
text = 'This script trains the LSTM models presented in the Appendix of MS-INS-19-00471'
parser = argparse.ArgumentParser(description=text)
parser.add_argument("--fold", "-f", help="The fold to train, validate, and test.", type=int, default=0)
parser.add_argument("--algorithm", "-a", help="The algorithm to use.", default="lstm")
parser.add_argument("--output", "-o", action='store_true', default=False,
                    help="Writes out the results---posted in the `trained_models` and `results_per_model` directories.")
parser.add_argument("--to_predict", "-P", action='store_true', default=False,
                    help="Uses trained models (stored in the `trained_models` directory) to provide raw predictions."
                         "The model is defined by the model_key.")
###########################################################
## LSTM Hyperparameters      ###############################
###########################################################
parser.add_argument("--units", "-u", type=int, default=-1,
                    help="The dimensionality of the output space of the hidden layer(s) of the LSTM")
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

def lstm_predict(df_test_x, df_test_y, model, n_steps=55):
    X_test, _ = split_sequences(df_test_x, df_test_y, n_steps=n_steps)
    preds = model.predict(X_test, verbose=0)
    return preds

def run_lstm(df_train_x, df_train_y, df_test_x, df_test_y, epochs, batch_size, neurons, stacked=False, n_steps=55):
    np.random.seed(1234)
    tensorflow.random.set_seed(1234)
    X, y = split_sequences(df_train_x, df_train_y, n_steps)
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
    return model, preds


def create_LSTM_model_for_fs(n_steps=55, neurons=10, n_features=-1, stacked=False):
    if neurons == -1: neurons = int(0.67 * (n_features + n_steps))
    model = Sequential()
    if stacked:
        model.add(
            LSTM(units=neurons, activation='relu', return_sequences=True))
        model.add(LSTM(units=int(neurons / 2), activation='relu'))
    else:
        model.add(LSTM(units=neurons, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model


###########################################################
# ---- Dataset creation ----------------------------------#
###########################################################
key_dict = vars(args)
model_key = "_".join(
    [str(key_dict[lstm_arg]) for lstm_arg in ['fold', 'batch_size',
                                              'epochs', 'stacked']])
dataset = 'restaurant_ncv.csv'
employer_id = 'user'
task_id = 'choicesetId'
application_id = 'application'
d = pd.read_csv("../../data/" + dataset)
d = d.sort_values([employer_id, 'daysTrend', task_id])
pipe = Pipeline([('imputer', SimpleImputer(missing_values=np.nan, strategy="median",
                                           add_indicator=False)),
                 ('scaler', MinMaxScaler())])
train = d[d['set_annotation'].str.contains(pat=('train_' + str(fold)) + ',|train_' + str(fold) + "$",
                                           regex=True) == True].copy()
validation = d[d['set_annotation'].str.contains(pat=('validation_' + str(fold)) + ',|validation_' + str(fold) + "$",
                                                regex=True) == True].copy()
test = d[d['set_annotation'].str.contains(pat=('test_' + str(fold)) + ',|test_' + str(fold) + "$",
                                          regex=True) == True].copy()
trainedPipeline = pipe.fit(train[predictive_variables])
trainTransformed = pd.DataFrame(data=trainedPipeline.transform(train[predictive_variables]),
                                index=None, columns=predictive_variables)
validation_transformed = pd.DataFrame(data=trainedPipeline.transform(validation[predictive_variables]),
                                      index=None, columns=predictive_variables)
test_transformed = pd.DataFrame(data=trainedPipeline.transform(test[predictive_variables]),
                                index=None, columns=predictive_variables)
trainTransformed['set_annotation'] = 'train'
validation_transformed['set_annotation'] = 'validation'
train_and_validation = pd.concat([trainTransformed, validation_transformed])
train_y = train['Y_it']
test_y = test['Y_it']
validation_y = validation['Y_it']
train_and_validation_y = pd.concat([train_y, validation_y])
train_and_validation.reset_index(inplace=True, drop=True)
train_and_validation_y.reset_index(inplace=True, drop=True)
###########################################################
# ---- End of dataset creation ---------------------------#
###########################################################


###########################################################
# ---- Trained model predictions -------------------------#
###########################################################
if to_predict:
    model_to_predict = keras.models.load_model("../../data/trained_models/" + algorithm + "/" + model_key + ".h5")
    final_features = load("../../data/trained_models/" + algorithm + "/" + model_key + 'final_features')
    probs = lstm_predict(test_transformed[final_features], test_y, model_to_predict)
    probs = pd.DataFrame(probs, columns=['label_' + str(i) for i in [0, 1, 2]])
    if 'truth' not in probs.columns:
        probs['truth'] = list(test_y)
        probs['application'] = list(test[application_id])
    probs['fold'] = fold
    probs['algorithm'] = algorithm
    probs['hire_positive_truth'] = np.where(probs['truth'] == 2, 1, 0)
    probs['hire_negative_truth'] = np.where(probs['truth'] == 1, 1, 0)
    probs = pd.merge(probs, d[[application_id, 'employer_total_tasks_so_far', 'task_id']],
                     left_on='application', right_on=application_id, how='inner')
    probs.to_csv("../../data/raw_predictions/" + algorithm + str(fold) + ".csv", index=False)
    print("Raw predictions for ", algorithm, " fold", fold, "posted in: ../../data/raw_predictions/")
    sys.exit(0)

###########################################################
# ---- End of Trained model predictions ------------------#
###########################################################


###########################################################
# ---- Feature selection  --------------------------------#
###########################################################
model = lstm_sklearn(neurons=units, epochs=epochs, batch_size=batch_size, stacked=stacked)
selected = set()
print_border_line()
min_features = len(standard_vars) + 1
max_features = len(predictive_variables) + 1
for focal_class in [0, 1, 2]:
    cur_results = do_feature_selection(model, train_and_validation, train_and_validation_y, min_features, max_features,
                                       predictive_variables, fixed_features, focal_class)
    selected.add(max(cur_results, key=cur_results.get))
    print("\tClass: ", focal_class, "| Different sets of features so far:", len(selected), "| AUC:",
          round(max(cur_results.values()), 2))

###########################################################
# ---- End of Feature selection --------------------------#
###########################################################


###########################################################
# ---- Select based on average auc score across labels ---#
###########################################################
print_border_line()
results_three_labels = {}
fsind = 0
for fs in selected:
    fs = fs.split(",")
    fsind += 1
    _, predictions = run_lstm(trainTransformed[fs], train_y, validation_transformed[fs], validation_y,
                              epochs, batch_size, units, stacked)
    aucs = get_auc_per_class(validation_y, predictions)
    average_auc = round(np.mean(list(aucs.values())), 5)
    results_three_labels[",".join(sorted(fs))] = (average_auc, round(aucs[2], 5))
    print("\tAverage AUC on validation for fsind ", fsind, ":", average_auc)
tmp_dict = {k: np.mean(v) for k, v in results_three_labels.items()}
final_features = max(tmp_dict, key=tmp_dict.get)
validation_average_auc, validation_auc_positive = results_three_labels[final_features]
final_features = final_features.split(",")

###########################################################
# ---- End of Validate with three labels------------------#
###########################################################


###########################################################
# ---- Evaluate on test set      -------------------------#
###########################################################

r, _ = run_lstm(train_and_validation[final_features], train_and_validation_y,
                               test_transformed[final_features], test_y, epochs, batch_size, units, stacked)
print_border_line()
print("--- %.3f seconds ---" % (time.time() - start_time))
if output:
    L = converged = np.nan
    r.save("../../data/trained_models/" + algorithm + "/" + model_key + ".h5")
    dump(final_features, "../../data/trained_models/" + algorithm + "/" + model_key + 'final_features')
    fout = open("../../data/results_per_model/" + algorithm + "/" + model_key, "w")
    strout = model_key
    strout += "," + ",".join([str(i) for i in [fold, algorithm, validation_average_auc,
                                               validation_auc_positive]])
    fout.write(strout + "\n")
    fout.close()
