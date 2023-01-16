############################################
##### Author: Marios Kokkodis            ###
##### email: marios.kokkodis@gmail.com   ###
############################################
##### >>  Python notes: tested on PY38   ###
############################################

import sys
import pandas as pd
import numpy as np
from importlib import reload
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import custom_util_functions
import time
from hmm_gbu import HMM
import argparse

reload(custom_util_functions)
from custom_util_functions import do_feature_selection, run_logistic, get_auc_per_class, print_border_line
from custom_util_functions import prepare_hmm_matrix, run_forest, run_xg, run_svm, get_model_key
from sklearn.pipeline import Pipeline
from sklearn import linear_model, svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore', 'Solver terminated early.*')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
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
## Global options for all models      #####################
###########################################################
text = 'This script trains the multiple baselines and the custom HMM presented in the Appendix of MS-INS-19-00471'
parser = argparse.ArgumentParser(description=text)
parser.add_argument("--fold", "-f", help="The fold to train, validate, and test.", type=int, default=0)
parser.add_argument("--algorithm", "-a", help="The algorithm to use.", default="logit")
parser.add_argument("--output", "-o", action='store_true', default=False,
                    help="Writes out the results---posted in the `trained_models` and `results_per_model` directories.")
parser.add_argument("--to_predict", "-P", action='store_true', default=False,
                    help="Uses trained models (stored in the `trained_models` directory) to provide raw predictions."
                         "The model is defined by the model_key.")
parser.add_argument("--min_num_features_to_search", "-F", default=len(standard_vars) + 1, type=int,
                    help="Minimum number of  features to explore for selection.")
parser.add_argument("--max_num_features_to_search", "-M", type=int, default=len(predictive_variables) + 1,
                    help="Maximum number of features to explore for selection.")
###########################################################
## HMM Hyperparameters      ###############################
###########################################################
parser.add_argument("--iterations_feature_selection", "-i",
                    help="The number of maximum iterations for feature selection.",
                    default=1000, type=int)
parser.add_argument("--iterations_train", "-I", help="The number of maximum iterations for training.",
                    default=15000, type=int)
parser.add_argument("--states", "-s", help="the number of hmmm sates.", default=3, type=int)
parser.add_argument("--n_tran_vars", "-n", help="The number of transition variables.",
                    default=1, type=int)
###########################################################
## xgboost and Random forest Hyperparameters  #############
###########################################################
parser.add_argument("--max_depth", "-d", type=int, default=10,
                    help="maximum depth for xgboost and random forest")
parser.add_argument("--n_estimators", "-e", type=int, default=100,
                    help="estimators for xgboost and random forest")
parser.add_argument("--subsample", "-S", type=float, default=1,
                    help="subsample option for xgboost")
parser.add_argument("--learning_rate", "-l", type=float, default=0.01,
                    help="learning rate option for xgboost")
###########################################################
## Parse arguments from the command line  #################
###########################################################
args = parser.parse_args()
globals().update(vars(args))
###########################################################
# ---- End of arguments initialization -------------------#
###########################################################


###########################################################
# ---- Dataset creation ----------------------------------#
###########################################################
model_key = get_model_key(algorithm, vars(args))
transition_variables = [potential_transition_vars[i] for i in range(n_tran_vars)]
transition_index = transition_variables.index('focalUserReviews')
dataset = 'restaurant_ncv.csv'
employer_id = 'user'
task_id = 'choicesetId'
application_id = 'application'
d = pd.read_csv("../../data/" + dataset)
d = d.sort_values([employer_id, 'daysTrend', task_id])
pipe = Pipeline([('imputer', SimpleImputer(missing_values=np.nan, strategy="median", add_indicator=False)),
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
    model_to_predict, final_features = load("../../data/trained_models/" + algorithm + "/" + model_key)
    if 'hmm' in algorithm:
        probs = model_to_predict.predict_proba(test_transformed[final_features].to_numpy(), test_y.to_numpy())
    else:
        probs = model_to_predict.predict_proba(test_transformed[final_features].to_numpy())
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
transition_application_employer_train = prepare_hmm_matrix(trainTransformed[transition_variables],
                                                           train[application_id],
                                                           train[employer_id])
transition_application_employer_validation = prepare_hmm_matrix(validation_transformed[transition_variables],
                                                                validation[application_id],
                                                                validation[employer_id])
if 'logit' in algorithm or iterations_feature_selection == -1:
    model = linear_model.LogisticRegression(penalty='l2', solver="lbfgs")

elif 'hmm' in algorithm:
    model = HMM(transition_application_employer_train, transition_application_employer_validation, transition_index,
                verbal=False, iterations=iterations_feature_selection, states=states)
elif algorithm == 'xg':
    # https: // xgboost.readthedocs.io / en / latest / python / python_api.html
    # use_label_encoder (bool) – (Deprecated) Use the label encoder from scikit-learn to encode the labels.
    # For new code, we recommend that you set this parameter to False.
    # Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic'
    # was changed from 'error' to 'logloss'.Explicitly set eval_metric if you'd like to restore the old behavior.
    model = xgb.XGBClassifier(n_estimators=n_estimators,
                              max_depth=max_depth,
                              learning_rate=learning_rate,
                              subsample=subsample,
                              random_state=1234,
                              use_label_encoder=False, eval_metric='logloss')

elif algorithm == 'rf':
    # https: // scikit - learn.org / stable / modules / generated / sklearn.ensemble.RandomForestClassifier.html
    # max_features: The number of features to consider when looking for the best split: If None, then max_features=n_features.
    model = RandomForestClassifier(n_estimators=n_estimators,
                                   max_depth=max_depth,
                                   max_features=None,
                                   bootstrap=False)

elif 'svm' in algorithm:
    model = svm.SVC(probability=True, kernel='linear', max_iter=iterations_feature_selection)

print_border_line()

selected = set()
if min_num_features_to_search > len(standard_vars) and max_num_features_to_search > min_num_features_to_search:
    for focal_class in [0, 1, 2]:
        cur_results = do_feature_selection(model, train_and_validation, train_and_validation_y,
                                           min_num_features_to_search, max_num_features_to_search,
                                           predictive_variables, fixed_features, focal_class)
        selected.add(max(cur_results, key=cur_results.get))
        print("\tClass: ", focal_class, "| Different sets of features so far:", len(selected), "| AUC:",
              round(max(cur_results.values()), 2))
else:
    selected = [",".join(standard_vars)]
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
    if algorithm == 'logit':
        _, predictions = run_logistic(trainTransformed[fs].to_numpy(), train_y.to_numpy(),
                                      validation_transformed[fs].to_numpy())
        aucs = get_auc_per_class(validation_y, predictions)
    elif 'hmm' in algorithm:
        model = HMM(transition_application_employer_train, transition_application_employer_validation, transition_index,
                    verbal=False, iterations=iterations_train, states=states)
        Z_timelines = trainTransformed[fs].to_numpy()
        r = model.fit(Z_timelines, train_y.to_numpy())
        probs = r.predict_proba(validation_transformed[fs].to_numpy(), validation_y.to_numpy())
        aucs = get_auc_per_class(probs['truth'], probs[probs.columns.difference(["truth", "application"])].to_numpy())
    elif algorithm == 'rf':
        _, predictions = run_forest(trainTransformed[fs], train_y, validation_transformed[fs], validation_y,
                                    n_estimators, max_depth)
        aucs = get_auc_per_class(validation_y, predictions)
    elif algorithm == 'xg':
        _, predictions = run_xg(trainTransformed[fs], train_y, validation_transformed[fs], validation_y,
                                n_estimators, max_depth, learning_rate, subsample)
        aucs = get_auc_per_class(validation_y, predictions)
    elif algorithm == 'svm':
        _, predictions = run_svm(trainTransformed[fs], train_y, validation_transformed[fs])
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
# - Train final model on both validation and train sets - #
###########################################################

L = np.nan
converged = np.nan
print_border_line()
if algorithm == 'logit':
    r, _ = run_logistic(train_and_validation[final_features].to_numpy(),
                        train_and_validation_y.to_numpy(),
                        test_transformed[final_features].to_numpy())
elif 'hmm' in algorithm:
    transition_application_employer_train_val = prepare_hmm_matrix(train_and_validation[transition_variables],
                                                                   list(train[application_id]) + list(
                                                                       validation[application_id]),
                                                                   list(train[employer_id]) + list(
                                                                       validation[employer_id]))

    transition_application_employer_test = prepare_hmm_matrix(test_transformed[transition_variables],
                                                              test[application_id],
                                                              test[employer_id])
    model = HMM(transition_application_employer_train_val, transition_application_employer_test, transition_index,
                verbal=False, iterations=iterations_train,
                states=states)
    r = model.fit(train_and_validation[final_features].to_numpy(), train_and_validation_y.to_numpy())
    L = round(r.optimization_result.fun, 2)
    converged = r.optimization_result.success
    print("\t> minimized ll:", round(r.optimization_result.fun))
    print("\t> convergence:", r.optimization_result.success)

elif algorithm == 'rf':
    r, _ = run_forest(train_and_validation[final_features], train_and_validation_y,
                      test_transformed[final_features], test_y,
                      n_estimators, max_depth)
elif algorithm == 'xg':
    r, _ = run_xg(train_and_validation[final_features], train_and_validation_y,
                  test_transformed[final_features], test_y,
                  n_estimators, max_depth, learning_rate,
                  subsample)
elif algorithm == 'svm':
    r, _ = run_svm(train_and_validation[final_features], train_and_validation_y,
                   test_transformed[final_features])

print_border_line()
print("--- %.3f seconds ---" % (time.time() - start_time))

if output:
    dump((r, final_features), "../../data/trained_models/" + algorithm + "/" + model_key)
    fout = open("../../data/results_per_model/" + algorithm + "/" + model_key, "w")
    strout = model_key
    strout += "," + ",".join([str(i) for i in [fold, algorithm, validation_average_auc,
                                               validation_auc_positive]])
    fout.write(strout + "\n")
    fout.close()
