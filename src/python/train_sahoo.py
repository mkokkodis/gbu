############################################
##### Author: Marios Kokkodis            ###
##### email: marios.kokkodis@gmail.com   ###
############################################
##### >>  Python notes: tested on PY38   ###
############################################


import sys
from scipy.optimize import minimize
from scipy.special import softmax
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from custom_util_functions import print_border_line
import time
import argparse
from hmm_functions import createPriors, minimize_log_likelihood_sahoo, get_stochastic_draw_from_probs, init_params
from joblib import dump, load

###########################################################
## Options                            #####################
###########################################################
text = 'This script trains the HMM-sahoo model presented in the Appendix of MS-INS-19-00471'
parser = argparse.ArgumentParser(description=text)
# Add long and short argument
parser.add_argument("--fold", "-f", help="the fold to train, validate and test.", type=int, default=0)
parser.add_argument("--iterations", "-i", help="the number of iterations to search (for hmm).",
                    default=15000, type=int)
parser.add_argument("--states", "-s", help="number of hmmm sates.", default=2, type=int)
parser.add_argument("--output", "-o", action='store_true', default=False,
                    help="number of hmmm sates.")
parser.add_argument("--to_predict", "-P", action='store_true', default=False,
                    help="uses trained model to provide raw predictions."
                         " The model is defined by the model_key.")

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
states = [j for j in range(states)]
number_of_items = 58
data_file = "restaurant_ncv.csv"
employer_id = 'user'
task_id = 'choicesetId'
application_id = 'application'
d = pd.read_csv("../../data/" + data_file)
d['item_sequential'] = d['item_sequential'] - 1  # so that it starts from zero
d = d.sort_values([employer_id, 'daysTrend', task_id])
train = d[d['set_annotation'].str.contains(pat=('train_' + str(fold)) + ',|train_' + str(fold) + "$",
                                           regex=True) == True].copy()
validation = d[d['set_annotation'].str.contains(pat=('validation_' + str(fold)) + ',|validation_' + str(fold) + "$",
                                                regex=True) == True].copy()
test = d[d['set_annotation'].str.contains(pat=('test_' + str(fold)) + ',|test_' + str(fold) + "$",
                                          regex=True) == True].copy()
trainAndValidation = pd.concat([train, validation])
train_y = train['item_sequential']
test_y = test['item_sequential']
validation_y = validation['item_sequential']
trainAndValidation_y = pd.concat([train_y, validation_y])
trainAndValidation.reset_index(inplace=True, drop=True)
trainAndValidation_y.reset_index(inplace=True, drop=True)
transition_model = 'multinomial'
emission_distro_pars = number_of_items - 1  # items -1


# create timelines for sahoo
# transitions, emissions only constant term:
def get_hmm_timelines(d_X, d_y):
    y = d_y
    X_timelines = []
    X = d_X[['user', 'application']].to_numpy()
    X = np.insert(X, X.shape[1], y, axis=1)
    outcome_index = X.shape[1] - 1
    employer_column_index = 0
    application_column_index = 1
    applications_timelines = []
    O = []
    Z_timelines = []
    for i in np.unique(X[:, employer_column_index]):
        curTimeline = X[X[:, employer_column_index] == i, :]
        applications_timelines.append(curTimeline[:, application_column_index])
        # get timeline's transitions (X), emissions (Z) and outcomes.
        X_timeline = np.ones((len(curTimeline), 1))
        X_timelines.append(X_timeline)
        Z_timeline = np.ones((len(curTimeline), 1))
        Z_timelines.append(Z_timeline)
        O_timeline = curTimeline[:, outcome_index]
        O.append(O_timeline.astype(np.int32))

    return X_timelines, Z_timelines, O, applications_timelines


def run_hmm(cur_train_X, cur_train_y, cur_test_X, cur_test_y):
    X_timelines, Z_timelines, O, applications_timelines = get_hmm_timelines(cur_train_X, cur_train_y)
    variables_in_Z = Z_timelines[0].shape[1]
    variables_in_X = X_timelines[0].shape[1]
    verbal = False
    (theta_prior, transitionParameters, emissionParameters, pi) = createPriors(variables_in_X,
                                                                               variables_in_Z,
                                                                               states,
                                                                               transition_model,
                                                                               emission_distro_pars,
                                                                               verbal)
    tol = 0.001
    print('Total number of parameters to be estimated:', len(theta_prior))
    options = {'maxiter': iterations, 'disp': False}
    optimization_result = minimize(minimize_log_likelihood_sahoo,
                                   theta_prior, args=(X_timelines,
                                                      Z_timelines,
                                                      O,
                                                      verbal,
                                                      variables_in_Z,
                                                      variables_in_X,
                                                      states,
                                                      emission_distro_pars,
                                                      transition_model,
                                                      ),
                                   method='COBYLA',
                                   options=options, tol=tol)

    theta_star = optimization_result.x

    return get_predictions(theta_star, cur_test_X, cur_test_y), optimization_result


def get_predictions(theta_star, cur_test_X, cur_test_y):
    # %%
    X_timelines, Z_timelines, O, applications_timelines = get_hmm_timelines(cur_test_X, cur_test_y)
    variables_in_Z = Z_timelines[0].shape[1]
    variables_in_X = X_timelines[0].shape[1]
    (pi, gammaCoeffsForZ, betaCoeffsforX, sigmaPars) = init_params(theta_star, variables_in_Z, states,
                                                                   emission_distro_pars,
                                                 variables_in_X,
                                                 transition_model)

    zCoeffsMat = []
    for state in states:
        zCoeffsMat.append([])
        for symbol in range(emission_distro_pars + 1):
            zCoeffsMat[state].append(gammaCoeffsForZ[state][symbol])
    zCoeffsMat = np.array(zCoeffsMat).astype(np.float32)

    xCoeffsMat = []
    for stateFrom in states:
        xCoeffsMat.append([])
        for stateTO in states:
            xCoeffsMat[stateFrom].append(betaCoeffsforX[stateFrom][stateTO])
    xCoeffsMat = np.array(xCoeffsMat).astype(np.float32)

    actualOutcomes = {}
    for ind, row in cur_test_X.iterrows():
        actualOutcomes[str(row['application'])] = row['Y_it']

    probs = []
    truth = []
    application = []
    deterministicFlag = True
    iterationZDict = {}
    iterationXDict = {}
    T = xCoeffsMat
    T = np.exp(T - np.logaddexp.reduce(T, axis=1, keepdims=True))
    for i in range(len(X_timelines)):
        client_X = X_timelines[i]
        client_O = O[i]
        client_Z = Z_timelines[i]
        apps = applications_timelines[i]
        curState = np.argmax(pi) if deterministicFlag else get_stochastic_draw_from_probs(pi)
        for j in range(len(client_O)):
            for key in client_Z[j]:
                if (key in iterationZDict) and (curState in iterationZDict[key]):
                    sampleProbs = iterationZDict[key][curState]
                else:
                    sampleProbs = [np.dot(zCoeffsMat[curState][symbol], client_Z[j]) for symbol in
                                   range(number_of_items)]
                    sampleProbs = softmax(sampleProbs)

                    if key not in iterationZDict: iterationZDict[key] = {}
                    iterationZDict[key][curState] = sampleProbs

            for key in client_X[j]:
                if (key in iterationXDict) and (curState in iterationXDict[key]):
                    newState = iterationXDict[key][curState]
                else:
                    newState = np.argmax(T[curState])
                    if key not in iterationXDict: iterationXDict[key] = {}
                    iterationXDict[key][curState] = newState

            cur_app = str(apps[j])
            curState = newState
            probs.append([1 - sampleProbs[int(client_O[j])], sampleProbs[int(client_O[j])]])
            truth.append(actualOutcomes[cur_app])
            application.append(int(cur_app))

    predictions = pd.DataFrame(probs, columns=['label_' + str(i) for i in [0, 1]])
    predictions['truth'] = list(truth)
    predictions['application'] = list(application)
    predictions['hire_positive_truth'] = np.where(predictions['truth'] == 2, 1, 0)
    curAuc = roc_auc_score(predictions['hire_positive_truth'], predictions['label_1'])
    return predictions, curAuc


start_time = time.time()
np.random.seed(1234)
key_dict = vars(args)
model_key = "_".join([str(key_dict[hmm_arg]) for hmm_arg in ['fold', 'states']])
###########################################################
# ---- Trained model predictions -------------------------#
###########################################################

if to_predict:
    optimization_result = load("../../data/trained_models/sahoo/" + model_key)
    theta_star = optimization_result.x
    probs, auc_test = get_predictions(theta_star, test, test_y)
    probs['fold'] = fold
    probs['algorithm'] = 'sahoo'
    probs['hire_positive_truth'] = np.where(probs['truth'] == 2, 1, 0)
    probs['hire_negative_truth'] = np.where(probs['truth'] == 1, 1, 0)
    probs = pd.merge(probs, d[[application_id, 'employer_total_tasks_so_far', task_id]],
                     left_on='application', right_on=application_id, how='inner')
    probs.to_csv("../../data/raw_predictions/sahoo" + str(fold) + ".csv", index=False)
    print("Raw predictions for sahoo fold", fold, "posted in: ../../data/raw_predictions/")
    sys.exit(0)

(_, validation_auc_positive), _ = run_hmm(train, train_y, validation, validation_y)
validation_auc_positive = round(validation_auc_positive, 5)
print("\t(AUC score on validation:", round(validation_auc_positive, 5), ")")
print_border_line()
(_, _), optimization_result = run_hmm(trainAndValidation, trainAndValidation_y, test, test_y)
print("Minimized ll:", round(optimization_result.fun))
print("Convergence:", optimization_result.success)
print("--- %.3f seconds ---" % (time.time() - start_time))

if output:
    fout = open("../../data/results_per_model/sahoo/" + model_key, "w")
    strout = model_key
    strout += "," + ",".join([str(i) for i in [fold, 'sahoo', 'nan',
                                               validation_auc_positive]])
    fout.write(strout + "\n")
    fout.close()
    dump(optimization_result, "../../data/trained_models/sahoo/" + model_key)
