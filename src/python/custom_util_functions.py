
############################################
##### Author: Marios Kokkodis            ###
##### email: marios.kokkodis@gmail.com   ###
############################################
##### >>  Python notes: tested on PY38   ###
############################################


import sys
import xgboost as xgb
import pandas as pd
from mlxtend.evaluate import PredefinedHoldoutSplit
from sklearn import linear_model, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.metrics import roc_auc_score
import numpy as np




def print_border_line():
    print("-----------------------------------------------------")

def do_feature_selection(model, trainAndValidation, trainAndValidation_y, minFeatures, maxFeatures, mainFeatures,
                         fixed_features, focal_class):
    validation_indices = trainAndValidation[trainAndValidation.set_annotation == 'validation'].index
    validSet = PredefinedHoldoutSplit(validation_indices)
    trainAndValidation_y = np.where(trainAndValidation_y == focal_class, 1, 0)
    X, y = trainAndValidation[mainFeatures], trainAndValidation_y
    results = {}
    for totFeatures in range(minFeatures, maxFeatures):
        # Build step forward feature selection
        curSFS = sfs(model, k_features=totFeatures, forward=True,
                     verbose=0,
                     # If 0, no output,
                     # if 1 number of features in current set,
                     # if 2 detailed logging including timestamp and cv scores at step.
                     scoring=auc_scorer,  # 'roc_auc',
                     cv=validSet, n_jobs=1,
                     fixed_features=fixed_features)
        curSFS = curSFS.fit(X, y)
        feat_cols = list(curSFS.k_feature_idx_)
        sel = []
        for i in feat_cols:
            sel.append(mainFeatures[i])
        key = ",".join(sorted(sel))
        if key in results: print("error")
        results[key] = curSFS.k_score_
    return results


def run_logistic(trainX, trainY, testX):
    lr = linear_model.LogisticRegression(penalty='l2', solver="lbfgs",  multi_class='ovr')
    lr.fit(trainX, trainY)
    predictions = lr.predict_proba(testX)
    return lr, predictions

def run_forest(train_X, train_y, test_X, test_y, n_estimators,  max_depth):
    train_X = train_X.astype('float32')
    train_y = train_y.astype('float32')
    test_X = test_X.astype('float32')
    # The number of features to consider when looking for the best split: If None, then max_features = n_features:
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    clf = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,
                                 max_features=None,bootstrap=False,
                                 random_state=1234)
    clf.fit(train_X, train_y)
    preds = clf.predict_proba(test_X)
    return clf, preds


def run_xg(train_X, train_y, test_X, test_y, n_estimators, max_depth, learning_rate, subsample):
    train_X = train_X.astype('float32')
    train_y = train_y.astype('float32')
    test_X = test_X.astype('float32')
    xg = xgb.XGBClassifier(n_estimators=n_estimators,
                           max_depth=max_depth,
                           learning_rate=learning_rate,
                           subsample=subsample, random_state=1234, use_label_encoder=False,
                           eval_metric='logloss')
    xg.fit(train_X, train_y)
    preds = xg.predict_proba(test_X)
    return xg, preds


def run_svm(train_X, train_y, test_X):
    train_y = train_y.astype('int')
    clf = svm.SVC(decision_function_shape='ovo', probability=True, kernel='linear',
                   random_state=1234)
    clf.fit(train_X, train_y)
    predictions = clf.predict_proba(test_X)
    return clf, predictions




def auc_scorer(model, X, y_true):
    if 'HMM' in str(type(model)) or 'lstm' in str(type(model)):
        y_pred_proba = model.predict_proba(X, y_true)
    else:
        y_pred_proba = model.predict_proba(X)
    if isinstance(y_pred_proba, (np.ndarray, np.generic)):
        y_pred_proba = y_pred_proba[:, 1]
    else:
        y_pred_proba = y_pred_proba['label_1']
    return roc_auc_score(y_true, y_pred_proba)


def prepare_hmm_matrix(transition_df, application_column, employer_column):
    tmp = transition_df.copy()
    tmp['application_id'] = list(application_column)
    tmp['employer_id'] = list(employer_column)
    return tmp.to_numpy(copy=True)


def get_ranking_performance(focalDf, step=50):
    sortingFeatures = ['label_2', 'application']
    ascendingList = [True, True]
    focalDf = focalDf.sort_values(sortingFeatures, ascending=ascendingList)
    thrL = 0
    thrs = [i / 100.0 for i in range(step, 100, step)]
    rates = []
    curPos = 0
    for thr in thrs:
        denom = step * len(focalDf)
        thrRow = int(thr * len(focalDf))
        pos = focalDf[thrL:thrRow]['hire_positive_truth'].sum()
        thrL = thrRow
        if pos == 0:
            pos = 1
        curPos += pos
        rates.append(pos / float(denom))
    denom = (1 - thr) * len(focalDf)
    pos = focalDf[thrL:]['hire_positive_truth'].sum()
    curPos += pos
    #smoothing:
    if pos == 0:
        pos = 1
        denom += 1
    rates.append(pos / float(denom))
    thrs.append(1)
    preds = pd.DataFrame(thrs)
    preds['percentile'] = thrs
    preds['rate'] = rates
    return preds


def get_within_opening_perf(focalDf, opTotalApps, top_n_thr):
    sortingFeatures = ['label_2', 'label_1', 'label_0']
    ascendingList = [False, True, True]
    focalDf = focalDf.sort_values(sortingFeatures, ascending=ascendingList)
    withinOpRank = []
    opIndex = {}
    totalApps = []
    for ind, row in focalDf.iterrows():
        o = row['task_id']
        if o not in opIndex: opIndex[o] = 1
        withinOpRank.append(opIndex[o])
        opIndex[o] += 1
        totalApps.append(opTotalApps[int(o)])
    focalDf['withinOpRank'] = withinOpRank
    focalDf['totalApps'] = totalApps
    # how much better the algorithms suggestions are
    # compared to those that have not been suggested.
    nRes = {'topVsBottomSR': {}}
    for ed in range(0, 1):
        nRes['topVsBottomSR'][ed] = {}
        curDf = focalDf[focalDf['employer_total_tasks_so_far'] >= ed].copy()
        for curN in range(1, top_n_thr+1):
            posSuggested = curDf[curDf['withinOpRank'] <= curN]['hire_positive_truth'].sum()
            totalSuggested = len(curDf[curDf['withinOpRank'] <= curN])
            posNotSuggested = curDf[curDf['withinOpRank'] > curN]['hire_positive_truth'].sum()
            totalNotSuggested = len(curDf[curDf['withinOpRank'] > curN])
            # smoothing:
            if posSuggested == 0:
                posSuggested += 1
                totalSuggested += 1
            rateSuggested = posSuggested / float(totalSuggested)
            rateNotSuggested = posNotSuggested / float(totalNotSuggested)
            nRes['topVsBottomSR'][ed][curN] = rateSuggested / rateNotSuggested
            if np.isnan(nRes['topVsBottomSR'][ed][curN]):
                print(curN, rateSuggested)
                sys.exit("nan")
    return nRes


def get_model_key(algorithm, key_dict):
    if 'hmm' in algorithm:
        model_key = "::".join(
            [str(key_dict[hmm_arg]) for hmm_arg in ['fold','states']])
    elif algorithm in ['logit']:
        model_key = "::".join([str(key_dict[logit_arg]) for logit_arg in ['fold']])
    elif algorithm in ['svm']:
        model_key = "::".join([str(key_dict[svm_arg]) for svm_arg in ['fold']])
    elif algorithm == 'xg':
        model_key = "::".join(
            [str(key_dict[xg_arg]) for xg_arg in ['fold', 'n_estimators', 'subsample',  'max_depth']])
    elif algorithm == 'rf':
        model_key = "::".join(
            [str(key_dict[rf_arg]) for rf_arg in ['fold', 'max_depth',  'n_estimators']])
    return model_key




def write_best_validation_models(df, algorithm, fout):
    l1 = []
    for fold in range(df.fold.max() + 1):
        tmp_df = df[df.fold == fold].copy()
        if len(tmp_df) == 0: continue
        tmp_df.reset_index(drop=True)
        tmp_df = tmp_df.sort_values('validation_auc_positive', ascending=[False])
        row = tmp_df[0:1]
        val = row['validation_auc_positive'].values[0]
        model_key = row['model_key'].values[0]
        fout.write(",".join([str(i) for i in [fold,  algorithm, model_key]]) + "\n")
        l1.append(val)




def split_sequences(df_train_x, df_train_y, n_steps):
    if not isinstance(df_train_x, (np.ndarray, np.generic)):
        sequences = df_train_x.copy()
        sequences['y'] = list(df_train_y)
        sequences = sequences.to_numpy()
    else:
        sequences = np.insert(df_train_x, df_train_x.shape[1], df_train_y, axis=1)
    padding = np.zeros((n_steps - 1, sequences.shape[1]))
    sequences = np.concatenate((padding, sequences), axis=0)
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
# pureObs is a list of observations.
# pureProbs is a list of lists of all the probs.
def get_auc_per_class(pureObs, pureProbs, curclasses=[0, 1, 2]):
    totalClasses = len(np.unique(pureObs))
    y = label_binarize(pureObs, classes=curclasses)
    na = np.array(pureProbs)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(totalClasses):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], na[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return roc_auc
