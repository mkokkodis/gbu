

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
from hmm_gbu import HMM
from custom_util_functions import  prepare_hmm_matrix, get_auc_per_class,print_border_line

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

random_seed = 1234
np.random.seed(random_seed)
text = 'This script test the speed of the HMM'
parser = argparse.ArgumentParser(description=text)
parser.add_argument("--n_cores", "-c", help="Number of cores",type = int,default=1)
args = parser.parse_args()
globals().update(vars(args))

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
transition_application_employer_train = prepare_hmm_matrix(trainTransformed[transition_variables],
                                                               train[application_id],
                                                               train[employer_id])
transition_application_employer_validation = prepare_hmm_matrix(validation_transformed[transition_variables],
                                                              validation[application_id],
                                                              validation[employer_id])

Z_timelines = trainTransformed[emission_variables].to_numpy()
# %%
if __name__ == '__main__':

    start_time = time.time()
    hmm = HMM(transition_application_employer_train,
              transition_application_employer_validation,transition_index,
              verbal=True, iterations=200,
              random_seed=1234, n_cores=n_cores, solver="COBYLA", states=3)
    r = hmm.fit(Z_timelines, train_y.to_numpy())
    print("Minimized ll:", round(r.optimization_result.fun))
    probs = r.predict_proba(validation_transformed[emission_variables].to_numpy(),
                            validation_y.to_numpy())
    test_y = probs['truth']
    test_predictions = probs[probs.columns.difference(["truth", "application"])].to_numpy()
    aucs = get_auc_per_class(validation_y, test_predictions)
    test_auc_positive = round(aucs[2], 5)
    print_border_line()
    print("\tFinal AUC score on Hire-positive:", test_auc_positive)
    print_border_line()
    print("--- %.3f seconds ---" % (time.time() - start_time))
