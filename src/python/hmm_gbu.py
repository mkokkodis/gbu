
############################################
##### Author: Marios Kokkodis            ###
##### email: marios.kokkodis@gmail.com   ###
############################################
##### >>  Python notes: tested on PY38   ###
############################################


from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from scipy.optimize import minimize
from custom_util_functions import print_border_line
import hmm_functions
import random
# %% HMM functions loading:
from importlib import reload
reload(hmm_functions)
from hmm_functions import transform_input_to_HMM_timelines, createPriors, minimize_log_likelihood, \
    get_predictions
from sklearn.utils import check_random_state



class HMM(ClassifierMixin, BaseEstimator):


    def __init__(self, transition_application_employer_train, transition_application_employer_test,
                 transition_index,
                  random_seed=1234,
                 transition_model='multinomial_constrained',
                 states=3, tol=0.0001,
                 iterations=1000, solver='COBYLA', verbal=True, n_cores=1):

        """
           ----------
           params :
                solver:  'COBYLA','L-BFGS-B',  'BFGS',...
           Attributes
           ----------
           X_ : ndarray, shape (n_samples, n_features)
               The input passed during :meth:`fit`.
           y_ : ndarray, shape (n_samples,)
               The labels passed during :meth:`fit`.
           classes_ : ndarray, shape (n_classes,)
               The classes seen at :meth:`fit`.
        """
        self.random_seed = random_seed
        self.random_state = check_random_state(random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        self.transition_model = transition_model  # 'multinomial_constrained'#
        self.transition_index = transition_index
        self.transition_application_employer_train = transition_application_employer_train
        self.transition_application_employer_test = transition_application_employer_test
        self.states = states if isinstance(states, list) else [state for state in range(states)]
        self.tol = tol
        self.iterations = iterations
        self.solver = solver
        self.verbal = verbal
        self.n_cores = n_cores
        self.variables_in_X = -1
        self.variables_in_Z = -1
        self.sequences_X = None  # transitions
        self.sequences_Z = None  # emissions
        self.sequences_O = None  # outcomes, y
        self.theta_prior = None  # initial
        self.theta_star = None  # optimized
        self.optimization_result = None
        self.prob_predictions = None
        self.emission_distro_pars = None
        if self.verbal:
            print_border_line()
            print("Note: The HMM assumes ranked timelines as inputs")
            print_border_line()


    def update_test_transition(self, transition_application_employer_test):
        """

        Parameters
        ----------
        transition_application_employer_test

        Returns
        -------

            Allows model application to different test sets

        """
        self.transition_application_employer_test = transition_application_employer_test

    def get_params(self, deep=True):

        return dict(transition_model=self.transition_model,
                    random_seed=self.random_seed,
                    states=self.states, tol=self.tol, iterations=self.iterations,
                    solver=self.solver, verbal=self.verbal, n_cores=self.n_cores,
                    transition_index=self.transition_index,
                    transition_application_employer_train = self. transition_application_employer_train,
                    transition_application_employer_test = self.transition_application_employer_test)

    def fit(self, X, y):


        np.random.seed(self.random_seed)
        if self.verbal: print("fit() was called...")
        # Sorting classes according to API documentation: https://scikit-learn.org/stable/developers/develop.html
        self.classes_, y = np.unique(y, return_inverse=True)
        self.X_ = X
        self.y_ = y
        self.emission_distro_pars = len(np.unique(y)) - 1
        (self.sequences_X, self.sequences_Z, self.sequences_O, self.applications) = transform_input_to_HMM_timelines(self.X_, self.y_,
                                                                                                  self.transition_application_employer_train,
                                                                                                  self.verbal,
                                                                                                  self.n_cores)
        self.variables_in_Z = self.sequences_Z[0].shape[1] if self.n_cores == 1 else self.sequences_Z[0][0].shape[1]
        self.variables_in_X = self.sequences_X[0].shape[1] if self.n_cores == 1 else self.sequences_X[0][0].shape[1]
        # initialize parameters
        (self.theta_prior, transitionParameters, emissionParameters, pi) = createPriors(self.variables_in_X,
                                                                                        self.variables_in_Z,
                                                                                        self.states,
                                                                                        self.transition_model,
                                                                                        self.emission_distro_pars)
        if self.verbal:
            print('Total number of parameters to be estimated:', len(self.theta_prior))
        options = {'maxiter': self.iterations, 'disp': False}

        self.optimization_result = minimize(minimize_log_likelihood, self.theta_prior, args=(self.sequences_X,
                                                                                             self.sequences_Z,
                                                                                             self.sequences_O,
                                                                                             self.verbal,
                                                                                             self.variables_in_Z,
                                                                                             self.variables_in_X,
                                                                                             self.states,
                                                                                             self.emission_distro_pars,
                                                                                             self.transition_model,
                                                                                             self.transition_index,
                                                                                             self.n_cores),
                                            method=self.solver,
                                            options=options, tol=self.tol)

        self.theta_star = self.optimization_result.x
        return self


    def decision_function(self, X, y, thr=0.5):
        """

        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
           The target values. An array of int.
        Returns
        -------
        predicted_labels : object
            The predicted labels, by default with threshold at 0.5

        Implemented only for binary classification -- required by the feature selection process
        For more info see here: https://scikit-learn.org/stable/glossary.html#term-predict_proba
        """

        if self.verbal: print("**** decision_function was called ****")
        if self.prob_predictions is None:
            self.prob_predictions = self.predict_proba(X, y)
        predicted_labels = [1 if x >= thr else 0 for x in self.prob_predictions['label_1']]
        return list(predicted_labels)

    def score(self, X, y):
        return self.decision_function(X, y)

    def predict(self, X):
        return self.decision_function(X)

    def predict_proba(self, X, y):
        if self.verbal: print("predict_proba() was called...")
        np.random.seed(self.random_seed)
        _, y = np.unique(y, return_inverse=True)  # same as in fit
        data_to_predict_X = np.concatenate((self.X_, X), axis=0)
        data_to_predict_Y = np.concatenate((self.y_, y), axis=0)
        transition_matrices = np.concatenate((self.transition_application_employer_train, self.transition_application_employer_test), axis=0)
        applications_in_test_set = self.transition_application_employer_test[:,self.transition_application_employer_test.shape[1]-2]
        #deterministic_flag defines whether or not transitions will happen on a threshold = 0.5, determinitistically.
        deterministicFlag = True
        (self.sequences_X, self.sequences_Z, self.sequences_O, self.applications) = transform_input_to_HMM_timelines(data_to_predict_X,
                                                                                                  data_to_predict_Y,
                                                                                                  transition_matrices,
                                                                                                  self.verbal,
                                                                                                  self.n_cores)

        # even in binary, return 2D array, membership probability for each class : https://scikit-learn.org/stable/glossary.html#term-predict_proba
        self.prob_predictions = get_predictions(self.theta_star, self.sequences_X, self.sequences_Z, self.sequences_O,
                                                self.applications,
                                                self.variables_in_Z, self.variables_in_X, self.states,
                                                self.emission_distro_pars, self.classes_,
                                                self.transition_model, self.transition_index,deterministicFlag)
        return self.prob_predictions[self.prob_predictions['application'].isin(applications_in_test_set)].copy()
