
############################################
##### Author: Marios Kokkodis            ###
##### email: marios.kokkodis@gmail.com   ###
############################################
##### >>  Python notes: tested on PY38   ###
############################################

import sys
from multiprocessing import Process, Queue
import numpy as np
import pandas as pd
import scipy
from scipy.special import softmax
import itertools


# define global variables
z_coeffs_mat: np.ndarray = None
states: np.ndarray = None
pi: np.ndarray = None
x_coeffs_mat: np.ndarray = None
transition_model: str = None
transition_index: int = None
prob_stored_values_dict: dict = None
T_diag: np.ndarray = None
I = None



def get_stochastic_draw_from_probs(vectorProbs):
    r = np.random.random()
    cum = 0
    draw = 0
    for p in vectorProbs:
        cum += p
        if r <= cum: return draw
        draw += 1
    if cum < 0.98:
        print("Error")
        print(vectorProbs)
        print(cum, sum(vectorProbs))
        sys.exit()
    return len(vectorProbs) - 1

def get_emission_probs_from_state(curState, coeffsZ, curZ, emissionDistroPars):
    probs = [np.dot(coeffsZ[curState][symbol], curZ) for symbol in range(emissionDistroPars + 1)]
    return softmax(probs)

def get_next_state(curX, coeffsX, deterministicFlag, curState,
                   transitionModel, states, ):
    if 'multinomial' in transitionModel:
        trans = []
        for stateTo in states:
            trans.append(np.dot(coeffsX[curState][stateTo], curX))
        trans = softmax(trans)
        if not deterministicFlag:
            return get_stochastic_draw_from_probs(trans)
        else:
            return np.argmax(trans)





def minimize_log_likelihood(theta, X, Z, O, verbal, variables_in_Z, variables_in_X, states_to_assign, emission_distro_pars,
                            transition_model_to_assign, transition_index_to_assign, n_cores=1):
    """
    # minimize log likelihood
    # transition_index = featuresX.index('employer_completed_tasks')
    """

    global beta_coeffsfor_x
    global pi, x_coeffs_mat, z_coeffs_mat, states, transition_index, T_diag, transition_model, I, beta_coeffsfor_x, prob_stored_values_dict
    states = states_to_assign
    transition_model = transition_model_to_assign
    I = np.ones(len(states))
    T_diag = np.zeros(shape=(len(states), len(states)))
    np.fill_diagonal(T_diag, 1)
    transition_index = transition_index_to_assign
    prob_stored_values_dict = {}

    (pi, gamma_coeffs_for_z, beta_coeffsfor_x, sigmaPars) = init_params(theta, variables_in_Z,states,  emission_distro_pars, variables_in_X,transition_model)

    tmp_z_c = []
    for state in states:
        tmp_z_c.append([])
        for symbol in range(emission_distro_pars + 1):  # number of symbols for multinomial emissions.
            tmp_z_c[state].append(gamma_coeffs_for_z[state][symbol])
    z_coeffs_mat = np.array(tmp_z_c).astype(np.float32)

    tmp_x_c = []
    for stateFrom in states:
        tmp_x_c.append([])
        for stateTO in states:
            tmp_x_c[stateFrom].append(beta_coeffsfor_x[stateFrom][stateTO])
    x_coeffs_mat = np.array(tmp_x_c).astype(np.float32)

    if n_cores == 1:
        L = get_partial_likelihood(X, Z, O)
        if verbal > 1:
            print("L:", round(L, 1))
        return L


    #Parallelization with multiple cores
    output = Queue()
    processes = []
    for cur_core in range(n_cores):
        processes.append(Process(target=get_likelihood_validation,
        args=(X[cur_core], Z[cur_core], O[cur_core], pi,
                                  x_coeffs_mat, z_coeffs_mat,
                                  transition_model, states, transition_index,
                                  output)))

    for p in processes: p.start()
    for p in processes: p.join()
    L = 0
    for i in range(n_cores):
        partialL = output.get()
        L += partialL
    if verbal > 1:
        print("L1:", round(L, 1))
    return L



def get_partial_likelihood(X, Z, O,output=None):

    partial_L = -sum(map(get_single_timeline_likelihood_map, zip(O, X, Z)))
    if np.isnan(partial_L):
        sys.exit("Fatal error: NaN log-likelihood.")
    if partial_L < 0:
        sys.exit("Fatal error: Negative log-likelihood.")
    if output is not None:
        output.put(partial_L)
    else:
        return partial_L





def minimize_log_likelihood_sahoo(theta, X, Z, O, verbal, variables_in_Z, variables_in_X, states, emission_distro_pars,
                            transition_model):
    """
    Implementation of the Sahoo et al. 2021 many assessment adaptation:
    Sahoo,  Nachiketa,  Param  Vir  Singh,  Tridas  Mukhopadhyay.  2012.
    A  hidden  Markov  model  for  collaborative filtering. MIS Quarterly.

    Adaptation details in the paper's Appendix.

    """
    global beta_coeffsfor_x
    (pi, gammaCoeffsForZ, beta_coeffsfor_x,
      sigmaPars) = init_params(theta, variables_in_Z, states, emission_distro_pars, variables_in_X,
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
            xCoeffsMat[stateFrom].append(beta_coeffsfor_x[stateFrom][stateTO])
    xCoeffsMat = np.array(xCoeffsMat).astype(np.float32)
    L = getPartialLikelihood_sahoo(X, Z, O, pi, xCoeffsMat, zCoeffsMat, transition_model,
                                 states)
    if verbal > 1:
        print("L:", round(L, 1))
    return L




def getPartialLikelihood_sahoo(X, Z, O, pi1, xCoeffsMat1, zCoeffsMat1,
                         transitionModel, states1):
    global pi, x_coeffs_mat, z_coeffs_mat, states, transition_index,  transition_model, I, beta_coeffsfor_x,prob_stored_values_dict
    states = states1
    transition_model = transitionModel
    I = np.ones(len(states))
    prob_stored_values_dict = {}
    pi = pi1
    x_coeffs_mat = xCoeffsMat1
    z_coeffs_mat = zCoeffsMat1

    partial_L = -sum(map(get_single_timeline_likelihood_map_sahoo, zip(O, X, Z)))
    if np.isnan(partial_L):
        sys.exit("Fatal error: NaN log-likelihood.")
    if partial_L < 0:
        sys.exit("Fatal error: Negative log-likelihood.")
    return partial_L


def get_single_timeline_likelihood_map_sahoo(timeline_tupple):
    """
    # this implementation is based on Zuchinni, p.49:
    # Hidden Markov Models for Time Series: An Introduction Using R, Second Edition
    """

    global pi, x_coeffs_mat, z_coeffs_mat, states, transition_index, transition_model, prob_stored_values_dict, T_diag, beta_coeffsfor_x

    timelineO, timelineX, timelineZ = timeline_tupple
    E = get_emission_probs(timelineZ[0],  timelineO[0])
    tmp = scipy.linalg.blas.sgemv(alpha=1.0, a=E.T, x=pi)
    w_1 = scipy.linalg.blas.ddot(tmp, I.T)
    phi_prev = tmp / w_1
    l = np.empty(len(timelineO))
    l[0] = w_1
    for i in range(1, len(timelineO)):
        #treansition probs:
        T = np.matmul(x_coeffs_mat, timelineX[i - 1])
        T = np.exp(T - np.logaddexp.reduce(T, axis=1, keepdims=True))
        E = get_emission_probs(timelineZ[i],  timelineO[i])
        tmp = scipy.linalg.blas.sgemv(alpha=1.0, a=T.T, x=phi_prev)
        v = scipy.linalg.blas.sgemv(alpha=1.0, a=E, x=tmp.T)
        u = scipy.linalg.blas.ddot(v, I.T)
        if u > 1:
            if u < 1.0001:
                u = 1  # (rounding error)
            else:
                print(">>>>>:", v, u)
                sys.exit(">>> Fatal error: u > 1")
        l[i] = u
        phi_prev = v / u
    return sum(np.log(l))


def get_single_timeline_likelihood_map(timeline_tupple):
    """
    # this implementation is based on Zuchinni, p.49:
    # Hidden Markov Models for Time Series: An Introduction Using R, Second Edition
    """
    global pi, x_coeffs_mat, z_coeffs_mat, states, transition_index, transition_model, prob_stored_values_dict, T_diag, beta_coeffsfor_x
    timelineO, timelineX, timelineZ = timeline_tupple
    E = get_emission_probs(timelineZ[0],  timelineO[0])
    # BLAS:
    tmp = scipy.linalg.blas.sgemv(alpha=1.0, a=E.T, x=pi)
    w_1 = scipy.linalg.blas.ddot(tmp, I.T)
    phi_prev = tmp / w_1
    l = np.empty(len(timelineO))
    l[0] = w_1
    ### This is for controlling for contrained transitions (after completion of a task)
    prev_number_ofTasks = timelineX[0][transition_index]
    for i in range(1, len(timelineO)):
        T = T_diag
        if timelineX[i - 1][transition_index] != prev_number_ofTasks:
            T, prev_number_ofTasks = get_transition_probs(timelineX[i - 1])
        E = get_emission_probs(timelineZ[i],  timelineO[i])
        # BLAS:
        tmp = scipy.linalg.blas.sgemv(alpha=1.0, a=T.T, x=phi_prev)
        v = scipy.linalg.blas.sgemv(alpha=1.0, a=E, x=tmp.T)
        u = scipy.linalg.blas.ddot(v, I.T)
        if u > 1:
            if u < 1.0001:
                u = 1  # (rounding error)
            else:
                print(">>>>>:", v, u)
                sys.exit(">>> Fatal error: u > 1")
        l[i] = u
        phi_prev = v / u
    return sum(np.log(l))




def init_params(theta, variables_in_z, states,
                emission_distro_pars, variables_in_x, transition_model):


    gamma_coeffs_for_z, ind = get_coeffs_z_from_theta(theta, variables_in_z, states, emission_distro_pars)
    beta_coeffsfor_x, ind = get_beta_coeffs(theta, ind, variables_in_x, states, transition_model)
    sigma_pars = None
    pi = []
    for j in range(ind, len(theta)):
        pi.append(theta[j])
    pi = softmax(pi)
    return pi, gamma_coeffs_for_z, beta_coeffsfor_x, sigma_pars


# emissions
def get_coeffs_z_from_theta(theta, variables_in_z, states, emission_distro_pars):
    coeffsZ = {}
    l = 0
    for i in states:
        coeffsZ[i] = {}
        #let's fill w zeros our baseline:
        coeffsZ[i][0] = []
        for k in range(variables_in_z):
            coeffsZ[i][0].append(0)
        for j in range(1, emission_distro_pars + 1):  # Emission params is number of symbols - 1
            if j not in coeffsZ[i]: coeffsZ[i][j] = []
            for _ in range(variables_in_z):
                coeffsZ[i][j].append(theta[l])
                l += 1
    return coeffsZ, l





def get_beta_coeffs(theta, k, variablesInX, states, transitionModel):
    betaCoeffsforX = {}
    if 'multinomial' in transitionModel:
        for state in states:
            betaCoeffsforX[state] = {}
            for stateTo in states:
                betaCoeffsforX[state][stateTo] = []
                if stateTo == (len(states) - 1):
                    for i in range(variablesInX):
                        betaCoeffsforX[state][stateTo].append(0)
                else:
                    for i in range(variablesInX):
                        betaCoeffsforX[state][stateTo].append(theta[k])
                        k += 1


    return betaCoeffsforX, k




def get_emission_probs(curZ, outcome):

    global z_coeffs_mat
    E = np.matmul(z_coeffs_mat, curZ)
    E = np.exp(E - np.logaddexp.reduce(E, axis=1, keepdims=True))
    return np.diag(E[:, outcome])



# transition_model = 'multinomial'
def get_transition_probs(curX):
    global x_coeffs_mat, transition_model,prob_stored_values_dict,transition_index
    cur_number_of_tasks = curX[transition_index]
    if 'multinomial' in transition_model:
        T = np.matmul(x_coeffs_mat, curX)
        T = np.exp(T - np.logaddexp.reduce(T, axis=1, keepdims=True))
    return T, cur_number_of_tasks


# ### Create priors

def createPriors(variables_in_X, variables_in_Z, states, transition_model,
                 emission_distro_pars):
    """
    # transition_model = 'multinomial',
    # assumes multinomial emissions.
    #############################
    # Softmax overparametrization: http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/
    #############################
    # Each state requires variables_in_X * (len(states)-1).
    # That is because the possible classes from each state are all the other states K=(len(states)).
    # Hence, I can set the parameters for one of these classes to zero, which allows the estimation of the
    # K-1  vectors. (i.e., K-1 * variables_in_X)
    """
    theta = []
    emission_parameters = len(states) * variables_in_Z * emission_distro_pars
    for i in range(emission_parameters):
        theta.append(np.random.uniform(-2, 2))
    K = len(states)
    if 'multinomial' in transition_model:
        transition_parameters = K * (K - 1) * variables_in_X
    for i in range(transition_parameters):
        theta.append(np.random.uniform(-1, 1))
    pi = []
    for i in range(K):
        pi.append(np.random.random())
    # softmax of pi happens in init_params
    for p in pi:
        theta.append(p)
    return theta, transition_parameters, emission_parameters, pi


def transform_input_to_HMM_timelines(X, y, transition_application_employer_matrix,
                                     verbal, n_cores):
    """
    #ranked by employer, assume first column to be employer_id (employer_index)
    #use also transition indices [transition_indices]
    #results should be three lists of np.arrays: Xtimelines, Z_timelines, O_timelines.
    #employer column, application column, transition_columns
    """

    if verbal: print("Transforming input matrix to HMM sequences...")
    X_timelines = {core: [] for core in range(n_cores)} if n_cores > 1 else []
    Z_timelines = {core: [] for core in range(n_cores)} if n_cores > 1 else []
    O = {core: [] for core in range(n_cores)} if n_cores > 1 else []
    X = np.insert(X, X.shape[1], np.full(X.shape[0], 1), axis=1)
    constant_ind = X.shape[1] - 1
    emissionIndices = [x for x in range(X.shape[1])]
    X = np.insert(X, X.shape[1], y, axis=1)
    outcome_index = X.shape[1] - 1
    X = np.concatenate((X,transition_application_employer_matrix), axis=1)
    employer_column_index = X.shape[1]-1 #last
    application_column_index = employer_column_index-1 # one before last
    applications_timelines = []
    transition_columns_indices = [i for i in range(outcome_index+1,application_column_index)]
    if n_cores == 1:
        for i in np.unique(X[:, employer_column_index]):
            curTimeline = X[X[:, employer_column_index] == i, :]
            applications_timelines.append(curTimeline[:, application_column_index])
            X_timeline = curTimeline[:, transition_columns_indices + [constant_ind]].astype(np.float32)
            X_timelines.append(X_timeline)
            Z_timeline = curTimeline[:, emissionIndices].astype(
                np.float32)  # includes constant --see creation of emission indices
            Z_timelines.append(Z_timeline)
            O_timeline = curTimeline[:, outcome_index]
            O.append(O_timeline.astype(np.int32))
    else:
        points_per_core = X.shape[0] / n_cores
        cur_core = 0
        cur_points = 0
        for i in np.unique(X[:, employer_column_index]):
            # get single employer timeline
            curTimeline = X[X[:, employer_column_index] == i, :]
            cur_points += curTimeline.shape[0]
            #
            # get timeline's transitions (X), emissions (Z) and outcomes.
            X_timeline = curTimeline[:, transition_columns_indices + [constant_ind]].astype(np.float32)
            X_timelines[cur_core].append(X_timeline)
            Z_timeline = curTimeline[:, emissionIndices].astype(
                np.float32)  # includes constant --see creation of emission indices
            Z_timelines[cur_core].append(Z_timeline)
            O_timeline = curTimeline[:, outcome_index]
            O[cur_core].append(O_timeline.astype(np.int32))
            if cur_points > points_per_core and cur_core < (n_cores - 1):
                cur_core += 1
                if verbal: print("new core:", cur_points)
                cur_points = 0
        if verbal: print("final core:", cur_points)
    return X_timelines, Z_timelines, O, applications_timelines


# ### Predict probabilities
def get_predictions(theta_star, X, Z, O, applications_timelines, variables_in_z, variables_in_x, states, emission_distro_pars, classes,
                    transition_model, transition_index, deterministic_flag=True):


    (pi, gammaCoeffsForZ, betaCoeffsforX, sigmaPars) = init_params(theta_star, variables_in_z, states,
                                                                                      emission_distro_pars, variables_in_x,
                                                                                      transition_model)

    if not deterministic_flag:
        np.random.seed(0)
    probs = []
    truth = []
    application = []
    for i in range(len(X)):
        client_X = X[i]
        client_O = O[i]
        client_Z = Z[i]
        apps = applications_timelines[i]
        curState = np.argmax(pi) if deterministic_flag else get_stochastic_draw_from_probs(pi)  ###independent of structure.
        ### This is for controlling the transitions
        prev_number_ofTasks = client_X[0][transition_index]
        for j in range(len(client_O)):
            sampleProbs = get_emission_probs_from_state(curState, gammaCoeffsForZ, client_Z[j]
                                                        , emission_distro_pars)
            if '_constrained' in transition_model:
                curNumberOfTasks = client_X[j][transition_index]
                if prev_number_ofTasks == curNumberOfTasks:
                    newState = curState
                else:
                    newState = get_next_state(client_X[j], betaCoeffsforX,  deterministic_flag,
                                              curState,
                                              transition_model, states)
                    prev_number_ofTasks = curNumberOfTasks
            ###Mulrionomial, transition possible at every instance
            else:
                newState = get_next_state(client_X[j], betaCoeffsforX, deterministic_flag, curState,
                                          transition_model, states)

            curState = newState
            probs.append(sampleProbs)
            truth.append(client_O[j])
            application.append(apps[j])
    predictions = pd.DataFrame(probs, columns = ['label_' + str(i) for i in classes])
    predictions['truth'] =list(truth)
    predictions['application'] = list(application)
    return predictions


def get_individual_log_l(client_X, client_Z, client_O,
                         Z_coeffs_mat, probDict, transition_index1, transitionModel,
                         states, xCoeffsMat, pi):
    sumTotal = 0
    global z_coeffs_mat, x_coeffs_mat,transition_model, prob_stored_values_dict,transition_index
    z_coeffs_mat = Z_coeffs_mat
    x_coeffs_mat = xCoeffsMat
    transition_model = transitionModel
    prob_stored_values_dict, transition_index = probDict,transition_index1
    T_diag = np.zeros(shape=(len(states), len(states)))
    np.fill_diagonal(T_diag, 1)
    for potSeq in itertools.product(states, repeat=len(client_O)):
        prev_number_ofTasks = client_X[0][transition_index]
        cur_simulated_seq = np.array(potSeq)
        initial_state = cur_simulated_seq[0]
        Ediag = get_emission_probs(client_Z[0],
                                   client_O[0])

        firt_term = np.dot(pi, Ediag)[initial_state]

        product = firt_term
        # for nextState in potSeq:
        state_at_t_minus_1 = initial_state
        for t in range(1, cur_simulated_seq.shape[0]):
            state_at_t = cur_simulated_seq[t]
            Ediag = get_emission_probs(client_Z[t],
                                       client_O[t])
            T = T_diag
            if client_X[t - 1][transition_index] != prev_number_ofTasks:
                T, prev_number_ofTasks = get_transition_probs(client_X[t - 1])
            product *= Ediag.diagonal()[state_at_t] * T[state_at_t_minus_1][state_at_t]
            state_at_t_minus_1 = state_at_t
        sumTotal += product

    l = sumTotal
    print(sumTotal)
    return np.log(l)



def get_likelihood_validation(X, Z, O, pi1, xCoeffsMat,zCoeffsMat,
                          transition_model1,states1,transition_index1,output=None):
    global  T_diag,I,prob_stored_values_dict,states,pi,z_coeffs_mat, x_coeffs_mat, transition_model, prob_stored_values_dict, transition_index
    z_coeffs_mat = zCoeffsMat
    x_coeffs_mat = xCoeffsMat
    transition_model = transition_model1
    transition_index =  transition_index1
    states,pi = states1,pi1
    I = np.ones(len(states))
    prob_stored_values_dict = {}
    T_diag = np.zeros(shape=(len(states), len(states)))
    np.fill_diagonal(T_diag, 1)
    partial_L = -sum(map(get_single_timeline_likelihood_map, zip(O, X, Z)))

    if np.isnan(partial_L):
        sys.exit("Fatal error: NaN log-likelihood.")
    if partial_L < 0:
        sys.exit("Fatal error: Negative log-likelihood.")
    if output is not None:
        output.put(partial_L)
    else:
        return partial_L

