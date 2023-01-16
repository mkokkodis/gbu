############################################
##### Author: Marios Kokkodis            ###
##### email: marios.kokkodis@gmail.com   ###
############################################
##### >>  Python notes: tested on PY38   ###
############################################


from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.utils import np_utils
import tensorflow
from custom_util_functions import split_sequences


class lstm_sklearn(ClassifierMixin, BaseEstimator):

    def __init__(self, neurons, epochs, batch_size, stacked=False):

        self.neurons = neurons
        self.stacked = stacked
        self.n_steps = 55  # median number of applications (restaurats) per task (choice set) -- see appendix O
        self.epochs = epochs
        self.batch_size = batch_size
        self.trained_model = None

    def get_params(self, deep=True):

        return dict(neurons=self.neurons,
                    stacked=self.stacked, epochs=self.epochs, batch_size=self.batch_size)

    def fit(self, X, y):

        np.random.seed(1234)
        tensorflow.random.set_seed(1234)
        X, y = split_sequences(X, y, self.n_steps)
        y = np_utils.to_categorical(y)
        self.n_features = X.shape[2]
        if self.neurons == -1: self.neurons = int(0.67 * (self.n_features + self.n_steps))
        model = Sequential()
        if self.stacked:
            model.add(LSTM(units=self.neurons, activation='relu', return_sequences=True,
                           input_shape=(self.n_steps, self.n_features)))
            model.add(LSTM(units=int(self.neurons / 2), activation='relu', input_shape=(self.n_steps, self.n_features)))
        else:
            model.add(LSTM(units=self.neurons, activation='relu', input_shape=(self.n_steps, self.n_features)))
        model.add(Dense(len(np.unique(y)), activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        model.fit(X, y, epochs=self.epochs, verbose=0, batch_size=self.batch_size)
        self.trained_model = model
        return self


    def decision_function(self, X, y, thr=0.5):
        if self.verbal: print("**** decision_function was called ****")
        # in binary, return 1D array : https://scikit-learn.org/stable/glossary.html#term-predict_proba
        if self.prob_predictions is None:
            self.prob_predictions = self.predict_proba(X, y)
        predicted_labels = [1 if x >= thr else 0 for x in self.prob_predictions['label_1']]
        return list(predicted_labels)

    def score(self, X, y):
        return self.decision_function(X, y)

    def predict(self, X):

        return self.decision_function(X)

    def predict_proba(self, X, y):
        X_test, _ = split_sequences(X, y, n_steps=self.n_steps)
        preds = self.trained_model.predict(X_test, verbose=0)
        return preds
