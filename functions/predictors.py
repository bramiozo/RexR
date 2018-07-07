# Here come the predictors, i.e. the (semi)-supervised learning algorithms
#
#class classifier
#
# Tree methods
# Stochastic methods
# stacked nonlinear methods
# template matching
# linear methods
## ensembling

#class regressor
# Tree methods
# Stochastic methods
# stacked nonlinear methods
# template matching
# linear methods
##  ensembling
import numpy as np
import pandas as pd

from sklearn import preprocessing, svm, tree, ensemble, naive_bayes, neural_network, model_selection, metrics
from sklearn import discriminant_analysis, decomposition, cross_decomposition
from sklearn.metrics import confusion_matrix
from sklearn.cluster import AffinityPropagation
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.manifold import Isomap as ISO
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectFdr, SelectFpr
from sklearn.feature_selection import f_classif, chi2
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn import preprocessing, svm, tree, naive_bayes
from sklearn import linear_model, neural_network, model_selection, metrics
from sklearn import discriminant_analysis, gaussian_process


from collections import Counter
from math import*

from scipy.spatial.distance import minkowski
from scipy.spatial.distance import cdist
from scipy import sparse
from scipy.stats import wilcoxon, mannwhitneyu

from decimal import Decimal
from time import time
import matplotlib.pyplot as plt
from itertools import cycle
from itertools import product

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import MaxPooling1D
from keras.callbacks import Callback
from keras.layers.convolutional import Conv1D
from keras.layers import Input

import lightgbm as lgb
from xgboost.sklearn import XGBClassifier as xgb


from time import time
from functools import wraps

import _helpers
import copy
import itertools
#import sys
#sys.setrecursionlimit(10000)

import gc
gc.enable()


def _benchmark_classifier(model, x, y, splitter, framework='sklearn', Rclass=None):
    pred = np.zeros(shape=y.shape)
    acc = np.zeros(shape=y.shape)
    coef = np.zeros(shape=(1, x.shape[1]))
    threshold = 0.5
    if framework == 'sklearn':
        for train_index, test_index in splitter.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            print("Training model: {}".format(model[0]))
            model[1].fit(x_train, y_train)
            # pred_test = model[1].predict_proba(x_test) # (model[1].predict_proba(x_test)>threshold).astype(int)
            pred_test_ = model[1].predict(x_test)  # [np.round(l[1]).astype(int) for l in pred_test]
            pred[test_index] = pred_test_  # np.round(pred_test)[0]
            acc[test_index] = metrics.accuracy_score(y_test, pred_test_)
            # coef += model.coef_
        ######################################################
        ##### For last split, show confusion matrix and ROC ##
        ######################################################
        # X_train, X_test, y_train, y_test = train_test_split(X_prep, y, test_size=0.2, random_state=42)
        if Rclass.VIZ == True:
            pred_train = model[1].predict_proba(x_train)
            fig, ax = plt.subplots(1, 3)
            fig.set_size_inches(15, 5)
            plot_cm(ax[0], y_train, pred_train, [0, 1], 'Confusion matrix (TRAIN)', threshold)
            plot_cm(ax[1], y_test, pred_test, [0, 1], 'Confusion matrix (TEST)', threshold)
            plot_auc(ax[2], y_train, pred_train, y_test, pred_test, threshold)
            plt.tight_layout()
            plt.show()


    elif framework == 'custom_rvm':
        import rvm
        for train_index, test_index in splitter.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = rvm.rvm(x_train, y_train, noise=0.01)
            model.iterateUntilConvergence()
            pred_test = np.reshape(np.dot(x_test, model.wInferred), newshape=[len(x_test), ]) / 2 + 0.5
            pred_test_ = np.round(pred_test);
            pred[test_index] = pred_test_
            acc[test_index] = metrics.accuracy_score(y_test, pred_test_)

        ######################################################
        ##### For last split, show confusion matrix and ROC ##
        ######################################################
        # X_train, X_test, y_train, y_test = train_test_split(X_prep, y, test_size=0.2, random_state=42)
        if Rclass.VIZ == True:
            pred_train = np.reshape(np.dot(x_train, model.wInferred), newshape=[len(x_train), ]) / 2 + 0.5
            fig, ax = plt.subplots(1, 3)
            fig.set_size_inches(15, 5)
            plot_cm(ax[0], y_train, pred_train, [0, 1], 'Confusion matrix (TRAIN)', threshold)
            plot_cm(ax[1], y_test, pred_test, [0, 1], 'Confusion matrix (TEST)', threshold)
            plot_auc(ax[2], y_train, pred_train, y_test, pred_test, threshold)
            plt.tight_layout()
            plt.show()
    elif framework == 'keras':
        for train_index, test_index in splitter.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model[1].fit(x_train, y_train, batch_size=10, epochs=5, verbose=1,
                         callbacks=[BL], validation_data=(np.array(x_test), np.array(y_test)))
            # score = model[1].evaluate(np.array(x_test), np.array(y_test), verbose=0)
            # print('Test log loss:', score[0])
            # print('Test accuracy:', score[1])
            pred_test = model[1].predict(x_test)[:, 0]
            pred_test_ = np.round(pred_test)
            pred[test_index] = pred_test_
            acc[test_index] = metrics.accuracy_score(y_test, pred_test_)

        ######################################################
        ##### For last split, show confusion matrix and ROC ##
        ######################################################
        # https://github.com/natbusa/deepcredit/blob/master/default-prediction.ipynb
        if Rclass.VIZ == True:
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 2, 1)
            plt.title('loss, per batch')
            plt.plot(BL.get_values('loss', 1), 'b-', label='train');
            plt.plot(BL.get_values('val_loss', 1), 'r-', label='test');
            plt.legend()
            #
            plt.subplot(1, 2, 2)
            plt.title('accuracy, per batch')
            plt.plot(BL.get_values('acc', 1), 'b-', label='train');
            plt.plot(BL.get_values('val_acc', 1), 'r-', label='test');
            plt.legend()
            plt.show()

            y_train_pred = model[1].predict_on_batch(np.array(x_train))[:, 0]
            y_test_pred = model[1].predict_on_batch(np.array(x_test))[:, 0]

            fig, ax = plt.subplots(1, 3)
            fig.set_size_inches(15, 5)

            plot_cm(ax[0], y_train, y_train_pred, [0, 1], 'Confusion matrix (TRAIN)')
            plot_cm(ax[1], y_test, y_test_pred, [0, 1], 'Confusion matrix (TEST)')

            plot_auc(ax[2], y_train, y_train_pred, y_test, y_test_pred)

            plt.tight_layout()
            plt.show()
    return pred, acc

def _benchmark_classifier_ensemble(models, x, y, splitter, framework='sklearn', Rclass=None):
    pred = np.zeros(shape=y.shape)
    acc = np.zeros(shape=y.shape)
    coef = np.zeros(shape=(1, x.shape[1]))
    threshold = 0.5
    if framework == 'sklearn':
        for train_index, test_index in splitter.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            print("Training model: {}".format(model[0]))
            model[1].fit(x_train, y_train)
            # pred_test = model[1].predict_proba(x_test) # (model[1].predict_proba(x_test)>threshold).astype(int)
            pred_test_ = model[1].predict(x_test)  # [np.round(l[1]).astype(int) for l in pred_test]
            pred[test_index] = pred_test_  # np.round(pred_test)[0]
            acc[test_index] = metrics.accuracy_score(y_test, pred_test_)
            # coef += model.coef_
        ######################################################
        ##### For last split, show confusion matrix and ROC ##
        ######################################################
        # X_train, X_test, y_train, y_test = train_test_split(X_prep, y, test_size=0.2, random_state=42)
        if Rclass.VIZ == True:
            pred_train = model[1].predict_proba(x_train)
            fig, ax = plt.subplots(1, 3)
            fig.set_size_inches(15, 5)
            plot_cm(ax[0], y_train, pred_train, [0, 1], 'Confusion matrix (TRAIN)', threshold)
            plot_cm(ax[1], y_test, pred_test, [0, 1], 'Confusion matrix (TEST)', threshold)
            plot_auc(ax[2], y_train, pred_train, y_test, pred_test, threshold)
            plt.tight_layout()
            plt.show()


    elif framework == 'custom_rvm':
        import rvm
        for train_index, test_index in splitter.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = rvm.rvm(x_train, y_train, noise=0.01)
            model.iterateUntilConvergence()
            pred_test = np.reshape(np.dot(x_test, model.wInferred), newshape=[len(x_test), ]) / 2 + 0.5
            pred_test_ = np.round(pred_test);
            pred[test_index] = pred_test_
            acc[test_index] = metrics.accuracy_score(y_test, pred_test_)

        ######################################################
        ##### For last split, show confusion matrix and ROC ##
        ######################################################
        # X_train, X_test, y_train, y_test = train_test_split(X_prep, y, test_size=0.2, random_state=42)
        if Rclass.VIZ == True:
            pred_train = np.reshape(np.dot(x_train, model.wInferred), newshape=[len(x_train), ]) / 2 + 0.5
            fig, ax = plt.subplots(1, 3)
            fig.set_size_inches(15, 5)
            plot_cm(ax[0], y_train, pred_train, [0, 1], 'Confusion matrix (TRAIN)', threshold)
            plot_cm(ax[1], y_test, pred_test, [0, 1], 'Confusion matrix (TEST)', threshold)
            plot_auc(ax[2], y_train, pred_train, y_test, pred_test, threshold)
            plt.tight_layout()
            plt.show()
    elif framework == 'keras':
        for train_index, test_index in splitter.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model[1].fit(x_train, y_train, batch_size=10, epochs=5, verbose=1,
                         callbacks=[BL], validation_data=(np.array(x_test), np.array(y_test)))
            # score = model[1].evaluate(np.array(x_test), np.array(y_test), verbose=0)
            # print('Test log loss:', score[0])
            # print('Test accuracy:', score[1])
            pred_test = model[1].predict(x_test)[:, 0]
            pred_test_ = np.round(pred_test)
            pred[test_index] = pred_test_
            acc[test_index] = metrics.accuracy_score(y_test, pred_test_)

        ######################################################
        ##### For last split, show confusion matrix and ROC ##
        ######################################################
        # https://github.com/natbusa/deepcredit/blob/master/default-prediction.ipynb
        if Rclass.VIZ == True:
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 2, 1)
            plt.title('loss, per batch')
            plt.plot(BL.get_values('loss', 1), 'b-', label='train');
            plt.plot(BL.get_values('val_loss', 1), 'r-', label='test');
            plt.legend()
            #
            plt.subplot(1, 2, 2)
            plt.title('accuracy, per batch')
            plt.plot(BL.get_values('acc', 1), 'b-', label='train');
            plt.plot(BL.get_values('val_acc', 1), 'r-', label='test');
            plt.legend()
            plt.show()

            y_train_pred = model[1].predict_on_batch(np.array(x_train))[:, 0]
            y_test_pred = model[1].predict_on_batch(np.array(x_test))[:, 0]

            fig, ax = plt.subplots(1, 3)
            fig.set_size_inches(15, 5)

            plot_cm(ax[0], y_train, y_train_pred, [0, 1], 'Confusion matrix (TRAIN)')
            plot_cm(ax[1], y_test, y_test_pred, [0, 1], 'Confusion matrix (TEST)')

            plot_auc(ax[2], y_train, y_train_pred, y_test, y_test_pred)

            plt.tight_layout()
            plt.show()
    return pred, acc

# class Classifier:
#     """
#     # Tree methods: LGBM, XGB, RF, ET
#     # Stochastic methods: GPC, NB
#     # stacked nonlinear methods: DNN, CNN
#     # template matching: lSVM
#     # linear methods: LR
#     """
#
#
#     def __init__(self, ):
#
#
#
#
#
#     def fit(self):
#
#
#
#     def predict(self):
#
#
#
#     def weights(self):
#         """
#
#         :return: [{'method': xxx, 'weights': pd.DataFrame(weights, index)}]
#         """



class Ensemble:
    """
     ens = ensemble()
     ens._add_trained_models(models)
     preds, acc = ens.predict(X_test)

     ens._add_untrained_models(models)
     ens.fit(X_train, y_train, models)
     preds, acc = ens.predict(X_test)
    """
    _models_trained  = []
    _models_untrained = []

    def __init__(self, weighted = True,
                       use_accuracy = True,
                       use_uncertainty = True,
                       voting = 'soft',
                       cross_validation = True,
                       folds = 10,
                       SEED = 1234,
                       sample_weights = None,
                       training_viz = True):
        """

        :param weighted: weighted by accuracy as estimated through cross-validation
        :param use_accuracy:  ..
        :param use_uncertainty: use probability estimate as proxy for uncertainty and use as weight
        :param voting: 'soft' or 'hard'
        :param cross_validation: use cross-validation for accuracy estimation
        :param folds: number of folds for cross-validation
        :param SEED: seed for cross-validation
        :param sample_weights: list of tuples to map weights on classifications, e.g. [('HR', 0.8), ('LR', 0.2)
        """


        self.parameters = {'weighted':weighted,
                           'use_accuracy': use_accuracy,
                           'use_uncertainty': use_uncertainty,
                           'voting': voting,
                           'cross_validation': cross_validation,
                           'folds': folds}
        self.sample_weights =  sample_weights
        self.SEED = SEED
        self.VIZ= training_viz


    def _add_trained_models(self, models):
        """
            _models: [] with models: {'method':xxx, 'model':xxx, 'accuracy':[{'acc': xxx, 'var': xxx}]
        """
        self._models_trained += models

    def _add_untrained_models(self, models):
        """
            _models: [] with models {'method': xxx, 'model': xxx}
        """
        self._models_untrained += models

    def fit(self, X_train, y_train, MODELS = []):
        splitter = StratifiedKFold(self.parameters['folds'], random_state=self.SEED)
        def fw_fun(x): return(
            'rvm' if x in ['rvm'] else ('keras' if x in ['dnn', 'cnn'] else 'sklearn')
        )
        if str(type(MODELS)) == 'list' and len(MODELS) > 0:
            self._add_untrained_models(MODELS)
        else:
            MODELS = []
            if len(self._models_untrained) > 0:
                raise ValueError("_models_untrained cannot be empty..")

        # go through untrained models, fit them and add them to the _models_trained list.
        if self.parameters['cross_validation']:
            acc = _benchmark_classifier(self._models_untrained, X_train, y_train, splitter, Rclass=self)
            acc_var = np.var(acc)

        else:
            for model in self._models_untrained:
                acc = None
                acc_var = None
                _model = model['model'].fit(X_train, y_train)
                MODELS.append({'method': model['method'],
                               'model': _model,
                               'accuracy': {'acc': np.mean(acc), 'var': acc_var}})

        self._add_trained_models(MODELS)


    def predict(self, x):
        preds = []
        if len(self._models.trained) == 0:
            raise ValueError("There are no trained models loaded yet :(")

        for MODEL in self._models_trained:
            if MODEL['method'] == 'RVM':
                _pred = 1 - np.reshape(np.dot(x, MODEL['model'].wInferred), newshape=[len(x), ]) / 2 + 0.5
            elif MODEL['method'] == 'DNN':
                _pred = 1 - MODEL['model'].predict_on_batch(np.array(x))[:, 0]
            elif MODEL['method'] == 'CNN':
                _pred = 1 - MODEL['model'].predict(np.expand_dims(x, axis=2))[:, 0]
            elif MODEL['method'].lower() in ['lgbm', 'lightgbm']:
                _pred = 1 - MODEL['model'].predict_proba(x)[:, 0]
            else:
                _pred = 1 - MODEL['model'].predict_proba(x)[:, 0]
            preds.append(_pred)

        if ~self.parameters['weighted']:
            _preds = sum(preds) / len(preds)
            return _preds
        else:
            pd_list = []
            for idx, _pred in enumerate(preds):
                df = pd.DataFrame(data=_pred, columns=['proba'])
                df['id'] = df.index
                if (self.parameters['use_accuracy']==True):
                    df['acc'] = self._models_trained[idx]['accuracy'][0]['acc']
                pd_list.append(df)
            dfconcat = pd.concat(pd_list)
            if (self.parameters['use_accuracy']==True):
                dfconcat['weight'] = 2 * (dfconcat['proba'] - 0.5).abs() * dfconcat['acc']
            else:
                dfconcat['weight'] = 2 * (dfconcat['proba'] - 0.5).abs()
            _preds = dfconcat.groupby(by='id').apply(lambda x: (x.weight * x.proba).sum() / x.weight.sum()) \
                .reset_index()
            return _preds



