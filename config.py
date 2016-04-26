import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

from utils import (
    create_Cs,
    cross_validate_logistic,
    validate_logistic)



class Config(object):
    def __init__(self,
                 train_filename,
                 test_filename,
                 output_filename,
                 n_folds,
                 lcurve,
                 train_model,
                 validate,
                 n_components,
                 scaler,
                 C_lo,
                 C_hi):
        self.train_filename = train_filename
        self.test_filename = test_filename
        self.output_filename = output_filename
        self.n_folds = n_folds
        self.cv_shuffle = True
        self.lcurve = lcurve
        self.train_model = train_model
        self.validate = {'cv': cross_validate_logistic,
                         'split': validate_logistic}[validate]
        self.train_size = 0.8  # consider adding as a command-line argument
        self.random_state = 42
        self.index_col = 'match_id'
        self.target_name = 'radiant_win'
        self.Cs = create_Cs(C_lo, C_hi)
        self.voting_weights = [[x, 1-x] for x in np.arange(0, 1.1, 0.2)]
        self.voting = 'soft'
        self.scoring = 'roc_auc'
        self.scoring_fn = {'roc_auc': roc_auc_score}[self.scoring]
        self.train_sizes = [0.25, 0.5, 0.75, 1]
        self.n_components=n_components
        self.scaler = {'standard': StandardScaler,
                       'minmax': MinMaxScaler,
                       'maxabs': MaxAbsScaler}[scaler]()
