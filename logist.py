import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


from features_extraction import features_matrix, target_vector
from utils import timedcall, cross_validate, split_validate


print 'reading train data...'
df   = pd.read_csv('data/train.csv')
X, y = features_matrix(df), target_vector(df)

print 'reading test data...'
df_test  = pd.read_csv('data/test.csv')
ids      = df_test.match_id
features = filter(lambda col: col != 'match_id', df_test.columns)
X_test   = df_test[features]




@timedcall
def cross_validate_logistic(X, y, params = None):
    params = params or {'C': np.power(10.0, np.arange(-3, 0))}
    return cross_validate(LogisticRegression(),
                          params,
                          X,
                          y,
                          'roc_auc')


@timedcall
def split_validate_logistic(X, y, validation_size = 0.2, params = None):
    params = params or {'C': np.power(10.0, np.arange(-5, 6))}
    return split_validate(LogisticRegression(),
                          params,
                          X,
                          y,
                          validation_size,
                          'roc_auc')



print 'tuning parameters for logistic regression...'

clf = cross_validate_logistic(X, y)

params = clf.best_params_.items()
show_pair = lambda pair: '%r: %f' % pair

print 'parameters chosen: %s' % ', '.join(map(show_pair, params))

print 'CV score = %f' % clf.best_score_

print 'making predictions...'

predictions = clf.predict_proba(X_test)
radiant_win = np.array(map(lambda x: x[1], predictions))



if __name__ == '__main__':
    path = raw_input('Where do you want to write the file to? Enter a relative path.\n')
    outcome = pd.DataFrame()
    outcome['match_id'] = ids
    outcome['radiant_win'] = radiant_win

    outcome.to_csv(path, index = False)
