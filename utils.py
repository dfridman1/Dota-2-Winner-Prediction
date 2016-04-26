import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt
from time import clock
import random


def timedcall(f):
    def g(*args, **kwrgs):
        start = clock()
        res = f(*args, **kwrgs)
        g.elapsed = clock() - start
        return res
    return g


def create_Cs(lo, hi, q=10):
    ans = []
    curr = lo
    while curr <= hi:
        ans.append(curr)
        curr *= q
    return ans


def split(df, config):
    features = [col for col in df if col != config.target_name]
    return df[features], np.array(df[config.target_name])


def make_cv(y, config):
    return KFold(len(y),
                 n_folds=config.n_folds,
                 random_state=config.random_state,
                 shuffle=config.cv_shuffle)


@timedcall
def cross_validate_logistic(X, y, config, cv=None):
    params = {'C': config.Cs}
    cv = cv or make_cv(y, config)
    gs = GridSearchCV(LogisticRegression(),
                      param_grid=params,
                      scoring=config.scoring,
                      cv=cv)
    gs.fit(X, y)
    return gs


@timedcall
def validate_logistic(X, y, config):
    cv = cv_generator(len(y), config)
    return cross_validate_logistic(X, y, config, cv)


def sample(xs, k, random_state=None):
    if random_state is not None:
        random.seed(random_state)
    s = random.sample(xs, k)
    random.seed()
    return s


def cv_generator(size, config):
    mask = [False] * size
    k = int(size * config.train_size)
    train = sorted(sample(xrange(size), k, config.random_state))
    for num in train:
        mask[num] = True
    test = []
    for num, taken in enumerate(mask):
        if not taken:
            test.append(num)
    return [(np.array(train), np.array(test))]


@timedcall
def learning_curve(X, y, config):
    X, y = np.array(X), np.array(y)
    m, n = X.shape
    train_sizes = [int(m * i) for i in config.train_sizes]
    train_scores, test_scores = [], []
    for size in train_sizes:
        indices = sorted(sample(xrange(m), size, config.random_state))
        X_new, y_new = X[indices, :], y[indices]
        gs = config.validate(X_new, y_new, config)
        test_scores.append(gs.best_score_)
        y_pred = gs.predict_proba(X_new)
        y_pred = map(lambda x: x[1], y_pred)
        train_scores.append(config.scoring_fn(y_new, y_pred))
    return train_sizes, train_scores, test_scores


def plot_learning_curve(lc):
    sizes, train, val = lc
    plt.figure()
    plt.plot(sizes, train, label='training set')
    plt.plot(sizes, val, label='validation set')
    plt.xlabel('number of training examples')
    plt.ylabel('roc auc score')
    plt.legend()
    plt.show()
