from time import clock
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold, train_test_split





def timedcall(f):
    def g(*args, **kwrgs):
        start = clock()
        res = f(*args, **kwrgs)
        g.elapsed = clock() - start
        return res
    return g




def concatMap(f, xs):
    res = []
    for x in xs:
        ys = f(x)
        for y in ys:
            res.append(y)
    return res




def sample(df, size, random_state = 241):
    indices = np.arange(0, df.shape[0])
    np.random.seed(random_state)
    selected = sorted(np.random.choice(indices,
                                       size = size,
                                       replace = False))
    np.random.seed()
    df = df.loc[selected, : ]
    df.index = np.arange(df.shape[0])
    return df




def cross_validate(clf, params, X, y, scoring, random_state = 241, cv = None):
    params = params or {}
    cv = cv or KFold(len(y),
                     n_folds = 5,
                     shuffle = True,
                     random_state = random_state)
    gs = GridSearchCV(clf,
                      param_grid = params,
                      scoring = scoring,
                      cv = cv)
    gs.fit(X, y)
    return gs




def create_validation_split_generator(size, validation_size = 0.4, random_state = 241):
    indices, mask = np.arange(size), np.zeros(size, dtype = bool)
    np.random.seed(random_state)
    validation = np.random.choice(indices,
                                  size = int(validation_size * size),
                                  replace = False)
    for x in validation:
        mask[x] = True

    train = []
    for i, x in enumerate(mask):
        if not x:
            train.append(i)
    
    np.random.seed()
    return [(np.array(train), validation)]



def split_validate(clf, params, X, y, validation_size, scoring, random_state = 241):
    size = X.shape[0]
    cv_generator = create_validation_split_generator(size,
                                                     validation_size,
                                                     random_state)
    return cross_validate(clf,
                          params,
                          X,
                          y,
                          scoring,
                          random_state,
                          cv_generator)
