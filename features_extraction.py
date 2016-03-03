import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import re





def features_pipeline(df, feature_func):
    '''Takes a dataframe df and functions feature_func, which in turn
    take a dataframe as an argument and return a new dataframe.
    Returns a new dataframe after applying all feature_func's.'''
    
    func = lambda d, f: f(d)
    
    return reduce(func, feature_func, df)




def transform_pipeline(df, transforms, fit = False):
    f = 'fit_transform' if fit else 'transform'
    for tr in transforms:
        df = getattr(tr, f)(df)
    return df, transforms




def remove_feature(df, feature_name):
    try:
        df = df.drop(feature_name, axis = 1)
    except:
        print 'Warning: trying to remove an inexistent feature %r' % feature_name
    return df




def remove_features(df, feature_names):
    return reduce(remove_feature, feature_names, df)




def add_feature(df, name, value):
    new_df = df.copy()
    new_df[name] = value
    return new_df




def add_features(df, D):
    for name, value in D.items():
        df = add_feature(df, name, value)
    return df




def categorical_features(df):
    features = ['lobby_type'] if 'lobby_type' in df.columns else []
    return features + filter(lambda col: col.endswith('_hero'), df.columns)




def remove_categorical(df):
    return remove_features(df, categorical_features(df))




def future_features():
    return ['tower_status_dire',
            'barracks_status_dire',
            'barracks_status_radiant',
            'tower_status_radiant',
            'duration',
            'radiant_win']




def remove_future_features(df):
    return remove_features(df, future_features())




_heros_cnt = len(pd.read_csv('data/dictionaries/heroes.csv').id.unique())




def heros_bag(df, num_heros = _heros_cnt):
    shape = rows, cols = (df.shape[0], num_heros)
    X_pick = np.zeros(shape)

    for i in xrange(rows):
        for p in xrange(5):
            X_pick[i, df.ix[i, 'r%d_hero' % (p+1)]-1] = 1
            X_pick[i, df.ix[i, 'd%d_hero' % (p+1)]-1] = -1
        
    return X_pick




def add_heros_bag(df, num_heros = _heros_cnt):
    colnames = list(df.columns) + map(lambda i: 'hero_%d' % i, xrange(1, num_heros + 1))
    return pd.DataFrame(np.hstack([df, heros_bag(df, num_heros)]),
                        columns=colnames)




_lobbies_cnt = len(pd.read_csv('data/dictionaries/lobbies.csv').id.unique())




def lobby_bag(lobby_feature, num_lobbies = _lobbies_cnt):
    lobby_vec = np.array(lobby_feature)
    shape = rows, cols = (lobby_vec.shape[0], num_lobbies)
    bag = np.zeros(shape)
    
    for i in xrange(rows):
        bag[i, lobby_vec[i]] = 1
    
    return bag




def add_lobby_bag(df, num_lobbies = _lobbies_cnt):
    bag = lobby_bag(df.lobby_type, num_lobbies)
    colnames = list(df.columns) + map(lambda i: 'lobby_%d' % i, xrange(1, num_lobbies + 1))
    return pd.DataFrame(np.hstack([df, bag]), columns = colnames)




def fill_nan_with(df, value = 0):
    return df.fillna(value)




def target_name():
    return 'radiant_win'




def target_vector(df):
    return df[target_name()]




def features_matrix(df):
    return remove_feature(df, target_name())



def features_label(df):
    return features_matrix(df), target_vector(df)



def remove_redundant(df):
    features_to_remove = ['start_time']
    return remove_features(df, features_to_remove)




def preprocess(df, train = True):
    preprocessors = [remove_future_features,
                     remove_redundant,
                     features_matrix,
                     fill_nan_with,
                     add_heros_bag,
                     add_lobby_bag,
                     remove_categorical,
                     team_items_difference,
                     team_abilities_difference]

    s = 'train' if train else 'test'
    preprocessors = [add_abilities_features('data/abilities_%s.csv' % s),
                     add_items_features('data/items_%s.csv' % s)] + preprocessors
    
    return features_pipeline(df, preprocessors)




def add_abilities_features(filename):
    abilities = pd.read_csv(filename)
    def add(df):
        return df.merge(abilities, on = 'match_id')
    return add



def add_items_features(filename):
    items = pd.read_csv(filename)
    def add(df):
        return df.merge(items, on = 'match_id')
    return add





def team_feature_difference(df, feature):
    regex      = re.compile('%s_(?:r|d)_([0-9]+)' % feature)
    extract_id = lambda colname: int(regex.match(colname).group(1))

    colnames = filter(lambda col: col.startswith('%s_' % feature), df.columns)
    ids      = set(map(extract_id, colnames))
    
    s, r, d = feature + '_%d', feature + '_r_%d', feature + '_d_%d'
    for i in ids:
        name = s % i
        df[name] = df[r % i] - df[d % i]

    df = df.drop(colnames, axis = 1, inplace = False)
    return df



def team_items_difference(df):
    return team_feature_difference(df, 'item')



def team_abilities_difference(df):
    return team_feature_difference(df, 'ability')




def extract_all(df, df_test):
    '''Preprocesses df and df_test and returns X, y, X_test.'''

    y = target_vector(df)
    
    X      = preprocess(df, train = True)
    X_test = preprocess(df_test, train = False)
    
    xnames     = X.columns
    xtestnames = X_test.columns
    
    X, transforms = transform_pipeline(X, [StandardScaler()], fit = True)
    X_test, _     = transform_pipeline(X_test, transforms)
    
    X      = pd.DataFrame(X, columns = xnames)
    X_test = pd.DataFrame(X_test, columns = xtestnames)
    
    return X, y, X_test



def write_train_test_dataframes(train_input,
                                test_input,
                                train_output,
                                test_output):
    df      = pd.read_csv(train_input)
    df_test = pd.read_csv(test_input)
    
    X, y, X_test = extract_all(df, df_test)
    
    X[target_name()] = y
    X_test['match_id'] = df_test.match_id
    
    X.to_csv(train_output, index = False)
    X_test.to_csv(test_output, index = False)





if __name__ == '__main__':
    reply = raw_input('Are you sure you want to create train and test dataframes? (Type Yes or No) ')
    if reply.lower() == 'yes':
        filenames = tuple(map(lambda name: 'data/%s.csv' % name,
                              ['features', 'features_test', 'train', 'test']))
        write_train_test_dataframes(*filenames)
