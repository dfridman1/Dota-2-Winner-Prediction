import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import re
from calc_rating import add_synergy_antisynergy





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
    return df.drop(feature_names, axis = 1, inplace = False)




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



def remove_all_zero(df, prefix):
    '''Given a dataframe df and a prefix, returns a function that removes
    all columns that start with 'prefix' and  whose sum of absolute values
    is equal to zero.'''

    columns = filter(lambda col: col.startswith(prefix), df.columns)
    features_sum_abs = np.sum(df[columns].apply(abs))

    zero_features = features_sum_abs[features_sum_abs == 0].index

    return lambda df_: df_.drop(zero_features, axis = 1, inplace = False)



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
    preprocessors = [remove_redundant,
                     fill_nan_with,
                     add_heros_bag,
                     add_lobby_bag,
                     remove_categorical,
                     add_multiples,
                     add_divisions,
                     add_team_means,
                     add_team_medians,
                     team_items_difference,
                     team_abilities_difference]

    if train:
        preprocessors = [remove_future_features] + preprocessors

    s = 'train' if train else 'test'
    filenames = map(lambda name: 'data/' + (name % s) + '.csv', ['abilities_%s',
                                                                 'items_%s',
                                                                 'observers_%s',
                                                                 'speed_%s'])

    additional_features = map(add_features_merge, filenames)
    
    return features_pipeline(df, additional_features + preprocessors)




def add_features_merge(filename):
    features = pd.read_csv(filename)
    return lambda df: df.merge(features, on = 'match_id')





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




def team_feature_names(df):
    '''Returns feature names, for which team 'average' can ba calculated.'''
    regex = re.compile('(?:r|l)[0-9]_(.*)')
    res = []
    for col in df.columns:
        m = regex.match(col)
        if m is not None:
            res.append(m.group(1))
    return set(res)



def add_team_features(df, f, statistic_name):
    df_ = df.copy()
    names = team_feature_names(df)
    for name in names:
        for team in 'rd':
            extract = df_[['%s%d_%s' % (team, i, name) for i in xrange(1, 6)]]
            df_['%s_%s_%s' % (team, name, statistic_name)] = extract.apply(f, axis = 1)
    return df_



def combine_two_features(feature1, feature2, f):
    return f(feature1, feature2)


def add_playerwise_combined_features(df, f, f_name):
    df_ = df.copy()
    names = list(team_feature_names(df_))
    print names
    print 'number of team columns = %d' % len(names)
    for i in 'rd':
        for j in xrange(1, 6):
            for k in xrange(1, len(names)):
                for m in xrange(k):
                    name1, name2 = names[m], names[k]
                    pre = '%s%d' % (i, j)
                    col1, col2 = df_['%s_%s' % (pre, name1)], df_['%s_%s' % (pre, name2)]
                    newcol = combine_two_features(col1, col2, f)
                    df_['%s_%s_%s_%s' % (name1, f_name, name2, pre)] = newcol
    return df_



def add_multiples(df):
    f = lambda x, y: (x + 1) * (y + 1)
    return add_playerwise_combined_features(df, f, 'mult')


def add_divisions(df):
    f = lambda x, y: (x + 1) / (y + 1)
    return add_playerwise_combined_features(df, f, 'div')



def add_team_means(df):
    return add_team_features(df, np.mean, 'mean')



def add_team_medians(df):
    return add_team_features(df, np.median, 'median')



def extract_all(df, df_test, minmax = False):
    '''Preprocesses df and df_test and returns X, y, X_test.'''

    y = target_vector(df)
    
    X      = preprocess(df, train = True)
    X_test = preprocess(df_test, train = False)

    zero_abilities_remover = remove_all_zero(X, 'ability_')
    zero_items_remover     = remove_all_zero(X, 'item_')

    X      = zero_abilities_remover(zero_items_remover(X))
    X_test = zero_abilities_remover(zero_items_remover(X_test))
    
    xnames     = X.columns
    xtestnames = X_test.columns

    scaler = MinMaxScaler() if minmax else StandardScaler()
    
    X, transforms = transform_pipeline(X, [scaler], fit = True)
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
    
    X, y, X_test = extract_all(df, df_test, minmax = True)
    
    X[target_name()] = y

    X['match_id']      = df.match_id
    X_test['match_id'] = df_test.match_id
    
    X.to_csv(train_output, index = False)
    X_test.to_csv(test_output, index = False)





if __name__ == '__main__':
    reply = raw_input('Are you sure you want to create train and test dataframes? (Type Yes or No) ')
    if reply.lower() == 'yes':
        filenames = tuple(map(lambda name: 'data/%s.csv' % name,
                              ['features', 'features_test', 'train', 'test']))
        write_train_test_dataframes(*filenames)
