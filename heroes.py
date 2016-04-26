import numpy as np
import pandas as pd
import argparse
from sklearn.decomposition import TruncatedSVD


def heroes_feature_names(df):
    return [col for col in df.columns if col.startswith('hero_')]


def create_heroes_pairs(df):
    arr = np.array(df)
    m, n = shape = arr.shape
    shape = (m, (n * (n - 1)) >> 1)
    matrix = np.zeros(shape)
    col = 0
    for j in xrange(1, n):
        for i in xrange(j):
            matrix[:, col] = arr[:, i] * arr[:, j]
            col += 1
            if col % 100 == 0:
                print 'processed %d pairs' % col
    return matrix


def reduce_dimensionality(arr, n_components=100, pca=None):
    print 'reducing dimensionality...'
    if pca is None:
        pca = TruncatedSVD(n_components=n_components,
                           random_state=42)
        newarr = pca.fit_transform(arr)
    else:
        newarr = pca.transform(arr)
    print 'finished reducing dimensionality...'
    return newarr, pca


def create_matrix(filename):
    print 'reading dataframe...'
    df = pd.read_csv(filename, index_col='match_id')
    columns = heroes_feature_names(df)
    df = df[columns]
    print 'finished reading reading...'
    return create_heroes_pairs(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_df')
    parser.add_argument('train_output')
    parser.add_argument('test_df')
    parser.add_argument('test_output')
    parser.add_argument('--n_components', type=int, default=100)
    args = parser.parse_args()

    train_m = create_matrix(args.train_df)
    train_pca, pca = reduce_dimensionality(train_m, args.n_components)

    test_m = create_matrix(args.test_df)
    test_pca, pca = reduce_dimensionality(test_m, pca=pca)

    np.savetxt(args.train_output, train_pca, delimiter=',')
    np.savetxt(args.test_output, test_pca, delimiter=',')
