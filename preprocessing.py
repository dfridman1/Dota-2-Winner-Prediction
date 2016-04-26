from sklearn.decomposition import PCA
import numpy as np
import re


class Preprocessor(object):

    def __init__(self, config):
        self.config = config
        self.features = None
        self.team_features = None
        self.pca = None
        self.x_train = None
        self.y_train = None
        self.x_test = None

    def fit_transform(self, X):
        self.y_train = X[self.config.target_name]

        future_features = ['tower_status_dire',
                           'barracks_status_dire',
                           'barracks_status_radiant',
                           'tower_status_radiant',
                           'duration',
                           'radiant_win']
        X = X.drop(future_features, axis=1, errors='ignore')
        X = self._preprocess(X)
        self.features = self._nonconstant_features(X)

        X = X[self.features]
        print 'normalizing features...'
        X = self.config.scaler.fit_transform(X)

        if self.config.n_components:
            self.pca = PCA(n_components=self.config.n_components)
            print 'reducing dimensionality...'
            X = self.pca.fit_transform(X)

        self.x_train = X
        return X

    def transform(self, X):
        X = self._preprocess(X)
        if self.features is not None:
            X = X[self.features]
        print 'normalizing features...'
        X = self.config.scaler.transform(X)

        print 'reducing dimensionality...'
        if self.pca:
            X = self.pca.transform(X)
        return X

    def _preprocess(self, X):
        X = X.fillna(0)

        redundant_features = [col for col in X.columns if 'lobby' in col]
        redundant_features += ['start_time',
                               'first_blood_player1',
                               'first_blood_player2']
        X = X.drop(redundant_features, axis=1)

        for team in 'rd':
            for p in xrange(1, 6):
                X['%s%d_xp_x_gold' % (team, p)] = (X['%s%d_xp' % (team, p)] *
                                                   X['%s%d_gold' % (team, p)])

        self.team_features = self._extract_team_features(X.columns)
        print 'adding medians...'
        X = self._add_team_median(X, self.team_features)
        print 'adding means...'
        X = self._add_team_mean(X, self.team_features)
        print 'adding stddev...'
        X = self._add_team_std(X, self.team_features)

        return X

    def _nonconstant_features(self, df):
        mask = df.apply(np.std) > 0
        return df.columns[mask]

    def _extract_team_features(self, names):
        pat = re.compile('(?:r|d)[0-9]_(.*)')
        matches = filter(None, [re.match(pat, s) for s in names])
        return list(set([m.group(1) for m in matches]))

    def _add_team_aggregate(self, df, team_features, agg, agg_name):
        for feat in team_features:
            for team in 'rd':
                xs = ['%s%d_%s' % (team, p, feat) for p in xrange(1, 6)]
                df['%s_%s_%s' % (team, feat, agg_name)] = df[xs].apply(agg,
                                                                       axis=1)
        return df

    def _add_team_mean(self, df, team_features):
        return self._add_team_aggregate(df, team_features, np.mean, 'mean')

    def _add_team_median(self, df, team_features):
        return self._add_team_aggregate(df, team_features, np.median, 'median')

    def _add_team_std(self, df, team_features):
        return self._add_team_aggregate(df, team_features, np.std, 'std')

    def _add_team_corr(self, df, team_features):
        return self._add_team_aggregate(df, team_features, np.corrcoef, 'corr')
