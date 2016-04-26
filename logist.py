import pandas as pd
import argparse

from config import Config
from preprocessing import Preprocessor
from utils import (
    learning_curve,
    plot_learning_curve)


def logist(config):
    print 'reading training data...'
    df_train = pd.read_csv(config.train_filename,
                           index_col=config.index_col)

    preprocessor = Preprocessor(config)
    X_train = preprocessor.fit_transform(df_train)

    y_train = preprocessor.y_train

    if config.train_model:
        print 'validating logistic regression...'
        gs = config.validate(X_train, y_train, config)
        elapsed = config.validate.elapsed
        print 'best params = ', gs.best_params_
        print 'best score = ', gs.best_score_
        print 'grid scores = ', gs.grid_scores_
        print 'training took %f seconds' % elapsed

    if config.lcurve:
        lc = learning_curve(X_train,
                            y_train,
                            config)
        plot_learning_curve(lc)

    if not config.train_model:
        return

    reply = raw_input('Save to file? Yes or No.\n').lower()
    if reply not in ['yes', 'y']:
        return

    X_test = pd.read_csv(config.test_filename,
                         index_col=config.index_col)
    X_test_match_id = X_test.index
    X_test = preprocessor.transform(X_test)

    predictions = gs.predict_proba(X_test)
    radiant_win = map(lambda x: x[1], predictions)

    print 'writing predictions to %r' % config.output_filename
    output = pd.DataFrame()
    output[config.index_col] = X_test_match_id
    output[config.target_name] = radiant_win
    output.to_csv(config.output_filename, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parser for logistic model')
    parser.add_argument('train_filename')
    parser.add_argument('test_filename')
    parser.add_argument('output_filename')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--C_lo', type=float, default=0.1)
    parser.add_argument('--C_hi', type=float, default=0.1)
    parser.add_argument('--pca', type=int)
    parser.add_argument('--scaler',
                        choices=['standard', 'minmax', 'maxabs'],
                        default='minmax')
    parser.add_argument('--validate',
                        choices=['cv', 'split'],
                        default='cv')
    parser.add_argument('-lcurve', dest='lcurve', action='store_true')
    parser.add_argument('-notrain', dest='notrain', action='store_true')
    args = parser.parse_args()

    config = Config(args.train_filename,
                    args.test_filename,
                    args.output_filename,
                    args.n_folds,
                    args.lcurve,
                    not args.notrain,
                    args.validate,
                    args.pca,
                    args.scaler,
                    args.C_lo,
                    args.C_hi)

    logist(config)
