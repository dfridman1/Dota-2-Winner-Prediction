import numpy as np
import pandas as pd

from utils import concatMap

from feature_bag import (
    features_map,
    team_bag_row,
    json_to_dataframe
)



items_ids = pd.read_csv('data/dictionaries/items.csv').id.unique()



ITEMS_MAP = features_map(items_ids)




def make_items_names(ids):
    ids = sorted(ids)
    f = lambda c: map(lambda i: 'item_%s_%d' % (c, i), ids)
    return f('r') + f('d')



ITEMS_NAMES = make_items_names(items_ids)



def get_items(player, seconds = 300):
    res = []
    purchases = player['purchase_log']
    for p in purchases:
        if p['time'] <= seconds:
            res.append(ITEMS_MAP[p['item_id']])
    return res




def items_bag_row(match):
    return team_bag_row(match, get_items, len(items_ids))



def json_to_items_dataframe(filepath):
    return json_to_dataframe(filepath,
                             items_bag_row,
                             ITEMS_NAMES)


def write_items_dataframe(source, destination):
    json_to_items_dataframe(source).to_csv(destination, index = False)

    


if __name__ == '__main__':
    reply = raw_input('Are you sure you want to create items dataframe (It\'s going to take a while)? Type Yes or No? ')
    if is_affirmative(reply):
        write_items_dataframe('data/matches.jsonlines.bz2',
                              'data/items_train.csv')
        write_items_dataframe('data/matches_test.jsonlines.bz2',
                              'data/items_test.csv')
