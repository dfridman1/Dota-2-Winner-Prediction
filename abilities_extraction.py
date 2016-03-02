import numpy as np
import pandas as pd

from utils import concatMap

from feature_bag import (
    features_map,
    team_bag_row,
    json_to_dataframe
)




abilities_ids = pd.read_csv('data/dictionaries/abilities.csv').id

# some abilities, which are present in train and test sets, are not in abilities dataframe
abilities_ids = xrange(min(abilities_ids),
                       max(abilities_ids) + 1)



ABILITIES_MAP    = features_map(abilities_ids)




def make_ability_names(ids):
    ids = sorted(ids)
    f = lambda c: map(lambda i: 'ability_%s_%d' % (c, i), ids)
    return f('r') + f('d')




ABILITIES_NAMES = make_ability_names(abilities_ids)



def get_abilities(player, seconds = 300):
    res = []
    abilities = player['ability_upgrades']
    for ab in abilities:
        if ab['time'] <= seconds:
            res.append(ABILITIES_MAP[ab['ability']])
    return res





def abilities_bag_row(match):
    return team_bag_row(match, get_abilities, len(abilities_ids))



def json_to_abilities_dataframe(filepath):
    return json_to_dataframe(filepath,
                             abilities_bag_row,
                             ABILITIES_NAMES)




def write_abilities_dataframe(source, destination):
    json_to_abilities_dataframe(source).to_csv(destination, index = False)



    

def is_affirmative(reply):
    return reply.lower() == 'yes'




if __name__ == '__main__':
    reply = raw_input('Are you sure you want to create abilities dataframe (It\'s going to take a while)? Type Yes or No? ')
    if is_affirmative(reply):
        write_abilities_dataframe('data/matches.jsonlines.bz2',
                                  'data/abilities_full_train.csv')
        write_abilities_dataframe('data/matches_test.jsonlines.bz2',
                                  'data/abilities_full_test.csv')
