import numpy as np
import pandas as pd
from feature_bag import get_radiant_players_list, get_dire_players_list, json_to_dataframe




def get_observers(player, type, seconds = 300):
    # names = ['obs_log', 'sen_log']
    cnt = 0
    for log in player[type]:
        cnt += log['time'] <= seconds
    return cnt



def get_observers_row(match, seconds = 300):
    radiant, dire = get_radiant_players_list(match), get_dire_players_list(match)

    types = ['obs_log', 'sen_log']
    row = []
    for t in types:
        row.append(sum(map(lambda p: get_observers(p, t, seconds), radiant)))
        row.append(sum(map(lambda p: get_observers(p, t, seconds), dire)))
    return np.array(row)


OBSERVERS_NAMES = ['obs_log_r', 'obs_log_d', 'sen_log_r', 'sen_log_d']


def json_to_observers_dataframe(filepath):
    return json_to_dataframe(filepath,
                             get_observers_row,
                             OBSERVERS_NAMES)



def write_observers_dataframe(source, destination):
    json_to_observers_dataframe(source).to_csv(destination, index = False)


if __name__ == '__main__':
    reply = raw_input('Are you sure you want to create observers dataframe (It\'s going to take a while)? Type Yes or No? ')
    if reply == 'yes':
        write_observers_dataframe('data/matches.jsonlines.bz2',
                                  'data/observers_train.csv')
        write_observers_dataframe('data/matches_test.jsonlines.bz2',
                                  'data/observers_test.csv')
