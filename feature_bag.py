import json
import bz2
import numpy as np
import pandas as pd
from utils import concatMap



def get_players_list(match):
    return match['players']




def get_radiant_players_list(match):
    return get_players_list(match)[:5]




def get_dire_players_list(match):
    return get_players_list(match)[5:]



def features_map(ids):
    ids = sorted(ids)
    return {x: i for i, x in enumerate(ids)}




def bag_empty_row(size):
    return np.zeros(size)



def team_bag_row(match, get, N):
    row = bag_empty_row(2 * N)

    radiant_team = get_radiant_players_list(match)
    dire_team    = get_dire_players_list(match)

    radiant_features = concatMap(get, radiant_team)
    dire_features    = concatMap(get, dire_team)

    def populate(features, radiant = True):
        offset = 0 if radiant else N
        for x in features:
            row[offset + x] += 1

    populate(radiant_features)
    populate(dire_features, radiant = False)

    return row




def json_to_dataframe(filepath,
                      features_bag_row,
                      features_names):
    features, ids = None, []
    cnt = 0
    with bz2.BZ2File(filepath) as matches_file:
        for line in matches_file:
            match = json.loads(line)
            ids.append(match['match_id'])
            if features is None:
                features = features_bag_row(match)
            else:
                features = np.vstack([features, features_bag_row(match)])
            cnt += 1
            if cnt % 100 == 0:
                print 'processed %d matches' % cnt
        data = np.hstack([np.reshape(ids, (len(ids), 1)), features])
        colnames = np.hstack(['match_id', features_names])
        return pd.DataFrame(data, columns = colnames)
