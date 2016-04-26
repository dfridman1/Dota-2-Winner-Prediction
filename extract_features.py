#!/usr/bin/env python

import bz2
import json
import pandas
import collections
import argparse
import pandas as pd


def last_value(series, times, time_point=60*5):
    values = [v for t, v in zip(times, series) if t <= time_point]
    return values[-1] if len(values) > 0 else 0


def filter_events(events, time_point=60*5):
    return [event for event in events if event['time'] <= time_point]


CategoryFeatMapper = collections.namedtuple('CategoryFeatMapper',
                                            ['to_id', 'from_id', 'count'])


def category_mapper(ids):
    ids = sorted(ids)
    mapping_from = {x: i for i, x in enumerate(ids)}
    mapping_to = {i: x for i, x in enumerate(ids)}
    return CategoryFeatMapper(lambda x: mapping_from[x],
                              lambda x: mapping_to[x],
                              len(ids))


def create_category_mapper(filename):
    ids = pd.read_csv(filename).id
    return category_mapper(ids)


HEROES_MAPPER = create_category_mapper('data/dictionaries/heroes.csv')
ITEMS_MAPPER = create_category_mapper('data/dictionaries/items.csv')
ABILITIES_MAPPER = create_category_mapper('data/dictionaries/abilities.csv')


def extract_match_features(match, time_point=None):
    extract_items_time = [
        (41, 'bottle'),
        (45, 'courier'),
        (84, 'flying_courier'),
    ]
    extract_items_count = [
        (46, 'tpscroll'),
        (29, 'boots'),
        (42, 'ward_observer'),
        (43, 'ward_sentry'),
    ]

    feats = [
        ('match_id', match['match_id']),
        ('start_time', match['start_time']),
        ('lobby_type', match['lobby_type']),
    ]

    heroes_feats = [0 for _ in xrange(HEROES_MAPPER.count)]
    items_feats = [0 for _ in xrange(2 * ITEMS_MAPPER.count)]
    abilities_feats = [0 for _ in xrange(2 * ABILITIES_MAPPER.count)]
    obs_log = collections.defaultdict(int)
    sen_log = collections.defaultdict(int)

    # player features

    times = match['times']

    for player_index, player in enumerate(match['players']):
        is_radiant = player_index < 5
        player_id = ('r%d' % (player_index+1)) if is_radiant else ('d%d' % (player_index-4))

        purchase_log = filter_events(player['purchase_log'], time_point)
        abilities_upgrades = filter_events(player['ability_upgrades'], time_point)

        feats += [
            (player_id + '_hero', player['hero_id']),
            (player_id + '_level', max([0] + [entry['level'] for entry in filter_events(player['ability_upgrades'], time_point)])),
            (player_id + '_xp', last_value(player['xp_t'], times, time_point)),
            (player_id + '_gold', last_value(player['gold_t'], times, time_point)),
            (player_id + '_lh', last_value(player['lh_t'], times, time_point)),
            (player_id + '_kills', len(filter_events(player['kills_log'], time_point))),
            (player_id + '_deaths', len([
                    1
                    for other_player in match['players']
                    for event in filter_events(other_player['kills_log'], time_point)
                    if event['player'] == player_index   
                ])),
            (player_id + '_items', len(purchase_log)),
        ]

        obs_log['%s_obs_log' % ('r' if is_radiant else 'd')] += len(filter_events(player['obs_log'], time_point))
        sen_log['%s_sen_log' % ('r' if is_radiant else 'd')] += len(filter_events(player['sen_log'], time_point))

        hero_id = HEROES_MAPPER.to_id(player['hero_id'])
        heroes_feats[hero_id] = 1 if is_radiant else -1

        for purchase in purchase_log:
            item_id = ITEMS_MAPPER.to_id(purchase['item_id'])
            offset = 0 if is_radiant else len(items_feats) / 2
            # items_feats[item_id] += (1 if is_radiant else -1)
            items_feats[offset + item_id] += 1

        for upgrade in abilities_upgrades:
            try:
                ability_id = ABILITIES_MAPPER.to_id(upgrade['ability'])
                offset = 0 if is_radiant else len(abilities_feats) / 2
                # abilities_feats[ability_id] += (1 if is_radiant else -1)
                abilities_feats[offset + ability_id] += 1
            except KeyError:
                pass
            

    heroes_feats = [('hero_%d' % i, heroes_feats[i]) for i in xrange(HEROES_MAPPER.count)]
    # items_feats = [('item_%d' % i, items_feats[i]) for i in xrange(ITEMS_MAPPER.count)]
    # abilities_feats = [('ability_%d' % i, abilities_feats[i]) for i in xrange(ABILITIES_MAPPER.count)]
    
    items_names = ['%s_item_%d' % (team, i) for team in 'rd' for i in xrange(ITEMS_MAPPER.count)]
    abilities_names = ['%s_ability_%d' % (team, i) for team in 'rd' for i in xrange(ABILITIES_MAPPER.count)]

    items_feats = zip(items_names, items_feats)
    abilities_feats = zip(abilities_names, abilities_feats)
    
        
    # first blood
    first_blood_objectives = filter_events([obj for obj in match['objectives'] if obj['type'] == 'firstblood'], time_point)
    fb = first_blood_objectives[0] if len(first_blood_objectives) > 0 else {}
    feats += [
        ('first_blood_time', fb.get('time')),
        ('first_blood_team', int(fb['player1'] >= 5) if fb.get('player1') is not None else None),
        ('first_blood_player1', fb.get('player1')),
        ('first_blood_player2', fb.get('player2')),
    ]
    
    # team features
    radiant_players = match['players'][:5]
    dire_players = match['players'][5:]
    
    for team, team_players in (('radiant', radiant_players), ('dire', dire_players)):
        for item_id, item_name in extract_items_time:
            item_times = [
                entry['time']
                for player in team_players
                for entry in filter_events(player['purchase_log'], time_point)
                if entry['item_id'] == item_id
            ]
            first_item_time = min(item_times) if len(item_times) > 0 else None
            feats += [
                ('%s_%s_time' % (team, item_name), first_item_time)
            ]
            
        for item_id, item_name in extract_items_count:
            item_count = sum([
                1
                for player in team_players
                for entry in filter_events(player['purchase_log'], time_point)
                if entry['item_id'] == item_id
            ])
            feats += [
                ('%s_%s_count' % (team, item_name), item_count)
            ]
            
        team_wards = filter_events([
            entry
            for player in team_players
            for entry in (player['obs_log'] + player['sen_log'])
        ], time_point)
        
        feats += [
            ('%s_first_ward_time' % team, min([entry['time'] for entry in team_wards]) if len(team_wards) > 0 else None),
        ]

    if 'finish' in match:
        finish = match['finish']
        feats += [
            ('duration', finish['duration']),
            ('radiant_win', int(finish['radiant_win'])),
            ('tower_status_radiant', finish['tower_status_radiant']),
            ('tower_status_dire', finish['tower_status_dire']),
            ('barracks_status_radiant', finish['barracks_status_radiant']),
            ('barracks_status_dire', finish['barracks_status_dire']),
        ]

    feats += heroes_feats + items_feats + abilities_feats + obs_log.items() + sen_log.items()

    return collections.OrderedDict(feats)


def iterate_matches(matches_filename):
    with bz2.BZ2File(matches_filename) as f:
        for n, line in enumerate(f):
            match = json.loads(line)
            yield match
            if (n+1) % 1000 == 0:
                print 'Processed %d matches' % (n+1)

                
def create_table(matches_filename, time_point):
    df = {}
    fields = None
    for match in iterate_matches(matches_filename):
        features = extract_match_features(match, time_point)
        if fields is None:
            fields = features.keys()
            df = {key: [] for key in fields}    
        for key, value in features.iteritems():
            df[key].append(value)
    df = pandas.DataFrame.from_records(df).ix[:, fields].set_index('match_id').sort_index()
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features from matches data')
    parser.add_argument('input_matches')
    parser.add_argument('output_csv')
    parser.add_argument('--time', type=int, default=5*60)
    args = parser.parse_args()
    
    features_table = create_table(args.input_matches, args.time)
    features_table.to_csv(args.output_csv)
