import numpy
import pandas
import matplotlib.pyplot as plt
import statsmodels.api as sm

data = pandas.read_csv("data/shot_logs.csv", low_memory=False)
# data = data[0:100]


def preprocess():
    global data

    del data['SHOT_RESULT']
    del data['W']

    #remove invalid data
    print(len(data[data['TOUCH_TIME'] < 0]))
    print(round(len(data[data['TOUCH_TIME'] < 0]) / float(len(data)), 3) * 100, '%')
    data = data[data['TOUCH_TIME'] > 0]

    print(len(data[data.SHOT_CLOCK.isnull() == True]))
    print(round(float(len(data[data.SHOT_CLOCK.isnull() == True])) / float(len(data)), 2) * 100, '%')
    data = data[data.SHOT_CLOCK.isnull() == False]




def process():
    preprocess()

    # defenders
    defenders = pandas.concat([data['CLOSEST_DEFENDER_PLAYER_ID'], data['CLOSEST_DEFENDER']], axis=1, keys=['PLAYER_ID','PLAYER'])
    defenders = defenders.drop_duplicates()

    # shooters
    shooters = pandas.concat([data['player_id'], data['player_name']], axis=1, keys=['PLAYER_ID','PLAYER'])
    shooters = shooters.drop_duplicates()

    # calculate FG%
    for index, row in shooters.iterrows():
        cur_idx = row['PLAYER_ID']
        shooters.loc[ (shooters['PLAYER_ID'] == cur_idx), 'FGM'] = data[ (data['FGM'] == 1) & (data['player_id'] == cur_idx)]['player_id'].count()

        shooters.loc[ (shooters['PLAYER_ID'] == cur_idx), 'FGA'] = data[ (data['player_id'] == cur_idx)]['player_id'].count()
        shooters.loc[ (shooters['PLAYER_ID'] == cur_idx), 'total_games'] = data[(data['player_id'] == cur_idx)]['GAME_ID'].drop_duplicates().count()

    shooters['FG%'] = shooters['FGM'] / shooters['FGA']

    shooter_map = {}
    for index, row in shooters.iterrows():
        shooter_map[row['PLAYER_ID']] = row['FG%']

    # calculate DFG%
    for index, row in defenders.iterrows():
        cur_idx = row['PLAYER_ID']
        defenders.loc[(defenders['PLAYER_ID'] == cur_idx), 'DFGM'] = data[(data['FGM'] == 1) & (data['CLOSEST_DEFENDER_PLAYER_ID'] == cur_idx)]['player_id'].count()
        defenders.loc[(defenders['PLAYER_ID'] == cur_idx), 'DFGA'] = data[(data['CLOSEST_DEFENDER_PLAYER_ID'] == cur_idx)]['player_id'].count()
        defenders['DFG%'] = defenders['DFGM'] / defenders['DFGA']

        # OFG%
        shooter_dict = {}
        for shooter_idx, shooter_row in shooters.iterrows():
            shooter_id = shooter_row['PLAYER_ID']
            shots_against_player = data[(data['CLOSEST_DEFENDER_PLAYER_ID'] == cur_idx) & (data['player_id'] == shooter_id)]['player_id'].count()
            if shots_against_player > 0:
                shooter_dict[shooter_id] = shots_against_player

        OFG_ratio = 0
        total_shots = defenders[defenders['PLAYER_ID'] == cur_idx]['DFGA']
        for shooter_id, shots in shooter_dict.items():
            OFG_ratio += shots / total_shots * shooters[(shooters['PLAYER_ID'] == shooter_id)].iloc[0]['FG%']
        defenders.loc[(defenders['PLAYER_ID'] == cur_idx), 'OFG%'] = OFG_ratio
    defenders['diff'] = defenders['OFG%'] - defenders['DFG%']
    defender_map = {}
    for index, row in defenders.iterrows():
        defender_map[row['PLAYER_ID']] = row['diff']

    for index, row in data.iterrows():
        data.loc[index, 'DEFENSE_LEVEL'] = defender_map[row['CLOSEST_DEFENDER_PLAYER_ID']]
        data.loc[index, 'OFFENSE_LEVEL'] = shooter_map[row['player_id']]

    diff_df = defenders.sort_values(by='diff', axis=0, ascending=False, inplace=False)
    print(diff_df[(diff_df['DFGA'] >= 100)].head(10))

    shooters.to_csv('shooters.csv')
    defenders.to_csv('defender.csv')

    data.to_csv('out.csv')

    # FG_ratio(shooters)

    # append to original data file


def defender_rank():
    pass


def FG_ratio(players):
    players['FG%_low'], players['FG%_upp'] = sm.stats.proportion_confint(players['FGM'],
                                                                                 players['FGA'],
                                                                                 method='jeffrey')
    players['FG%_low'] = players['FG%'] - players['FG%_low']
    players['FG%_upp'] = players['FG%_upp'] - players['FG%']

    players = players.sort_values('FG%', ascending=False)
    #plot
    plt.figure(figsize=(20, 10))
    #plt.scatter(players.index,players.FGM_ratio.values)
    plt.plot(players['FG%'], 'ko', color='black')
    plt.errorbar(numpy.arange(len(players)), players['FG%'], yerr=[players['FG%_low'], players['FG%_upp']])
    plt.grid()
    plt.ylim(0, 1)
    plt.xlim(-5, 290)
    plt.title('FG% by player')
    plt.ylabel('FG%')
    plt.xlabel('different players')
    plt.show()

    players['avg_FGA_per_game'] = players['FGA'] / players['total_games']
    players['avg_FGM_per_game'] = players['FGM'] / players['total_games']
    players = players.sort_values('avg_FGA_per_game', ascending=False)
    plt.figure(figsize=(20,10))
    plt.plot(players['avg_FGA_per_game'].values, 'ko', color='black')
    plt.plot(players['FGM'].values / players['total_games'].values, 'ko', color='green')
    plt.grid()
    plt.xlim(-5, 290)
    plt.title('players FGA, FGM and FG%')
    plt.ylabel('FGA & FGM')
    plt.xlabel('different players')

    plt.legend(['FGA', 'FGM'], markerscale=2, loc='upper left', prop={'size': 24})

    plt.twinx()

    plt.plot(players['FG%'].values, 'ko', color='red')
    plt.ylim(0, 1)
    plt.yticks(color='red')
    plt.legend(['FG%'], markerscale=2, prop={'size': 24})
    plt.show()


process()