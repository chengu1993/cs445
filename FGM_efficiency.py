import numpy
import pandas
import matplotlib.pyplot as plt
import statsmodels.api as sm

data = pandas.read_csv("data/shot_logs.csv", low_memory=False)


def count_shots(player_id):
    return len(data[data['player_id'] == player_id])


def count_shots_made(player_id):
    shots_made = data[data['FGM'] == 1]
    return len(shots_made[shots_made['player_id'] == player_id])

def count_game(player_id):
    game = data[data['player_id'] == player_id]
    return len(game.groupby('GAME_ID'))



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

    players = pandas.DataFrame(list(set(data['player_id'])))
    players.columns = ['player_id']

    players['total_attempts'] = players['player_id'].apply(count_shots)
    players['FGM'] = players['player_id'].apply(count_shots_made)
    players['FGM_ratio'] = players['FGM'] / players['total_attempts']
    players['FGM_ratio_low'], players['FGM_ratio_upp'] = sm.stats.proportion_confint(players['FGM'],
                                                                                     players['total_attempts'],
                                                                                     method='jeffrey')

    players['FGM_ratio_low'] = players['FGM_ratio'] - players['FGM_ratio_low']
    players['FGM_ratio_upp'] = players['FGM_ratio_upp'] - players['FGM_ratio']

    players['total_games'] = players['player_id'].apply(count_game)

    players['avg_attempts_per_game'] = players['total_attempts'] / players['total_games']

    players['avg_FGM_per_game'] = players['FGM'] / players['total_games']

    players = players.sort_values('FGM_ratio', ascending=False)

    FGM(players)

    efficiency(players)



def FGM(players):

    #plot
    plt.figure(figsize=(20, 10))
    plt.scatter(players.index,players.FGM_ratio.values)
    plt.plot(players.FGM_ratio.values, 'ko')
    plt.errorbar(numpy.arange(len(players)), players.FGM_ratio.values, yerr=[players['FGM_ratio_low'], players['FGM_ratio_upp']])
    plt.grid()
    plt.ylim(0, 1)
    plt.xlim(-5, 290)
    plt.title('FGM % by player')
    plt.ylabel('FGM %')
    plt.xlabel('different players')
    plt.show()


def efficiency(players):
    print(players.head())

    players = players.sort_values('avg_attempts_per_game', ascending=False)

    plt.figure(figsize=(20,10))
    plt.plot(players['avg_attempts_per_game'].values, 'ko', color='black')
    plt.plot(players['FGM'].values / players['total_games'].values, 'ko', color='green')
    plt.grid()
    plt.xlim(-5, 290)

    plt.title('players atempts, FGM and efficiency')
    plt.ylabel('attempts & FGM')
    plt.xlabel('different players')

    plt.legend(['attempts', 'FGM'], markerscale=2, loc='upper left', prop={'size': 24})

    plt.twinx()

    plt.plot(players['FGM_ratio'].values, 'ko', color='red')
    plt.ylim(0, 1)
    plt.yticks(color='red')
    plt.legend(['efficiency'], markerscale=2, prop={'size': 24})
    plt.show()

process()