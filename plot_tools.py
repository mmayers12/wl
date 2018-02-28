import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def set_date_for_plotting(data):
    single_year = True
    years = set(data['date'].str.split(', ', expand=True)[0])

    if len(years) > 1:
        single_year = False
        data['plot_date'] = data['date'].str.replace(', ', '\n')
    else:
        data['plot_date'] = data['date'].str.split(' ', expand=True).iloc[:, -1]
        data['plot_date'] = data['plot_date'].astype(int)

    return single_year, data


def plot_weekly_workouts(log, ax=None):
    workouts = log.get_weekly_workouts()
    pal = sns.color_palette()

    single_year, workouts = set_date_for_plotting(workouts)

    if ax is None:
        fig, ax = plt.subplots()

    sns.barplot(x='plot_date', y='days', data=workouts, color=pal[0], ax=ax)
    ax.set_title('Workouts per week')
    ax.set_ylabel('Days worked out')
    if single_year:
        ax.set_xlabel('Week of the year')
    else:
        ax.set_xlabel('Year and Week')
    ax.set_yticks(np.arange(7))

    if not single_year:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)


def plot_weekly_volume(volume_data, weight, ax=None):
    """

    :return:
    """

    # See if multiple years
    single_year, volume_data = set_date_for_plotting(volume_data)

    if ax is None:
        fig, ax = plt.subplots()

    # Plot the data
    sns.barplot(x='plot_date', y='volume', hue='exercise', data=volume_data, ax=ax)

    ax.set_ylabel('Volume ({})'.format(weight))
    if single_year:
        ax.set_xlabel('Week of the year')
    else:
        ax.set_xlabel('Year and Week')
    ax.legend(title='Exercise')
    ax.set_title('Weekly volume for each exercise')

    if not single_year:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)


def plot_exercise_volume(log, exercises, parent=False, ax=None):

    if type(exercises) == str:
        plot_weekly_volume(log.get_weekly_volume(exercises, parent=parent), log.get_weight(), ax=ax)

    elif type(exercises) == list:
        volume_data = pd.concat([log.get_weekly_volume(ex, parent=parent) for ex in exercises])
        plot_weekly_volume(volume_data, log.get_weight(), ax=ax)


def plot_category_volume(log, ax=None):
    """
    Not yet implemented
    :return:
    """
    if ax is None:
        fig, ax = plt.subplots()

    plot_exercise_volume(log, ['Squat', 'Pull', 'Shoulders'], parent=True, ax=ax)
    ax.legend(title='Exercise Category')
    ax.set_title('Weekly volume for each exercise category')


def dual_axis_plot(data1, data2, unit1, unit2, ax1=None):
    name1 = data1.name
    name2 = data2.name

    data1 = data1.reset_index()
    data2 = data2.reset_index()

    data = pd.merge(data1, data2, how='left', on=['date', 'exercise'])
    data.fillna(method='ffill', inplace=True)

    data['numeric_date'] = data['date'].apply(lambda d: int(d.split(',')[0]) + (int(d.split(' ')[-1]) / 52))

    ex = data.loc[0, 'exercise']

    pal = sns.color_palette()
    if ax1 is None:
        fig, ax1 = plt.subplots()

    ax1.plot(data['numeric_date'], data[name1], c=pal[0])
    ax1.set_xlabel('Date')
    ax1.set_ybound(lower=0)
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Weekly {} ({})'.format(name1.capitalize(), unit1), color=pal[0])
    ax1.tick_params('y', colors=pal[0])

    date_to_string = lambda x: str(int(x)) + '\nWeek ' + str(int((x % 1) * 52))

    ticks = [date_to_string(t) for t in ax1.get_xticks()]
    ax1.set_xticklabels(ticks)

    ax2 = ax1.twinx()

    ax2.plot(data['numeric_date'], data[name2], c=pal[1])
    ax2.set_ylabel('Weekly {} ({})'.format(name2.capitalize(), unit2), color=pal[1])
    ax2.set_ybound(lower=0)
    ax2.tick_params('y', colors=pal[1])
    plt.title("Weely {} and {} for {}".format(name1.capitalize(), name2.capitalize(), ex))
