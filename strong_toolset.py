import numpy as np
import pandas as pd
from datetime import datetime
from lifts import lbs_to_kg, kg_to_lbs, calc_1rm, calc_rm_from_1rm
import matplotlib.pyplot as plt
import seaborn as sns


class TrainingLog(object):
    """
    A class to contain all of the training data from a strong .csv export
    """

    def __init__(self, filename):
        """
        Initialzes training log from a strong .csv export

        :param filename: Name of the .csv to be imported
        """
        # Default parent exerciess, in order of priority
        self.parent_exercises = ['Deadlift', 'Pull Up', 'Pull', 'Squat', 'Jerk', 'Clean', 'Snatch', 'Press']
        self.filename = filename
        self.weight = None
        self.formula = 'Epley' # use Epley 1rm formula... may add opetions to change later.

        self.data = self.parse_csv(filename)

    def parse_csv(self, filename):
        """
        Reads the file, a log .csv export from Strong and parses it into a DataFrame

        :param filename: String, names of the file
        :return: Pandas.DataFrame, containing training log
        """

        # Read to DataFrame and do some data normalization
        dat = pd.read_csv(filename)
        dat.columns = dat.columns.str.lower().str.replace(' ', '_')
        dat['date'] = pd.to_datetime(dat['date'])

        # Determine weight type
        # TODO MAKE this so it only uses csv type on first import, not on changing date range
        if 'kg' in dat:
            self.weight = 'kg'
            dat['weight'] = dat['kg']
            dat['lb'] = dat['kg'].apply(kg_to_lbs)
        elif 'lb' in dat:
            self.weight = 'lbs'
            dat['weight'] = dat['lb']
            dat['kg'] = dat['lb'].apply(lbs_to_kg)
        else:
            raise ValueError("Could not determine weight type: lb or kg not found in csv columns")

        # Shoulder Press and Military Press are really just an Overhead Press (or Strict Press in CrossFit)
        dat['exercise_name'] = dat['exercise_name'].str.replace('Shoulder Press', 'Strict Press')
        dat['exercise_name'] = dat['exercise_name'].str.replace('Military Press', 'Strict Press')

        # Complexes are 2 exercises in one, they should be parsed differently
        # Replace and with + for easy splitting of complexes
        names = dat['exercise_name'].str.replace('and', '+')
        names = names.str.replace('Into', '+')

        # 'Shrug + Clean' is really 'Clean Pull + Clean', however, don't overwrite standard barbell shrugs
        names[names.str.contains('Clean')] = names[names.str.contains('Clean')].str.replace('Shrug', 'Clean Pull')
        # More normalization
        names = names.str.replace('OHS', 'Overhead Squat')
        names = names.str.replace('Over Head Squat', 'Overhead Squat')

        # Some exercises had notes written in parenthesis after the exercise
        # Add the notes to the notes
        notes = names.str.split('(', expand=True)[1]
        notes = notes.str.strip(')')
        dat['notes'].fillna(notes, inplace=True)
        # Remove the notes from the exercise name
        names = names.str.split('(', expand=True)[0]
        names = names.str.strip()

        # Split out the complexes
        split_df = names.str.split('+', expand=True)
        dat['ex1'] = split_df[0]
        dat['ex2'] = split_df[1]

        # Strip any whitespace that may be created by the split
        dat['ex1'] = dat['ex1'].str.strip()
        dat['ex2'] = dat['ex2'].str.strip()

        # Get parent exercises and exercise subtypes
        dat['p1'] = dat['ex1'].apply(self.get_parent)
        dat['p2'] = dat['ex2'].apply(self.get_parent)
        dat['s1'] = dat['ex1'].apply(self.get_subtype)
        dat['s2'] = dat['ex2'].apply(self.get_subtype)

        # Calculate volume
        dat['volume'] = self.calculate_volume(dat)

        # Set the order for the columns in the DataFrame
        column_order = ['date', 'workout_name', 'exercise_name', 'kg', 'lb', 'set_order', 'reps',
                        'weight', 'volume', 'ex1', 'ex2', 'p1', 'p2', 's1', 's2', 'notes',
                        'mi', 'seconds']

        return dat[column_order]

    def calculate_volume(self, df):
        return df['reps'] * df['weight']

    def change_weight_type(self):
        """
        Switches weights reported in data from kg to lbs or vice versa.
        :return: None
        """
        if self.weight == 'kg':
            self.data['weight'] = self.data['lb']
            self.data['volume'] = self.calculate_volume(self.data)
            self.weight = 'lbs'
        else:
            self.data['weight'] = self.data['kg']
            self.data['volume'] = self.calculate_volume(self.data)
            self.weight = 'kg'

    def get_exercise_max(self, exercise, reps=1, parent=False, calculate=True, verbose=True):
        """
        Get the maximum weight lifted for a given exercise

        :param exercise: str, the name of the exercise
        :param reps: int, the rep max to be found
        :param parent: bool, look at parent exercise class rather than specific exercises
        :param calculate: bool, calculate the rm if the number of reps has never been performed.
        :param verbose: bool, print out information when rep max isn't in database 
        :return: float, the max weight lifted for the given number of reps
        """

        filtered = self.filter_exercises(exercise, parent)

        max_weight = filtered.query('reps >= {}'.format(reps))['weight'].max()
        max_reps = filtered.query('weight == @max_weight')['reps'].max()

        if max_reps > reps:

            if verbose:
                print('Data not found for {} at reps={}.'.format(exercise, reps))

            if calculate:
                # Because formulas are more accurate on lower rep-counts of 3 or greater,
                # we will see if we can use a lower weight.
                if reps > 3:
                    new_max_weight = filtered.query('reps < {} and reps > 2'.format(reps))['weight'].max()
                    # Ensure the query worked before updating.
                    if not np.isnan(new_max_weight):
                        max_weight = new_max_weight
                        max_reps = filtered.query('weight == @max_weight')['reps'].max()

                if verbose:
                    print('Calculating {} rep max from best '.format(reps) +
                          'attempt: {} reps at {} {}'.format(max_reps, max_weight, self.weight))

                # Calculate the 1rm
                max_weight = calc_1rm(max_weight, max_reps)
                if reps != 1:
                    # Calcualte the rep max from the predicted 1RM
                    max_weight = calc_rm_from_1rm(max_weight, reps, self.formula)

        return max_weight

    def set_parent_exercises(self, parents):
        """
        Sets the parent exercies, then applies the changes to the data. Must be in
        order of priority

        :param parents: List of Strings, the names of the exercies that are parents
        :return: None
        """
        # Set the new list of parents
        self.parent_exercises = parents

        # Apply the changes to parents and subtypes
        self.data['p1'] = self.data['ex1'].apply(self.get_parent)
        self.data['p2'] = self.data['ex2'].apply(self.get_parent)
        self.data['s1'] = self.data['ex1'].apply(self.get_parent)
        self.data['s2'] = self.data['ex2'].apply(self.get_parent)

    def get_parent(self, exercise):
        """
        Determines the parent class of an exercise from a modified version
        e.g. "Romanian Deadlift" is a "Deadlift", "Overhead Squat" is a "Squat"

        :param exercise: String, the name of the exercise to have its parent extracted
        :return: String, the parent exercise
        """

        # In order of Priority.. i.e. a 'Snatch Deadlift' is a 'Deadlift' and not a 'Snatch'
        # A 'Clean Pull' is a Pull, not a Clean, and a 'Pull Up' is it's own thing

        if exercise is None:
            return exercise

        for ex in self.parent_exercises:
            # Find out which parent name is in the exercise
            if ex in exercise:
                return ex

    def get_subtype(self, exercise):

        # No subtype if no exercise
        if exercise is None:
            return exercise

        # Find the parent, remove it, and the remainder of the name is the subtype
        for par in self.parent_exercises:
            if par == exercise:
                return None
            elif par in exercise:
                # Subtype label can either lead or trial exercise name
                out = exercise.split(par)
                if out[0]:
                    return out[0].strip(' ')
                elif out[-1]:
                    return out[-1].strip(' ')
                else:
                    return None

    def group_by_week(self, df):
        # Group by week
        week_of_the_year = lambda x: tuple(x.isocalendar()[:2])
        grouped = df.groupby(df['date'].map(week_of_the_year))
        return grouped

    def filter_exercises(self, exercise, parent=True):

        # Go for the specific exercise, not the parent category of exercises
        if parent:
            kind = 'p'
        else:
            kind = 'ex'

        if ' and ' in exercise:
            exs = exercise.split(' and ')
            filtered = self.data.query('{0}1 == "{1}" & {0}2 == "{2}"'.format(kind, exs[0], exs[1]))
        else:
            filtered = self.data.query('{0}1 == "{1}" | {0}2 == "{1}"'.format(kind, exercise))

        return filtered

    def get_weekly_volume(self, exercise, parent=True):

        filtered = self.filter_exercises(exercise, parent)
        grouped = self.group_by_week(filtered)

        vol = (grouped.sum()['volume']
               .to_frame()
               .reset_index())

        vol['exercise'] = exercise
        vol['date'] = vol['date'].apply(lambda x: '{}, Week {}'.format(x[0], x[1]))

        return vol.reset_index(drop=True)

    def get_weekly_workouts(self):

        grouped = self.group_by_week(self.data)

        workouts = (grouped['date'].nunique()
                                   .rename('days')
                                   .to_frame()
                                   .reset_index())

        workouts['date'] = workouts['date'].apply(lambda x: '{}, Week {}'.format(x[0], x[1]))

        return workouts.reset_index(drop=True)


    def find_1rm_over_time(self, data, formula='Epley'):
        return max([calc_1rm(row.weight, row.reps, formula=formula) for row in data.itertuples()])

    def get_weekly_1rm(self, exercise, parent=True, formula='Epley'):

        filtered = self.filter_exercises(exercise, parent)
        grouped = self.group_by_week(filtered)

        data = {'maxes': [], 'date': []}
        for label, week in grouped:
            data['maxes'].append(self.find_1rm_over_time(week, formula=formula))
            data['date'].append('{}, Week {}'.format(label[0], label[1]))

        data = pd.DataFrame(data)
        data['exercise'] = exercise

        return data

    def get_data(self):
        return self.data

    def set_file(self, filename):
        self.filename = filename
        self.data = self.parse_csv(filename)

    def set_date_range(self, start=datetime(1900, 1, 1), end=datetime(2100, 12, 31)):
        """
        Sets a new date range for the data
        :param start:
        :param end:
        :return:
        """
        def format_date(date):
            d = date.__str__().split(' ')[0]
            d_split = [int(x) for x in d.split('-')]
            return datetime(d_split[0], d_split[1], d_split[2])

        dat = self.parse_csv(self.filename)

        # Make sure dates are datetime objects or pandas will throw error
        if type(start) == str:
            start = format_date(start)
        if type(end) == str:
            end = format_date(end)

        # Format the query string
        q_str = 'date >= {!r} and date < {!r}'.format(start, end)
        # Set the new date range
        self.data = dat.query(q_str)

    def plot_exercise_volume(self, exercises, parent=False):

        if type(exercises) == str:
            self.plot_weekly_volume(self.get_weekly_volume(exercises, parent=parent))

        elif type(exercises) == list:
            volume_data = pd.concat([self.get_weekly_volume(ex, parent=parent) for ex in exercises])
            self.plot_weekly_volume(volume_data)

    def plot_weekly_volume(self, volume_data):
        """

        :return:
        """

        # See if multiple years
        single_year = True
        years = set(volume_data['date'].str.split(', ', expand=True)[0])

        if len(years) > 1:
            single_year = False
            volume_data['date'] = volume_data['date'].str.replace(', ', '\n')
        else:
            volume_data['date'] = volume_data['date'].str.split(' ', expand=True).iloc[:, -1]
            volume_data['date'] = volume_data['date'].astype(int)

        # Plot the data
        sns.barplot(x='date', y='volume', hue='exercise', data=volume_data)
        # plt.xticks(rotation='vertical')
        plt.ylabel('Volume ({})'.format(self.weight))
        if single_year:
            plt.xlabel('Week of the year')
        else:
            plt.xlabel('Year and Week')
        plt.legend(title='Exercise')
        plt.title('Weekly volume for each exercise')


class WeightliftingLog(TrainingLog):
    def __init__(self, filename):
        TrainingLog.__init__(self, filename)

    def filter_exercises(self, exercise, parent=True):
        # Query for Ex
        # Special case for Pulls
        if exercise == 'Pull':
            pulls = ['Pull', 'Snatch', 'Clean', 'Deadlift']
            filtered = self.data.query('p1 in {0} | p2 in {0}'.format(pulls))
        # Special Case for Shoulder to Overhead type movements
        elif exercise == 'Shoulders':
            oh = ['Jerk', 'Press']
            filtered = self.data.query('p1 in {0} | p2 in {0}'.format(oh))
        else:
            return TrainingLog.filter_exercises(self, exercise, parent)
        return filtered

    def plot_category_volume(self):
        """
        Not yet implemented
        :return:
        """

        self.plot_exercise_volume(['Squat', 'Pull', 'Shoulders'], parent=True)
        plt.legend(title='Exercise Category')
        plt.title('Weekly volume for each exercise category')
