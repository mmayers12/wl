import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


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
        dat['volume'] = dat['reps'] * dat['kg']

        return dat


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

        # No subtyoe if no exercise
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
        year_week = lambda x: tuple(x.isocalendar()[:2])
        grouped = df.groupby(df['date'].map(year_week))
        return grouped

    def filter_exercises(self, exercise, parent=True):

        # Go for the specific exercise, not the parent category of exercises
        if not parent:
            filtered = self.data.query('ex1 == "{0}" | ex2 == "{0}"'.format(exercise))
        else:
            filtered = self.data.query('p1 == "{0}" | p2 == "{0}"'.format(exercise))

        return filtered


    def get_weekly_volume(self, exercise, parent=True):

        filtered = self.filter_exercises(exercise, parent)
        grouped = self.group_by_week(filtered)

        vol = (grouped
                 .sum()['volume']
                 .to_frame()
                 .reset_index())

        vol['exercise'] = exercise
        vol['date'] = vol['date'].apply(lambda x: '{}, Week {}'.format(x[0], x[1]))

        return vol.reset_index()





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

    def plot_category_volume():
        ps = pd.concat(
            [get_weekly_volume('Squat'),
             get_weekly_volume('Pull'),
             get_weekly_volume('Shoulders')])

        # See if multiple years
        single_year = True
        years = set(ps['date'].str.split(', ', expand=True)[0])

        if len(years) > 1:
            single_year = False
            ps['date'] = ps['date'].str.replace(', ', '\n')
        else:
            ps['date'] = ps['date'].str.split(' ', expand=True).iloc[:, -1]
            ps['date'] = ps['date'].astype(int)

        sns.barplot(x='date', y='volume', hue='exercise', data=ps)
        # plt.xticks(rotation='vertical')
        plt.ylabel('Volume (kg)')
        if single_year:
            plt.xlabel('Week of the year')
        else:
            plt.xlabel('Year and Week')
        plt.legend(title='Exercise Category')
        plt.title('Weekly volume for each exercise type');