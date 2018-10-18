import copy
import numpy as np
import pandas as pd
import datetime
from lifts import lbs_to_kg, kg_to_lbs, calc_1rm, calc_rm_from_1rm


class TrainingLog(object):
    """
    A class to contain all of the training data from a strong .csv export
    """

    def __init__(self, filename, units=None):
        """
        Initialzes training log from a strong .csv export

        :param filename: Name of the .csv to be imported
        :param units: str, either 'kg' or 'lbs'.  Needed for newer versions fo strong export that do not have
            units as a column header
        """
        # Default parent exerciess, in order of priority
        self.parent_exercises = ['Deadlift', 'Pull Up', 'Pull', 'Squat', 'Jerk', 'Row', 'Drive',
                                 'Clean', 'Snatch', 'Push Press', 'Press', 'Assistance']
        self.sub_exercises = {'Lift-Off': 'Pull', 'Balance': 'Push Press'}
        self.filename = filename

        # Make sure units are of appropriate type
        assert units in ['kg', 'lbs', None]
        self.units = units

        self.formula = 'Epley' # use Epley 1rm formula... may add opetions to change later.
        self.new_version = False

        self.data = self.parse_csv(filename)
        self.orig_data = self.data.copy()

    def parse_csv(self, filename):
        """
        Reads the file, a log .csv export from Strong and parses it into a DataFrame

        :param filename: String, names of the file
        :return: Pandas.DataFrame, containing training log
        """

        # Read to DataFrame and do some data normalization
        dat = pd.read_csv(filename)
        dat.columns = dat.columns.str.lower().str.replace(' ', '_')
        dat['date'] = pd.to_datetime(dat['date']).apply(lambda d: d.date())

        # Determine weight type
        if 'kg' in dat:
            self.units = 'kg'
            dat['weight'] = dat['kg']
            dat['lb'] = dat['kg'].apply(kg_to_lbs)
        elif 'lb' in dat:
            self.units = 'lbs'
            dat['weight'] = dat['lb']
            dat['kg'] = dat['lb'].apply(lbs_to_kg)
        elif 'weight' in dat:
            self.new_version = True
            if self.units is None:
                raise ValueError("Could not determine weight type: lb or kg not found in csv columns. " +
                                 "Please units=['kg', lbs'] argument")
            # Set the correct weights to match the units
            if self.units == 'kg':
                dat['lb'] = dat['weight'].apply(kg_to_lbs)
                dat['kg'] = dat['weight']
            elif self.units == 'lbs':
                dat['kg'] = dat['weight'].apply(lbs_to_kg)
                dat['lb'] = dat['weight']
        else:
            raise ValueError("Could not determine weight type: lb, kg, or weight not found in csv columns.")

        # Shoulder Press and Military Press are really just an Overhead Press (or Strict Press in CrossFit)
        dat['exercise_name'] = dat['exercise_name'].str.replace('Strict Military Press', 'Strict Press')
        dat['exercise_name'] = dat['exercise_name'].str.replace('Shoulder Press', 'Strict Press')
        dat['exercise_name'] = dat['exercise_name'].str.replace('Overhead Press', 'Strict Press')
        dat['exercise_name'] = dat['exercise_name'].str.replace('Military Press', 'Strict Press')

        # Clean and Jerk should be 'Clean and Split Jerk' cuz I don't power (and if I did it would be noted)
        dat['exercise_name'] = dat['exercise_name'].str.replace('Clean and Jerk', 'Clean and Split Jerk')

        # Complexes are 2 exercises in one, they should be parsed differently
        # Replace and with + for easy splitting of complexes
        names = dat['exercise_name'].str.replace(' and ', ' + ')
        names = names.str.replace(' Into ', ' + ')

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
        dat['volume'] = self._calculate_volume(dat)

        # Set the order for the columns in the DataFrame
        column_order = ['date', 'workout_name', 'exercise_name', 'kg', 'lb', 'set_order', 'reps',
                        'weight', 'volume', 'ex1', 'ex2', 'p1', 'p2', 's1', 's2', 'notes',
                        'mi', 'seconds']

        # New version no longer has distance units in column headers, instead just 'Distance'
        #TODO Make some kind of unit Determination for distance
        if self.new_version:
            ix = column_order.index('mi')
            column_order[ix] = 'distance'


        return dat[column_order]

    @staticmethod
    def _calculate_volume(df):
        return df['reps'] * df['weight']

    def change_weight_type(self):
        """
        Switches weights reported in data from kg to lbs or vice versa.
        :return: None
        """
        if self.units == 'kg':
            self.data['weight'] = self.data['lb']
            self.data['volume'] = self._calculate_volume(self.data)
            self.units = 'lbs'
        else:
            self.data['weight'] = self.data['kg']
            self.data['volume'] = self._calculate_volume(self.data)
            self.units = 'kg'

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
                          'attempt: {} reps at {} {}'.format(max_reps, max_weight, self.units))

                # Calculate the 1rm
                max_weight = calc_1rm(max_weight, max_reps)
                if reps != 1:
                    # Calcualte the rep max from the predicted 1RM
                    max_weight = calc_rm_from_1rm(max_weight, reps, self.formula)

            # Not Calculating, so just print out how many reps this weight was achieved at.
            else:
                if verbose:
                    print('Reps at max weight: {}'.format(max_reps))

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

        # Find out if we have a sub exercise first
        for ex, parent in self.sub_exercises.items():
            if ex in exercise:
                return parent

        # If not a sub exercise, then find the parent
        for ex in self.parent_exercises:
            if ex in exercise:
                return ex

    def get_subtype(self, exercise):
        """
        Determines the sub-category for an exercise

        :param exercise: String, the name of the exercise
        :return:
        """

        # No subtype if no exercise
        if exercise is None:
            return exercise

        # No subtype if it is the parent version of the exercise
        if exercise in self.parent_exercises:
            return None

        # Find the parent, remove it, and the remainder of the name is the subtype
        for par in self.parent_exercises:
            if par in exercise:
                # Subtype label can either lead or trial exercise name
                out = exercise.split(par)
                if out[0]:
                    return out[0].strip(' ')
                elif out[-1]:
                    return out[-1].strip(' ')
                else:
                    return None

    @staticmethod
    def _group_by_week(df):
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

    @staticmethod
    def _fill_missing_weeks(data):
        # if any weeks are missing, fill them in with zeros.
        first_date = data.iloc[0]['date']
        last_date = data.iloc[-1]['date']

        date_range = []

        for y in range(first_date[0], last_date[0]+1):
            for w in range(1, 53):
                if y == first_date[0] and w < first_date[1]:
                    continue
                elif y == last_date[0] and w > last_date[1]:
                    break
                date_range.append((y, w))

        all_dates = pd.DataFrame({'date': date_range})
        return pd.merge(all_dates, data, how='left', on='date').fillna(0)

    def weekly_exercise_information(self, exercise, parent, agg_func, fill_missing=True, add_exercise=True):

        if exercise is not None and parent is not None:
            filtered = self.filter_exercises(exercise, parent)
            grouped = self._group_by_week(filtered)
        else:
            grouped = self._group_by_week(self.data)

        data = agg_func(grouped)

        if fill_missing:
            data = self._fill_missing_weeks(data)

        if add_exercise:
            data['exercise'] = exercise

        data['date'] = data['date'].apply(lambda x: '{}, Week {}'.format(x[0], x[1]))

        return data.reset_index(drop=True)

    def get_weekly_volume(self, exercise, parent=True):

        def agg_function(grouped):
            volume = (grouped.sum()['volume']
                             .to_frame()
                             .reset_index())
            return volume

        return self.weekly_exercise_information(exercise, parent, agg_function)

    def get_weekly_workouts(self):

        def agg_function(grouped):
            workouts = (grouped['date'].nunique()
                                       .rename('days')
                                       .to_frame()
                                       .reset_index())
            return workouts

        return self.weekly_exercise_information(None, None, agg_function, add_exercise=False)

    def calc_highest_1rm(self, data, formula='Epley'):
        return max([calc_1rm(row.weight, row.reps, formula=formula) for row in data.itertuples()])

    def get_weekly_1rm(self, exercise, parent=True, formula='Epley'):

        # Ensure there's a calculated 1rm in the data for each entry
        self.data['calc_1rm'] = self.data.apply(lambda row: calc_1rm(row['weight'], row['reps'], formula), axis=1)

        def agg_function(grouped):
            calcd_1rm = (grouped.max()['calc_1rm']
                                .to_frame()
                                .reset_index())
            return calcd_1rm

        return self.weekly_exercise_information(exercise, parent, agg_function, fill_missing=False)

    def get_weekly_top_set_weight(self, exercise, parent=True):

        def agg_function(grouped):
            top_set = (grouped.max()['weight']
                              .to_frame()
                              .reset_index()
                              .rename(columns={'weight': 'top_set'}))
            return top_set
        return self.weekly_exercise_information(exercise, parent, agg_function)

    def get_weekly_intensity(self, exercise, parent=True):
        # Get 1RMs for each week
        one_rm = self.get_weekly_1rm(exercise, parent)

        # Assuming no De-training, 1rms shouldn't go down over time, so get cumulative 1rm:
        curr_max = 0
        for row in one_rm.itertuples():
            if row.calc_1rm > curr_max:
                one_rm.loc[row.Index, 'cum_max'] = row.calc_1rm
                curr_max = row.calc_1rm
            else:
                one_rm.loc[row.Index, 'cum_max'] = curr_max

        # Get the top set info for that week to find intensity
        top_set = self.get_weekly_top_set_weight(exercise, parent)
        intensity = pd.merge(top_set, one_rm, how='left', on=['date', 'exercise'])
        intensity['intensity'] = intensity['top_set'] / intensity['cum_max'].fillna(method='ffill')
        return intensity

    def get_data(self):
        return self.data

    def get_weight(self):
        return self.units

    def set_file(self, filename):
        self.filename = filename
        self.data = self.parse_csv(filename)
        self.orig_data = self.data.copy()

    @staticmethod
    def _format_date(date):
        if type(date) == datetime.datetime:
            return date.date()
        elif type(date) == str:
            d = date.__str__().split(' ')[0]
            d_split = [int(x) for x in d.split('-')]
            return datetime.date(d_split[0], d_split[1], d_split[2])
        else:
            raise TypeError('Dates must be given as either "YYYY-MM-DD" formatted strings, ' +
                            'datetime.date, datetime.datetime')

    def copy(self):
        return copy.copy(self)

    def set_date_range(self, start=datetime.date(1900, 1, 1), end=datetime.date(2100, 12, 31), inplace=False):
        """
        Sets a new date range for the data
        :param start:
        :param end:
        :param inplace:
        :return:
        """
        # Make sure dates are date objects or pandas will throw error
        if type(start) != datetime.date:
            start = self._format_date(start)
        if type(end) != datetime.date:
            end = self._format_date(end)

        # Format the query string
        q_str = 'date >= {!r} and date <= {!r}'.format(start, end)

        if inplace:
            # Set the new date range
            self.data = self.orig_data.query(q_str).reset_index(drop=True)
        else:
            out = self.copy()
            out.data = out.orig_data.query(q_str).reset_index(drop=True)
            return out

    def find_cycle_prs(self, cycle_start_date, cycle_end_date='2100-12-31', exercises=None,
                       rep_range=range(1, 13), previous_pr_start_date='1900-01-01'):
        """
        Routine to find all PRs attained during a training cycle. Will return both new rep-scheme PRs as well as
        absolute PRs. Boolean column 'absolute_max' will be True if this is the most weight lifted for a given
        exercise at any rep-range.

        :param cycle_start_date: str, datetime.date, or datetime.datetime, the start date for the cycle. Strings must
            be formatted as YYYY-MM-DD
        :param cycle_end_date: the end date for the training cycle. If none given, will use till end of training log.
            See `cycle_start_date` for foramming options
        :param exercises: list, names of exercises to check for PRs.  If none give, all exercises will be checked.
        :param rep_range: list or range of ints, rep numbers to check for PRs on.
        :param previous_pr_start_date: Start date to check for previous PRs. For example, if you got sick and only
            want to see progress since date returning from illness, you can limit this before time for PR comparison.
            see `cycle_start_date` for formatting options

        :return: DataFrame, exercises with PRs will be returned with old and new weight and rep values.
        """

        # Cut dates for before and after the split
        before_data = self.set_date_range(start=previous_pr_start_date, end=cycle_start_date)
        after_data = self.set_date_range(start=cycle_start_date, end=cycle_end_date)

        # Most useful column order for output
        col_order = ['exercise', 'reps', 'previous_max', 'new_max', 'weight_increase', 'rep_increase', 'new_reps',
                     'absolute_max']

        # Find exercises logged both before and after the split
        common_exercises = (set(before_data.get_data()['ex1'].unique()) |
                            set(before_data.get_data()['ex2'].dropna().unique()) &
                            set(after_data.get_data()['ex1'].unique()) |
                            set(after_data.get_data()['ex2'].dropna().unique()))

        # If no exercises supplied, use all common
        if exercises is None:
            exercises = sorted(list(common_exercises))
        else:
            # Otherwise ensure that there is overlap for these exercises...
            exercises = sorted(list(set(exercises) & common_exercises), key=lambda e: exercises.index(e))

        # Grab all maxes for given rep ranges from before split date
        before_maxes = []
        for ex in exercises:
            for num_reps in rep_range:
                rep_max = before_data.get_exercise_max(ex, num_reps, calculate=False, verbose=False)
                before_maxes.append({'exercise': ex, 'reps': num_reps, 'previous_max': rep_max})
        before_maxes = pd.DataFrame(before_maxes)

        # And grab all for after split date
        after_maxes = []
        for ex in exercises:
            for num_reps in rep_range:
                rep_max = after_data.get_exercise_max(ex, num_reps, calculate=False, verbose=False)
                after_maxes.append({'exercise': ex, 'reps': num_reps, 'new_max': rep_max})
        after_maxes = pd.DataFrame(after_maxes)

        # Merge exercises and reps so new weights for a given rep-scheme are paired
        changes = pd.merge(before_maxes, after_maxes, on=['exercise', 'reps'])
        # Only want the ones where the weight went up
        increases = changes.query('new_max > previous_max')

        # If no increases, return an empty DataFrame
        if increases.shape[0] < 1:
            return pd.DataFrame(columns=col_order)

        # Find out which ones are new maxes for the exercise at any rep scheme
        absolute_inc = increases.query('reps == 1')[['exercise', 'previous_max', 'new_max']]
        absolute_inc['absolute_max'] = True

        # Gets rid of cases where before and after 3RM == 4RM == 5RM and all increased
        # will only keep the 5RM
        all_prs = increases.drop_duplicates(subset=['exercise', 'new_max', 'previous_max'], keep='last')

        # Merge in the absolute maxes and label non-absolute maxes as such
        all_prs = all_prs.merge(absolute_inc, on=['exercise', 'new_max', 'previous_max'], how='left')
        all_prs['absolute_max'] = all_prs['absolute_max'].fillna(False)

        # Get Increaeses
        all_prs['weight_increase'] = all_prs['new_max'] - all_prs['previous_max']

        # Get rid of duplications where new weight lifted for more reps
        # e.g. previous 3rm = 60kg, previous 4RM == 5RM = 56kg,
        #     New 4RM = 62kg and new 5RM == 59kg.
        #     Only want to know 3RM @ 60 -> 4RM at 62, and 5RM @ 56 -> 5RM @ 59.
        #     Don't care about 4RM @ 56 -> 4RM @ 62, cuz that is redundant

        # Get the max weight for a number of reps
        best_reps = all_prs.groupby(['exercise', 'new_max'])['reps'].max().rename('new_reps').reset_index()
        # Merge the new rep values in, and drop cases
        all_prs = all_prs.merge(best_reps, on=['exercise', 'new_max'])
        all_prs = all_prs.drop_duplicates(subset=['exercise', 'new_max', 'new_reps']).reset_index(drop=True)

        # flag for if reps increased as well as weight
        all_prs['rep_increase'] = all_prs['new_reps'] > all_prs['reps']

        return all_prs[col_order]


class WeightliftingLog(TrainingLog):
    def __init__(self, filename,  units=None):
        TrainingLog.__init__(self, filename, units)

    def filter_exercises(self, exercise, parent=True):
        # Query for Ex
        # Special case for Pulls
        if exercise == 'Pull':
            pulls = ['Pull', 'Snatch', 'Clean', 'Deadlift']
            filtered = self.data.query('p1 in {0} | p2 in {0}'.format(pulls))
        # Special Case for Shoulder to Overhead type movements
        elif exercise == 'Shoulders':
            oh = ['Jerk', 'Press', 'Push Press']
            filtered = self.data.query('p1 in {0} | p2 in {0}'.format(oh))
        else:
            return TrainingLog.filter_exercises(self, exercise, parent)
        return filtered
