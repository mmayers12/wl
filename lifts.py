import numpy as np

CONV_FACT = 2.20462262


def percentage(percent, maximum):
    """
    Calculates the percentage for a given maximum.
    Returns a whole number, rounded to the nearest integer.

    :param percent: number, the percentage to be caluclated
    :param maximum: number, the maximum to take the percentage of
    :return: int, the percentage of that maximum
    """
    return int(round(maximum*percent / 100))


def calculate_percentages(maximum, percentages=None):
    """
    Calculates percentages for a lift when given a max

    :param maximum: the current maximum for the lift
    :param percentages: optional paramater, list of percentages to be calculated
    """

    if not percentages:
        percentages = [i for i in range(40, 100, 5)]

    for percent in percentages:
        print("\t{0:>3}%\t-{1:>7}".format(percent, percentage(percent, maximum)))


def kg_to_lbs(weight):
    """
    Converts a weight value to from kilogram units to pound units.  Rounds to the nearest 100th.

    :param weight: Int or float, the weight in kilograms to be converted
    :return: Float, the weight in pounds, rounded to the nearest hundredth of a pound
    """
    # Convert
    lbs = weight * CONV_FACT
    # Round to the nearest 100th
    return special_round(lbs)


def lbs_to_kg(weight):
    """
    Converts a weight value to from pound units to kilogram units.  Rounds to the nearest 100th.

    :param weight: Int or float, the weight in pounds to be converted
    :return: Float, the weight in kilograms, rounded to the nearest hundredth of a kilo
    """
    # Convert
    kgs = weight / CONV_FACT
    # Round to the nearest 100th
    return special_round(kgs)


def special_round(number):
    """
    Special round that helps retain precision of values that were orignially recorded in lbs,
    converted to kg, and are now being back converted to lbs.
    Certain numbers are mis-converted with a simple round: like 117.94 -> 260.01 or 138.35 -> 305.006 -> 305.01

    :param number: float, the number to be rounded with special conversion
    :return: float, the number rounded to 2 decimal places via the special formula
    """

    decimal = int(number * 100) % 100

    # Numbers ending in .00 or .50 should be returned the same regardless trailing digits
    if decimal == 00 or decimal == 50:
        return int(number) + float(decimal/100)
    # Numbers ending in .01 or .51 should be returned as .00 or .50
    elif decimal == 1 or decimal == 51:
        return int(number) + float((decimal-1) / 100)
    # Numbers ending in .99 or .49 should be rounded to .00 or .50
    elif decimal == 99 or decimal == 49:
        return int(number) + float((decimal+1) / 100)
    # If not one of these special cases, do a simple round
    else:
        return round(number, 2)


def calc_1rm(weight, reps, formula='Epley'):
    """
    Calculates a 1RM from given weight and reps. Different published formulas may be used.
    
    According to LeSeur et. al. (doi:10.1519/00124278-199711000-00001) the best formulas per exercise are as follows:
        Bench Press: Mayhew and Wathan
        Squat: Wathan
        Deadlift: There All tend to underestimate by at least 10%
        
    Formulas perform best in the <10RM range at loads of 85% or more.
    
    :param weight: int, weight lifed
    :param reps: int, number of reps attained
    :param formula: str, the formula used to calulate the 1rm: Currently supports the following formulas: 
        'Epley', 'Brzycki', 'McGlothin', 'Lombardi', 'Mayhew', 'OConner', 'Wathan'. 
        See https://en.wikipedia.org/wiki/One-repetition_maximum
     
    :return: float, calcualted 1RM weight. 
    """

    if  reps == 1:
        return weight

    formulas = ['Epley', 'Brzycki', 'McGlothin', 'Lombardi', 'Mayhew', 'OConner', 'Wathan']

    if formula not in formulas:
        raise ValueError('Formula must be one of {!r}'.format(formulas))

    if formula == 'Epley':
        return weight*(1+(reps/30))
    if formula == 'Brzycki':
        return weight*(36 / (37 - reps))
    if formula == 'McGlothin':
        return (100*weight) / (101.3 - 2.67123*reps)
    if formula == 'Lombardi':
        return weight*(reps**0.1)
    if formula == 'Mayhew':
        return (100*weight) / (52.2+ 41.9*np.exp(-0.055*reps))
    if formula == 'OConner':
        return weight*(1 + (reps / 40))
    if formula == 'Wathan':
        return (100 * weight) / (48.8 + 53.8*np.exp(-0.075*reps))


def calc_rm_from_1rm(one_rm, reps, formula='Epley'):
    """
    Given a 1RM and a number of reps, calculates the maximum one_rm that should be attainable for those reps.
    Essentially the inverse calculation of calc_1rm.
    
    According to LeSeur et. al. (doi:10.1519/00124278-199711000-00001) the best formulas per exercise are as follows:
        Bench Press: Mayhew and Wathan
        Squat: Wathan
        Deadlift: There All tend to underestimate by at least 10%
        
    Formulas perform best in the <10RM range at loads of 85% or more.
    
    :param one_rm: int, one_rm lifed
    :param reps: int, number of reps attained
    :param formula: str, the formula used to calulate the 1rm: Currently supports the following formulas: 
        'Epley', 'Brzycki', 'McGlothin', 'Lombardi', 'Mayhew', 'OConner', 'Wathan'. 
        See https://en.wikipedia.org/wiki/One-repetition_maximum
     
    :return: float, calculated maximum one_rm attainable for the given number of reps. 
    """

    if  reps == 1:
        return one_rm

    formulas = ['Epley', 'Brzycki', 'McGlothin', 'Lombardi', 'Mayhew', 'OConner', 'Wathan']

    if formula not in formulas:
        raise ValueError('Formula must be one of {!r}'.format(formulas))

    if formula == 'Epley':
        return one_rm / (1 + (reps / 30))
    if formula == 'Brzycki':
        return one_rm / (36 / (37 - reps))
    if formula == 'McGlothin':
        return (one_rm / 100) * (101.3 - 2.67123 * reps)
    if formula == 'Lombardi':
        return one_rm / (reps ** 0.1)
    if formula == 'Mayhew':
        return (one_rm / 100) * (52.2 + 41.9 * np.exp(-0.055 * reps))
    if formula == 'OConner':
        return one_rm / (1 + (reps / 40))
    if formula == 'Wathan':
        return (one_rm / 100) * (48.8 + 53.8 * np.exp(-0.075 * reps))


def calculate_lifts(back_squat):
    """
    Calcualtes therohetical maxes of all the classics based off of the
    1RM Back squat

    :param back_squat: current 1RM weight for the back squat
    """

    print(" --- Squats ---")
    # BS to FS = 85-90%
    print("Front Squat: {} - {}".format(percentage(85, back_squat),
                                        percentage(90, back_squat)))

    print('\n --- Classics --- ')

    # CLEAN 80% BS
    clean = percentage(80, back_squat)
    print("Clean: {}".format(clean))
    # Snatch = 80% Clean (64% of BS)
    snatch = percentage(64, back_squat)
    print("Snatch: {}".format(snatch))
    # Jerk = 102% * Clean (80% * 102% = 81.6%)
    jerk = percentage(81.6, back_squat)
    print("Jerk: {}".format(jerk))

    print("\n --- Power Variants --- ")
    # Power = 85 - 90 % of full lift
    print("Power Clean: {} - {}".format(percentage(85, clean),
                                        percentage(90, clean)))

    print("Power Snatch: {} - {}".format(percentage(85, snatch),
                                         percentage(90, snatch)))

    print("\n --- Pushing Overhead ---")
    # Push Press = 75% of jerk
    print("Push Press: {}".format(percentage(75, jerk)))
