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

CONV_FACT = 2.20462262

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

