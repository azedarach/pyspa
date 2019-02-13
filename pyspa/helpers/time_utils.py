import numpy as np

def is_leap_year(year):
    if (year % 4) != 0:
        return False
    elif (year % 100) != 0:
        return True
    elif (year % 400) != 0:
        return False
    else:
        return True

def get_days_in_year(year):
    if is_leap_year(year):
        return 366
    else:
        return 365

def get_days_in_months(year):
    if is_leap_year(year):
        return np.array([31, 29, 31, 30, 31, 30, 30, 31, 30, 31, 30, 31],
                        dtype="float64")
    else:
        return np.array([31, 28, 31, 30, 31, 30, 30, 31, 30, 31, 30, 31],
                        dtype="float64")

def datetime_to_decimal_time(dt):
    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour
    minute = dt.minute
    second = dt.second

    year_length = get_days_in_year(year)
    month_lengths = get_days_in_months(year)

    day_frac = (hour + minute / 60.0 + second / 3600.0) / 24.0

    return (year + (np.sum(month_lengths[:month - 1]) + day + day_frac)
            / year_length)

def to_decimal_times(dts):
    result = np.zeros(dts.shape[0])
    for i, dt in enumerate(dts):
        result[i] = datetime_to_decimal_time(dt)
    return result
