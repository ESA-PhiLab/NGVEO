import numpy as np

def filter_on_date(dates, date_str, type):
    """
    Filter list of dates
    :param dates: array of np.datetime64 to filter
    :param date_str: date-str (YYYY-MM-DD) to compare with
    :param type: Min or max
    :return: bool-array with same size as dates
    """
    def compare(a,b):
        a = a
        if type == 'min':
            return a >= b
        else:
            return a <= b

    #Sort on both year, month and day
    if '*' not in date_str:
        return compare( dates, np.datetime64(date_str))
    else:
        year = date_str[0:4]
        month = date_str[5:7]
        day = date_str[8:]

        #Bools indicating if we should filter on these date-elements
        y = '*' not in year
        m = '*' not in month
        d = '*' not in day

        years = (dates.astype('datetime64[Y]').astype(int) + 1970).astype('int32')
        months = (dates.astype('datetime64[M]').astype(int) % 12 + 1).astype('int32')
        days = (dates - dates.astype('datetime64[M]') + 1).astype('int32')



        # Sort on year only
        if y and not m and not d:
            return compare(years, int(year))

        #Sort on month only
        elif not y and  m and not d:
            return compare(months, int(month))

            # Sort on day only
        elif not y and not m and d:
            return compare(days, int(day))

        # Sort on month and day
        elif not y and m and d:
            raise NotImplementedError()
            months_days = months * 31 + days
            month_day = int(month)*31 + int(day)
            return compare(months_days, int(month_day))

        # Sort on year and day
        elif  y and not m and d:
            raise NotImplementedError()
            years_days = years * 31 + days
            year_day = int(year) * 31 + int(day)
            return compare(years_days, int(year_day))

        # Sort on year and month
        elif y and  m and not d:
            raise NotImplementedError()
            years_months = years * 12 + months
            year_month = int(year) * 12 + int(month)
            return compare(years_months, int(year_month))

