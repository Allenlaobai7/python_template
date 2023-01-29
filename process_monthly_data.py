import argparse
import datetime
import pandas as pd

def run(args):
    start_date, end_date = args.start_date, args.end_date
    try:
        _, _ = datetime.datetime.strptime(start_date, '%Y-%m-%d'), datetime.datetime.strptime(end_date, '%Y-%m-%d')
    except Exception as e: 
        print('inaccurate date. Error message:', e)
    def date_is_first_day_of_month(date):
        return bool(datetime.datetime.strptime(date, '%Y-%m-%d').day == 1)
    def date_is_eomonth(date):
        tmr_month = (datetime.datetime.strptime(date, '%Y-%m-%d') + datetime.timedelta(1)).month
        return bool(tmr_month != datetime.datetime.strptime(date, '%Y-%m-%d').month)
    assert date_is_first_day_of_month(start_date)
    assert date_is_eomonth(end_date)

    # get start date and end date for each month within the target period
    start_dates = pd.date_range(start_date, end_date, freq='MS')
    end_dates = pd.date_range(start_date, end_date, freq='M')
    dates = list(zip([i.strftime('%Y-%m-%d') for i in start_dates], [i.strftime('%Y-%m-%d') for i in end_dates]))
    for start_date, end_date in dates:
        print(f'Getting data from {start_date} to {end_date}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect collaboration data given start and end date')
    parser.add_argument('start_date', type=str)
    parser.add_argument('end_date', type=str)
    args = parser.parse_args()
    run(args)

