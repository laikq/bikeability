"""
This script cleans and compacts data so it can be handled easier.
"""
from helper.data_cleaning_helper import *


def clean_city(city, save, big_csv, stations_csv):
    """
    Compacts and cleans the call-a-bike data given by big_csv.
    :param city: city to use data as written in big_csv.
    :type city: str
    :param save: save name of the city
    :type save: str
    :param big_csv: path to call-a-bike csv
    :type big_csv: str
    :return: None
    """
    city_csv = 'data/csv/{}_trips.csv'.format(save)
    cleaned_csv = 'data/csv/{}_trips_compact.csv'.format(save)

    drop_c = ['booking_hal_id', 'category_hal_id', 'vehicle_hal_id',
              'customer_hal_id', 'date_booking', 'date_from', 'date_until',
              'compute_extra_booking_fee', 'traverse_use', 'distance',
              'start_rental_zone', 'end_rental_zone', 'rental_zone_hal_src',
              'city_rental_zone', 'technical_income_channel']
    delim_compact = ';'
    delim_clean = ','

    header_trips = {'start_rental_zone_hal_id': 'start station',
                    'end_rental_zone_hal_id': 'end station'}
    header_stations = {'rental_zone_hal_id': 'station id',
                       'location': 'location',
                       'latitude': 'latitude', 'longitude': 'longitude'}

    # compact_csv_db(big_csv, city_csv, drop_c, delim_compact, city)
    clean(city_csv, stations_csv, cleaned_csv, delim_clean,
          header_trips=header_trips, header_stations=header_stations)


def main():
    big_csv = 'data/csv/db_trips_all.csv'
    stations_csv = 'data/csv/db_stations.csv'
    city = 'Hamburg'
    save = 'hh'

    clean_city(city, save, big_csv, stations_csv)


if __name__ == '__main__':
    main()
