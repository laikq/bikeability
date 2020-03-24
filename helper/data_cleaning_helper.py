import pandas as pd
import numpy as np


def read_csv(path, delim):
    df = pd.read_csv(path, delimiter=delim)
    df.columns = map(str.lower, df.columns)
    return df


def write_csv(df, path):
    df.to_csv(path, index=False)


def creat_stations_csv(load_path, save_path, delim, header_trips):
    trips = read_csv(load_path, delim)
    trips.rename(columns=header_trips, inplace=True)

    header_start_stations = {'start station': 'station id',
                             'start latitude': 'latitude',
                             'start longitude': 'longitude'}
    header_end_stations = {'end station': 'station id',
                           'end latitude': 'latitude',
                           'end longitude': 'longitude'}
    stations = trips[['start station', 'start latitude', 'start longitude']]
    stations.rename(columns=header_start_stations, inplace=True)
    stations2 = trips[['end station', 'end latitude', 'end longitude']]
    stations.rename(columns=header_end_stations, inplace=True)
    stations.append(stations2, ignore_index=True)
    stations.drop_duplicates(keep='first', inplace=True)
    stations['location'] = list(zip(stations['latitude'],
                                    stations['longitude']))

    write_csv(stations, save_path)


def legacy_to_new(legacy_stations_path, legacy_to_new_path, legacy_trips_path,
                  new_trips_path, delim):
    l_stations = read_csv(legacy_stations_path, delim=delim)
    l2n = read_csv(legacy_to_new_path, delim=delim)
    l_trips = read_csv(legacy_trips_path, delim=delim)

    s_new = []
    e_new = []
    s_lat = []
    s_long = []
    e_lat = []
    e_long = []
    for row in l_trips.itertuples(index=False):
        s_old = row[0]
        e_old = row[2]
        if s_old in l_stations['legacy_id'] and e_old in l_stations[
            'legacy_id']:
            s_lat.append(l_stations.loc[l_stations['legacy_id'] == s_old][
                             'latitude'].iloc[0])
            s_long.append(l_stations.loc[l_stations['legacy_id'] == s_old][
                              'longitude'].iloc[0])
            e_lat.append(l_stations.loc[l_stations['legacy_id'] == e_old][
                             'latitude'].iloc[0])
            e_long.append(l_stations.loc[l_stations['legacy_id'] == e_old][
                              'longitude'].iloc[0])
        else:
            s_lat.append(np.nan)
            s_long.append(np.nan)
            e_lat.append(np.nan)
            e_long.append(np.nan)
        if s_old in l2n['legacy_id'] and e_old in l2n['legacy_id']:
            s_new.append(l2n.loc[l2n['legacy_id'] == s_old]['new_id'].iloc[0])
            e_new.append(l2n.loc[l2n['legacy_id'] == e_old]['new_id'].iloc[0])
        else:
            s_new.append(np.nan)
            e_new.append(np.nan)

    l_trips['start station new'] = s_new
    l_trips['start latitude'] = s_lat
    l_trips['start longitude'] = s_long
    l_trips['end station new'] = e_new
    l_trips['end latitude'] = e_lat
    l_trips['end longitude'] = e_long
    l_trips['duration'] = np.zeros(len(s_new))
    l_trips['start_station_name'] = np.zeros(len(s_new))
    l_trips['start_station_description'] = np.zeros(len(s_new))
    l_trips['end_station_name'] = np.zeros(len(s_new))
    l_trips['end_station_description'] = np.zeros(len(s_new))

    trips_header = {'start station new': 'start_station_id',
                    'end station new': 'end_station_id',
                    'start time': 'started_at', 'end time': 'ended_at',
                    'start latitude': 'start_station_latitude',
                    'start longitude': 'start_station_longitude',
                    'end latitude': 'end_station_latitude',
                    'end longitude': 'end__station_longitude'
                    }

    n_trips = l_trips.drop(columns=['start station', 'end station'])
    n_trips.rename(columns=trips_header, inplace=True)
    n_trips = n_trips[['started_at', 'ended_at', 'duration',
                       'start_station_id',
                       'start_station_name', 'start_station_description',
                       'start_station_latitude', 'start_station_longitude',
                       'end_station_id',
                       'end_station_name', 'end_station_description',
                       'end_station_latitude', 'end_station_longitude']]
    write_csv(n_trips, new_trips_path)


def combine(load_paths, save_path, delim):
    df = read_csv(load_paths[0], delim)
    for load_path in load_paths[1:]:
        df2 = read_csv(load_path, delim)
        df.append(df2, ignore_index=True)

    write_csv(df, save_path)


def clean(load_path_trips, load_path_stations, save_path, delim,
          header_trips, header_stations):
    trips = read_csv(load_path_trips, delim)
    trips.rename(columns=header_trips, inplace=True)

    stations = read_csv(load_path_stations, delim)
    stations.rename(columns=header_stations, inplace=True)

    stations.drop(stations[stations['longitude'].isna()].index, inplace=True)
    stations.drop(stations[stations['latitude'].isna()].index, inplace=True)

    trips['od pairs'] = list(zip(trips['start station'], trips['end station']))

    od_demand = trips['od pairs'].value_counts()
    od_demand = pd.DataFrame(od_demand).reset_index()
    od_demand.rename(columns={'od pairs': 'number of trips'}, inplace=True)

    od_list = []
    od_stations = []
    for row in od_demand.itertuples(index=False):
        s_id = int(row[0][0])
        e_id = int(row[0][1])
        if {s_id, e_id}.issubset(set(stations['station id'])):
            start = stations.loc[stations['station id'] == s_id]
            end = stations.loc[stations['station id'] == e_id]
            s_loc = (start['latitude'].iloc[0], start['longitude'].iloc[0])
            e_loc = (end['latitude'].iloc[0], end['longitude'].iloc[0])
            od_list.append((s_loc, e_loc))
            od_stations.append((s_id, e_id))
        else:
            od_list.append((np.nan, np.nan))
            od_stations.append((np.nan, np.nan))

    od_demand[['start station', 'end station']] = \
        pd.DataFrame(od_stations, columns=['start station', 'end station'])
    od_demand[['start location', 'end location']] = \
        pd.DataFrame(od_list, columns=['start location', 'end location'])

    od_demand.dropna(inplace=True)
    od_demand.drop(columns=['index'], inplace=True)

    od_demand['start latitude'] = od_demand['start location'].str[0]
    od_demand['start longitude'] = od_demand['start location'].str[1]
    od_demand['end latitude'] = od_demand['end location'].str[0]
    od_demand['end longitude'] = od_demand['end location'].str[1]

    od_demand = od_demand[['start station', 'start location',
                           'start latitude', 'start longitude',
                           'end station', 'end location',
                           'end latitude', 'end longitude',
                           'number of trips']]

    write_csv(od_demand, save_path)

    ut = od_demand['number of trips'].count()
    tt = sum(od_demand['number of trips'])
    print('total trips: {}, unique trips: {}'.format(tt, ut))


def compact_csv(load_path, save_path, drop_c, delim):
    df = read_csv(load_path, delim)

    df.drop(columns=drop_c, inplace=True)
    df.dropna(inplace=True)
    df.astype('int64', inplace=True)

    df.to_csv(save_path, index=False)


def compact_csv_db(load_path, save_path, drop_c, delim, city='Hamburg'):
    df = read_csv(load_path, delim)

    df.drop(df[df['city_rental_zone'] != city].index, inplace=True)
    df.drop(columns=drop_c, inplace=True)
    df.dropna(inplace=True)
    df.astype('int64')

    df.to_csv(save_path, index=False)


def main():
    print("Please start via a city specific script.")


if __name__ == '__main__':
    main()
