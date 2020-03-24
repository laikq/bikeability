import itertools
from clean_rawdata import clean_city
from prep_cleandata import prep_city
from run_algorithm import run_city
from plot_results import plot_city

# path to the call-a-bike csv
big_csv = 'csv/db_all.csv'
stations_csv = 'data/csv/db_stations.csv'

# nominatim request
city = 'Hamburg'
which_result = 2

# save name
save = 'hh'

# logging
logfile = 'log/algorithm'

# modes for the algorithm
# minmode: 0=unweighted loads, 1=weighted by penalty,
#          2=weighted by average trip length
# rev: False=Removing, True=Building
minmodes = [1]
rev = [False, True]
modes = list(itertools.product(rev, minmodes))

# Ignore clean_city() if you hav less than 16GB of RAM.
# clean_city(city, save, big_csv, stations_csv)
prep_city(city, which_result, save)
run_city(save, modes, logfile, processes=16)
plot_city(save, modes)
