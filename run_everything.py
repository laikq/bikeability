import itertools
from clean_rawdata import clean_city
from prep_cleandata import prep_city
from run_algorithm import run_city
from plot_results import plot_city, plot_algorithm

# path to the call-a-bike csv
big_csv = 'csv/db_all.csv'
stations_csv = 'data/csv/db_stations.csv'

# nominatim request
city = 'Hamburg'
which_result = 2

# save name
save = 'hh_part'

# logging
logfile = 'log/algorithm'

# modes for the algorithm
# minmode: 0=unweighted loads, 1=weighted by penalty,
#          2=weighted by average trip length
# rev: False=Removing, True=Building
minmodes = [1]
rev = [False]
#budget choice
total_budget = [20000]

# method choices
#build method: 0=Monte Carlo , 1=MFT
build_method = [0]
w = [0.9]
#cost method: 0 = equal, 1 = weighted
cost_method = [0]


modes = list(itertools.product(rev, minmodes, total_budget, build_method, 
                               w, cost_method))



# Ignore clean_city() if you hav less than 16GB of RAM.
# clean_city(city, save, big_csv, stations_csv)
prep_city(city, which_result, save)
run_city(save, modes, logfile, processes=4)
#plot_city(save, modes)
mode = modes[0]
plot_algorithm(save, mode, file_format='png',slice_by='iteration')
