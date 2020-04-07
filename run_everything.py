import itertools
from clean_rawdata import clean_city
from prep_cleandata import prep_city
from run_algorithm import run_city
from plot_results import plot_city, plot_algorithm, plot_city_mini
from helper.logger_helper import log_to_file
from helper.current_state_helper import calc_current_state
import numpy as np
import osmnx as ox

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
minmodes = [1, 2]
rev = [False]
#budget choice
trip_nbrs = np.load('data/algorithm/input/{}_demand.npy'.format(save),
                    allow_pickle=True)[0]
nxG = ox.load_graphml('{}.graphml'.format(save),
                      folder='data/algorithm/input', node_type=int)
nxG = nxG.to_undirected()
data_now = calc_current_state(nxG, trip_nbrs)
budget_now = data_now[1]
total_budget = [0.7*budget_now]
log_to_file(logfile, "Using budget = {}".format(total_budget))

# method choices
#build method: 0=Monte Carlo , 1=MFT, 2=random
build_method = [0, 1, 2]
w = [0.5]
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
plot_city_mini(save, modes)
plot_algorithm(save, mode, file_format='png',slice_by='iteration')
