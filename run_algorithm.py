from algorithm import run_simulation
from multiprocessing import Pool, cpu_count
from functools import partial
import itertools


def run_city(save, modes, logfile, processes):
    """
    Runs the algorithm with all modes possible (load weighting and
    building/removing).
    :param save: save name of data
    :type save: str
    :param logfile: path to logfile without .txt
    :type logfile: str
    :param processes: number of parallel processes
    :type processes: int
    :return: None
    """
    fnc = partial(run_simulation, save, logfile)

    p = Pool(processes=processes)
    data = p.map(fnc, modes)


def main():
    save = 'hh_part'
    logfile = 'log/algorithm'

    # Number of CPU cores used
    processes = cpu_count()
    
    
    # modes for the algorithm
    # minmode: 0=unweighted loads, 1=weighted by penalty,
    #          2=weighted by average trip length
    # rev: False=Removing, True=Building
    minmodes = [1]
    rev = [False]
    #budget choice
    total_budget = [1000000]
    
    # method choices
    build_method = ['Monte Carlo']
    w = [0.9]
    cost_method = ['equal']


    modes = list(itertools.product(rev, minmodes, total_budget, build_method, 
                               w, cost_method))

    run_city(save, modes, logfile, processes)


if __name__ == '__main__':
    main()



