from algorithm import run_simulation
from multiprocessing import Pool, cpu_count
from functools import partial
import itertools


def run_city(save, modes, logfile, processes, total_budget, build_method, w, cost_method):
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

    minmodes = [1]
    rev = [False, True]
    modes = list(itertools.product(rev, minmodes))

    run_city(save, modes, logfile, processes, total_budget, build_method, w, cost_method)


if __name__ == '__main__':
    main()



