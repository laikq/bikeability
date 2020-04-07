"""
Post-processing for the algorithm:
Selects the best bike network given the data generated by the algorithm.
"""
import osmnx as ox
import numpy as np
import networkx as nx
import osmnx as ox


def load_data(place, mode):
    data = np.load('data/algorithm/output/{}_data_mode_{:d}{}{}{}.npy'
                   .format(place, mode[0], mode[1], mode[3], mode[5]),
                   allow_pickle=True)
    d = {}
    d['edited edges'] = data[0]
    d['edited edges nx'] = data[1]
    d['total cost'] = data[2]
    d['bike lane perc'] = data[3]
    d['total real distance traveled'] = data[4]
    d['total felt distance traveled'] = data[5]
    d['nbr on street'] = data[6]
    d['len saved'] = data[7]
    d['nbr of cbc'] = data[8]
    d['gcbc size'] = data[9]
    d['edge action'] = data[10]
    d['bikeability'] = calculate_bikeability(d['total real distance traveled'])
    d['felt bikeability'] = calculate_bikeability(d['total felt distance traveled'])
    d['iteration'] = list(range(len(d['bikeability'])))
    # "unpack" the dictionaries 'total real distance traveled' and 'total felt
    # distance traveled'
    # goal: entries like d['total felt length on street'] return what you expect
    #       (that is: [t['total length on street']
    #                  for t in d['total felt distance traveled']])
    # step 1: transform list of dicts -> dict of lists
    tdt_dict = {}
    for rf in ('real', 'felt'):
        tdt_dict[rf] = {}
        for key in d['total ' + rf + ' distance traveled'][0]:
            tdt_dict[rf][key] = [t[key] for t
                                 in d['total ' + rf + ' distance traveled']]
    # step 2: unpack the tdt_dict into d
    for rf in ('real', 'felt'):
        for key, val in tdt_dict[rf].items():
            # key is something like 'total length on all'
            # val is a list
            d_key = key.replace('total length', 'total ' + rf + ' length')
            d[d_key] = val
    return d


def load_graph(place, mode):
    G = ox.load_graphml('{}.graphml'.format(place),
                        folder='data/algorithm/input', node_type=int)
    G = G.to_undirected()
    return G


def apply_edge_operations(G, edited_edges, edge_action):
    """
    Apply the operations listed in edited_edges and edge_action to G. Starts by
    assuming there is a bike lane everywhere (nx.set_edge_attributes(G, True,
    'bike lane')).
    :param G: networkx graph
    :type edge_action: list of bools
    :return: modified networkx graph
    """
    nx.set_edge_attributes(G, True, 'bike lane')
    for changed_edge, change_action in zip(edited_edges, edge_action):
        G.edges[(*changed_edge, 0)]['bike lane'] = change_action
    return G


def calculate_bikeability(total_real_distance_traveled):
    """
    Calculate the bikeability.
    :param total_real_distance_traveled: a list (each entry is generated in one
    iteration of the algorithm) of dictionaries, which each save the total
    length cyclists have to drive on each street type. Note: 'total length on
    street' is the sum of 'total length on
    {primary|secondary|tertiary|residential}' and 'total length on bike lane' is
    'total length on all' minus 'total length on street'.
    """
    max_dist_on_all = max({t['total length on all']
                           for t in total_real_distance_traveled})
    norm_dist_on_all = [t['total length on all'] / max_dist_on_all
                        for t in total_real_distance_traveled]
    max_norm = max(norm_dist_on_all)
    min_norm = min(norm_dist_on_all)
    bikeability = [1 - (t - min_norm) / (max_norm - min_norm)
                   for t in norm_dist_on_all]
    return bikeability


def get_best_graph(place, mode, budget=1000):
    """
    Return the networkx graph G with the best bikeability.
    """
    G = load_graph(place, mode)
    data = load_data(place, mode)
    bikeability = calculate_bikeability(data['total real distance traveled'])
    # set bikeability to 0 for all configurations that cost more than our budget
    # -- because bike paths are worth nothing if they can't be built!
    bikeability = [0 if cost > budget else ba
                   for cost, ba in zip(data['total cost'], bikeability)]
    max_bikeability_index = np.argmax(bikeability)
    apply_edge_operations(G, data['edited edges'][:max_bikeability_index+1],
                          data['edge action'][:max_bikeability_index+1])
    return G
