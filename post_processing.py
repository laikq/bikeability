"""
Post-processing for the algorithm:
Selects the best bike network given the data generated by the algorithm.
"""
import osmnx as ox
import numpy as np
from helper.data_maps_helper import load_graph_data
import networkx as nx


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


def get_best_graph(place, mode):
    """
    Return the networkx graph G with the best bikeability.
    """
    G, data = load_graph_data(place, mode)
    edited_edges = data[1]
    edited_edges_action = data[10]
    total_real_distance_traveled = data[4]
    max_dist_on_all = max({t['total length on all']
                           for t in total_real_distance_traveled})
    norm_dist_on_all = [t['total length on all'] / max_dist_on_all
                        for t in total_real_distance_traveled]
    max_norm = max(norm_dist_on_all)
    min_norm = min(norm_dist_on_all)
    bikeability = [1 - (t - min_norm) / (max_norm - min_norm)
                   for t in norm_dist_on_all]
    max_bikeability_index = np.argmax(bikeability)
    apply_edge_operations(G, edited_edges[:max_bikeability_index+1],
                          edited_edges_action[:max_bikeability_index+1])
    return G
