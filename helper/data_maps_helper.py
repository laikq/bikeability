# -*- coding: utf-8 -*-
"""
"""

import osmnx as ox
import networkx as nx
from random import uniform
import numpy as np


def get_street_type(G, edge):
    """
    Returns the street type of the edge in G. If 'highway' in G is a list,
    return first entry.
    :param G: Graph.
    :type G: networkx graph.
    :param edge: Edge.
    :type edge: tuple of integers
    :return: Street type.
    :rtype: str
    """
    street_type = G[edge[0]][edge[1]][0]['highway']
    if isinstance(street_type, str):
        return street_type
    else:
        return street_type[0]


def download_map(place, by_bbox, by_name, by_polygon, trunk=False):
    """
    Downloads, cleans and returns map.
    :param place: Place yout want to download the map for.
                If by_bbox=True it has to be a list of the northern, southern,
                eastern and western bound of the box.
                If by_name=True it hast to be a list of the name of the place
                and the correct Nominatim result you want to download.
                If by_polygon=True it has to be a polygon.
    :type place: list
    :param by_bbox: Download by boundary box.
                    If True, by_name and by_polygon has to be False.
    :type by_bbox: bool
    :param by_name: Download by place.
                    If True, by_bbox and by_polygon has to be False.
    :type by_name: bool
    :param by_polygon: Download place by polygon.
                        If True, by_bbox and  by_name has to be False.
    :type by_polygon: bool
    :param trunk: Should street type 'trunk' be kept in the graph.
    :type trunk: bool
    :return: Cleaned bike graph.
    :rtype: networkx graph
    """
    if by_bbox:
        print('Downloading map from bounding box. Northern bound: {}, '
              'southern bound: {}, eastern bound: {}, western bound: {}'
              .format(place[0], place[1], place[2], place[3]))
        G = ox.graph_from_bbox(place[0], place[1], place[2], place[3],
                               network_type='drive')
    elif by_name:
        print('Downloading map py place. Name of the place: {}, '
              'Nominatim result number {}.'.format(place[0], place[1]))
        G = ox.graph_from_place(place[0], which_result=place[1],
                                network_type='drive')
    elif by_polygon:
        print('Downloading map py polygon. Given polygon: {}'.format(place))
        G = ox.graph_from_polygon(place, network_type='drive')
    else:
        print('Your Input was illogical. Now you get a random map.')
        rand_point = (uniform(-90, 90), uniform(-180, 180))
        G = ox.graph_from_point(rand_point, network_type='drive')

    # Remove self loops
    self_loops = list(nx.selfloop_edges(G))
    G.remove_edges_from(self_loops)
    print('Removed {} self loops.'.format(len(self_loops)))

    # Remove motorways and trunks
    if trunk:
        s_t = ['motorway', 'motorway_link']
    else:
        s_t = ['motorway', 'motorway_link', 'trunk', 'trunk_link']
    edges_to_remove = [e for e in G.edges() if get_street_type(G, e) in s_t]
    G.remove_edges_from(edges_to_remove)
    print('Removed {} car only edges.'.format(len(edges_to_remove)))

    # Remove isolated nodes
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    print('Removed {} isolated nodes.'.format(len(isolated_nodes)))
    G = ox.get_largest_component(G)
    print('Reduce to largest connected component')

    # Bike graph assumed undirected.
    G = G.to_undirected()
    print('Turned graph to undirected.')

    return G


def save_map(G, save_name):
    ox.save_graphml(G, filename='{}.graphml'.format(save_name))


def load_graph_data(place, mode):
    G = ox.load_graphml('{}.graphml'.format(place),
                        folder='data/algorithm/input', node_type=int)
    G = G.to_undirected()
    data = np.load('data/algorithm/output/{}_data_mode_{}{}{}{}.npy'
                   .format(place, int(mode[0]), mode[1], mode[3], mode[5]), allow_pickle=True)
    return G, data


def main():
    print("Please start via a city specific script.")


if __name__ == '__main__':
    main()
