"""
This module includes usefull helper functions for working with networkx.
"""
import networkx as nx
import numpy as np


def get_street_type(G, edge):
    """
    Returns the street type of the edge in G. If 'highway' in G is al list,
    return first entry.
    :param G: Graph.
    :type G: networkx graph.
    :param edge: Edge in the networkx graph.
    :type edge: tuple of int
    :return: Street type.
    :rtype: str
    """
    street_type = G[edge[0]][edge[1]][0]['highway']
    if isinstance(street_type, str):
        return street_type
    else:
        return street_type[0]


def get_street_type_cleaned(G, edge):
    """
    Returns the street type of the edge. Street types are reduced to
    primary, secondary, tertiary and residential.
    :param G: Graph.
    :type G: networkx graph.
    :param edge: Edge in the networkx graph.
    :type edge: tuple of int
    :return: Street type·
    :rtype: str
    """
    st = get_street_type(G, edge)
    if st in ['primary', 'primary_link', 'trunk', 'trunk_link']:
        return 'primary'
    elif st in ['secondary', 'secondary_link']:
        return 'secondary'
    elif st in ['tertiary', 'tertiary_link', 'road']:
        return 'tertiary'
    else:
        return 'residential'


def get_all_street_types(G):
    """
    Returns all street types appearing in G.
    :param G: Graph.
    :type G: networkx graph.
    :return: List of all street types.
    :rtype: list of str
    """
    street_types = set()
    for edge in G.edges():
        street_types.add(get_street_type(G, edge))
    return list(street_types)


def get_all_street_types_cleaned(G):
    """
    Returns all street types appearing in G. Street types are reduced to
    primary, secondary, tertiary and residential.
    :param G: Graph.
    :type G: networkx graph.
    :return: List of all street types.
    :rtype: list of str
    """
    street_types_cleaned = set()
    for edge in G.edges():
        street_types_cleaned.add(get_street_type(G, edge))
    return list(street_types_cleaned)


def get_speed_limit(G, edge, nk2nx=False):
    """
    Returns speed limit of the edge in G.
    :param G: Graph.
    :type G: networkx graph
    :param edge: Edge in the networkx or networkit graph. If edge is from the
    networkit graph, nk2nx dict has to be given, if its from the networkx
    graph nk2nx has to be set False.
    :type edge: tuple of integers
    :param nk2nx: edge map form networkit graph to networkx graph.
    :type nk2nx: dict or bool
    :return: Speed limit.
    :rtype: float
    """
    if isinstance(nk2nx, dict):
        if edge in nk2nx:
            edge = nk2nx[edge]
        else:
            edge = nk2nx[(edge[1], edge[0])]
    if 'maxspeed' in G[edge[0]][edge[1]]:
        speed_limit = G[edge[0]][edge[1]]['maxspeed']
    else:
        speed_limit = 50
    if isinstance(speed_limit, float):
        return speed_limit
    elif isinstance(speed_limit, int):
        return speed_limit
    elif isinstance(speed_limit, list):
        return speed_limit[0]
    else:
        return 50


def get_street_length(G, edge, nk2nx=False):
    """
    Returns the length of the edge in G.
    :param G: Graph.
    :type G: networkx graph.
    :param edge: Edge in the networkx or networkit graph. If edge is from the
    networkit graph, nk2nx dict has to be given, if its from the networkx
    graph nk2nx has to be set False.
    :type edge: tuple of integers
    :param nk2nx: edge map form networkit graph to networkx graph.
    :type nk2nx: dict or bool
    :return: Length of edge.
    :rtype: float
    """
    if isinstance(nk2nx, dict):
        if edge in nk2nx:
            edge = nk2nx[edge]
        else:
            edge = nk2nx[(edge[1], edge[0])]
    length = G[edge[0]][edge[1]][0]['length']
    return length


def get_cost(bikepaths, edge_dict, cost_dict):
    """
    Returns the cost of building bike paths.
    :param bikepaths: Edges with bike paths.
    :type bikepaths: list of tuple of int
    :param edge_dict: Dictionary with all edge information.
    :type edge_dict: dict of dicts
    :param cost_dict: Dictionary with cost of edge depending on street type.
    :type cost_dict: dict
    :return: Cost of the edge
    :rtype: float
    """
    cost = 0
    for edge in bikepaths:
        street_type = edge_dict[edge]['street type']
        street_length = edge_dict[edge]['real length']
        cost += street_length * cost_dict[street_type]
    return cost


def get_trip_edges(edges_dict, trip_nodes):
    """
    Returns the edge sequence of a trip given by its node sequence.
    :param edges_dict: Dictionary with all information about the edges.
    :type edges_dict: Dict of dicts.
    :param trip_nodes: Node sequence of a trip.
    :type trip_nodes: list of integers
    :return: Edge sequence.
    :rtype: list of tuples of integers
    """
    edge_sequence = []
    for i in range(len(trip_nodes) - 1):
        f_n = trip_nodes[i]         # First node
        s_n = trip_nodes[i + 1]     # Second node of the edge
        # Dict doesn't accept (2, 1) for undirected edge (1, 2):
        if (f_n, s_n) in edges_dict:
            edge_sequence.append((f_n, s_n))
        else:
            edge_sequence.append((s_n, f_n))
    return edge_sequence


def get_minimal_loaded_edge(edge_dict, trips_dict, minmode=0, rev=False):
    """
    Returns the minimal loaded edge in edge_list.
    If unedited=True it returns the minimal loaded unedited edge.
    If there are multiple edges with the same minimal load, one is randomly
    drawn.
    :param edge_dict: Dictionary with all information about the edges.
    :type edge_dict: dict of dicts.
    :param trips_dict: Dictionary with al information about the trips.
    :type trips_dict: dict of dicts
    :param minmode: Mode in which way the min loaded edge should be searched.
                    0 = min(load), 1 = min(load*penalty)
    :type minmode: int
    :param rev: If True returns maximal loaded edge
    :type rev: bool
    :return: minimal loaded edge
    :rtype: Tuple of integers
    """
    edges = {edge: edge_info for edge, edge_info in edge_dict.items()
             if edge_info['bike lane']}

    # Different picking rules

    edges_load = {edge: edge_info['load'] * edge_info['penalty']
                  for edge, edge_info in edges.items()}
    if edges_load == {}:
        return 'We are done!'
    else:
        min_load = min(edges_load.values())
        min_edges = [e for e, load in edges_load.items()
                     if load == min_load]
        min_edge = min_edges[np.random.choice(len(min_edges))]
        return min_edge


def bike_lane_percentage(edge_dict):
    """
    Returns the bike lane percentage by length.
    :param edge_dict: Dictionary with all information about the edges
    :type edge_dict: dict of dicts
    :return: percentage of bike lanes by length.
    :rtype float
    """
    bike_length = 0
    total_length = 0
    for edge, edge_info in edge_dict.items():
        total_length += edge_info['real length']
        if edge_info['bike lane']:
            bike_length += edge_info['real length']
    return bike_length / total_length


def check_if_trip_on_street(trip_info, edge_dict):
    """
    Checks if given trip is somewhere on a street.
    :param trip_info: Dictionary with all information about the trip.
    :type trip_info: dict
    :param edge_dict: Dictionary with all information about the edges.
    :type edge_dict: dict of dicts
    :return: True if on street false if not.
    :rtype: bool
    """
    for edge in trip_info['edges']:
        if not edge_dict[edge]['bike lane']:
            return True
    return False


def nbr_of_trips_on_street(trips_dict):
    """
    Returns the number of trips that are somewhere on a street.
    :param trips_dict: Dictionary with all information about the trip.
    :type trips_dict: dict of dicts
    :return: Number of trips at least once on a street.
    :rtype: integer
    """
    nbr_on_street = 0
    for trip, trip_info in trips_dict.items():
        if trip_info['on street']:
            nbr_on_street += trip_info['nbr of trips']
    return nbr_on_street


def set_trips_on_street(trips_dict, edge_dict):
    """
    Sets "on street" value in trips_dict to the right value.
    :param trips_dict: Dictionary with all information about the trips.
    :type trips_dict: dict of dicts
    :param edge_dict: Dictionary with all information about the edges.
    :type edge_dict: dict of dicts
    :return: Updated trips_dict.
    :rtype: dict of dicts
    """
    for trip, trip_info in trips_dict.items():
        trip_info['on street'] = False
        for edge in trip_info['edges']:
            if not edge_dict[edge]['bike lane']:
                trip_info['on street'] = True
    return trips_dict


def get_len_of_trips_over_edge(edge, edge_list, trips_dict):
    """
    Returns the total traveled distance over the given edge.
    ttd = edge length * nbr of trips over edge
    :param edge: Edge.
    :type edge: tuple of integers
    :param edge_list: Dictionary with all information about the edges.
    :type edge_list: dict of dicts
    :param trips_dict: Dictionary with all information about the trips.
    :type trips_dict: dict of dicts
    :return: Total traveled distance.
    :rtype float
    """
    length = 0
    for trip in edge_list[edge]['trips']:
        length += trips_dict[trip]['nbr of trips'] * \
                  trips_dict[trip]['length real']
    return length


def real_trip_length(trip_info, edge_dict):
    """
    Returns the real length og a trip.
    :param trip_info: Dictionary with all information about the trip.
    :type trip_info: dict of dicts.
    :param edge_dict: Dictionary with all information about the edges.
    :type edge_dict: dict of dicts
    :return: Real length of the trip.
    :rtype: float
    """
    length = sum([edge_dict[edge]['real length']
                  for edge in trip_info['edges']])
    return length


def felt_trip_length(trip_info, edge_dict):
    """
    Returns the felt length og a trip.
    :param trip_info: Dictionary with all information about the trip.
    :type trip_info: dict of dicts.
    :param edge_dict: Dictionary with all information about the edges.
    :type edge_dict: dict of dicts
    :return: Felt length of the trip.
    :rtype: float
    """
    length = sum([edge_dict[edge]['felt length']
                  for edge in trip_info['edges']])
    return length


def len_on_types(trip_info, edge_dict, len_type='real'):
    """
    Returns a dict with the length of the trip on the different street types.
    len_type defines if felt or real length.
    :param trip_info: Dictionary with all information about the trip.
    :type trip_info: dict
    :param edge_dict: Dictionary with all information about the edges.
    :type edge_dict: dict of dicts
    :param len_type: 'real' or 'felt' length is used.
    :type len_type: str
    :return: Dictionary with length on different street types.
    :rtype: dict
    """
    len_on_type = {t: 0 for t, l in
                   trip_info[len_type+' length on types'].items()}
    for edge in trip_info['edges']:
        street_type = edge_dict[edge]['street type']
        street_length = edge_dict[edge][len_type+' length']
        if edge_dict[edge]['bike lane']:
            len_on_type['bike lane'] += street_length
        else:
            len_on_type[street_type] += street_length
    return len_on_type


def total_len_on_types(trips_dict, len_type):
    """
    Returns the total distance driven sorted by street type.
    :param trips_dict: Dictionary with all information about the trip.
    :type trips_dict: dict of dicts
    :param len_type: 'real' or 'felt' length is used.
    :type len_type: str
    :return: Dictionary with total length on different street types.
    :rtype: dict
    """
    lop = 0
    los = 0
    lot = 0
    lor = 0
    lob = 0
    for trip, trip_info in trips_dict.items():
        nbr_of_trips = trip_info['nbr of trips']
        lop += nbr_of_trips * \
               trip_info[len_type+' length on types']['primary']
        los += nbr_of_trips * \
               trip_info[len_type+' length on types']['secondary']
        lot += nbr_of_trips * \
               trip_info[len_type+' length on types']['tertiary']
        lor += nbr_of_trips * \
               trip_info[len_type+' length on types']['residential']
        lob += nbr_of_trips * \
               trip_info[len_type+' length on types']['bike lane']
    tlos = lop + los + lot + lor
    tloa = tlos + lob
    return {'total length on all': tloa, 'total length on street': tlos,
            'total length on primary': lop, 'total length on secondary': los,
            'total length on tertiary': lot,
            'total length on residential': lor,
            'total length on bike lanes': lob}


def set_len(trips_dict, edge_dict):
    """
    Sets the length of a trip to the correct value in the trips dictionary.
    :param trips_dict: Dictionary with all information about the trip.
    :type trips_dict: dict of dicts
    :param edge_dict:  Dictionary with all information about the edges.
    :type edge_dict: dict of dicts
    :return: Updated trips dictionary
    :rtype: dict of dicts
    """
    for trip, trip_info in trips_dict.items():
        trip_info['length real'] = real_trip_length(trip_info, edge_dict)
        trip_info['length felt'] = felt_trip_length(trip_info, edge_dict)
    return trips_dict


def set_len_on_types(trips_dict, edge_dict):
    """
    Sets the length by type of a trip to the correct value in the trips
    dictionary.
    :param trips_dict: Dictionary with all information about the trip.
    :type trips_dict: dict of dicts
    :param edge_dict:  Dictionary with all information about the edges.
    :type edge_dict: dict of dicts
    :return: Updated trips dictionary
    :rtype: dict of dicts
    """
    for trip, trip_info in trips_dict.items():
        trip_info['real length on types'] = len_on_types(trip_info, edge_dict,
                                                         'real')
        trip_info['felt length on types'] = len_on_types(trip_info, edge_dict,
                                                         'felt')
    return trips_dict


def add_load(edge_dict, trips_dict):
    """
    Adds load and trip_id of the given trips to the edges.
    :param edge_dict: Dictionary with all information about the edges.
    :type edge_dict: dict of dicts
    :param trips_dict: Dictionary with all information about the trip.
    :type trips_dict: dict of dicts
    :return: edge_dict with updated information
    :rtype: dict of dicts
    """
    for trip, trip_info in trips_dict.items():
        for e in trip_info['edges']:
            edge_dict[e]['trips'] += [trip]
            edge_dict[e]['load'] += trip_info['nbr of trips']
    return edge_dict


def delete_load(edge_dict, trips_dict):
    """
    Deletes load and trip_id of the given trips from the edges.
    :param edge_dict: Dictionary with all information about the edges.
    :type edge_dict: dict of dicts
    :param trips_dict: Dictionary with all information about the trip.
    :type trips_dict: dict of dicts
    :return: edge_dict with updated information
    :rtype: dict of dicts
    """
    for trip, trip_info in trips_dict.items():
        for e in trip_info['edges']:
            edge_dict[e]['trips'].remove(trip)
            edge_dict[e]['load'] -= trip_info['nbr of trips']
    return edge_dict


def calc_trips(nxG, edge_dict, trips_dict):
    """
    Calculates the shortest paths for all trips and sets all corresponding
    information in the trip_dict
    :param nxG: graph to calculate the s.p. in.
    :type nxG: networkx graph
    :param edge_dict: Dictionary with all information about the edges of G.
    :type edge_dict: dict of dicts
    :param trips_dict: Dictionary with all information about the trips.
    :type trips_dict: dict of dicts
    :return: Updated trips_dict and edge_dict.
    :rtype: dict of dicts
    """
    # Calculate single source paths for all origin nodes
    origin_nodes = list({k[0] for k, v in trips_dict.items()})
    for source in origin_nodes:
        shortest_paths = nx.single_source_dijkstra_path(nxG, source,
                                                        weight='length')
    # Set all information to trip_info and edge_info
        for trip, trip_info in trips_dict.items():
            if trip[0] == source:
                trip_info['nodes'] = shortest_paths[trip[1]]
                trip_info['edges'] = get_trip_edges(edge_dict,
                                                    trip_info['nodes'])
                trip_info['length felt'] = felt_trip_length(trip_info,
                                                            edge_dict)
                trip_info['length real'] = real_trip_length(trip_info,
                                                            edge_dict)
                trip_info['real length on types'] = len_on_types(trip_info,
                                                                 edge_dict,
                                                                 'real')
                trip_info['felt length on types'] = len_on_types(trip_info,
                                                                 edge_dict,
                                                                 'felt')
                trip_info['on street'] = check_if_trip_on_street(trip_info,
                                                                 edge_dict)
                for e in trip_info['edges']:
                    edge_dict[e]['trips'] += [trip]
                    edge_dict[e]['load'] += trip_info['nbr of trips']
    return trips_dict, edge_dict


def calc_current_state(nxG, trip_nbrs):
    """
    Calculates the observables for the current state.
    :param nxG: graph to calculate the s.p. in.
    :type nxG: networkx graph
    :param trip_nbrs: demand dict
    :type trip_nbrs: dict
    :return:
    """
    stypes = ['primary', 'secondary']
    bike_paths = [e for e in nxG.edges()
                  if get_street_type_cleaned(nxG, e) in stypes]

    # All street types in network
    street_types = get_all_street_types_cleaned(nxG)
    # Add bike lanes
    len_on_type = {t: 0 for t in street_types}
    len_on_type['primary'] = 0
    len_on_type['bike lane'] = 0

    # Set penalties for different street types
    penalties = {'primary': 7, 'secondary': 2.4, 'tertiary': 1.4,
                 'residential': 1.1}

    # Set cost for different street types
    street_cost = {'primary': 1, 'secondary': 1, 'tertiary': 1,
                   'residential': 1}

    trips_dict = {t_id: {'nbr of trips': nbr_of_trips, 'nodes': [],
                         'edges': [], 'length real': 0, 'length felt': 0,
                         'real length on types': len_on_type,
                         'felt length on types': len_on_type,
                         'on street': False}
                  for t_id, nbr_of_trips in trip_nbrs.items()}
    edge_dict = {edge: {'felt length': get_street_length(nxG, edge),
                        'real length': get_street_length(nxG, edge),
                        'street type': get_street_type_cleaned(nxG, edge),
                        'penalty': penalties[
                            get_street_type_cleaned(nxG, edge)],
                        'speed limit': get_speed_limit(nxG, edge),
                        'bike lane': True, 'load': 0, 'trips': []}
                 for edge in nxG.edges()}

    for edge, edge_info in edge_dict.items():
        if edge not in bike_paths:
            edge_info['bike lane'] = False
            edge_info['felt length'] *= edge_info['penalty']
            nxG[edge[0]][edge[1]][0]['length'] *= edge_info['penalty']

    calc_trips(nxG, edge_dict, trips_dict)

    # Initialise lists
    total_cost = get_cost(bike_paths, edge_dict, street_cost)
    bike_lane_perc = bike_lane_percentage(edge_dict)
    total_real_distance_traveled = total_len_on_types(trips_dict, 'real')
    total_felt_distance_traveled = total_len_on_types(trips_dict, 'felt')
    nbr_on_street = nbr_of_trips_on_street(trips_dict)

    # Save data of this run to data array
    data = np.array([bike_paths, total_cost, bike_lane_perc,
                     total_real_distance_traveled,
                     total_felt_distance_traveled, nbr_on_street])
    return data
