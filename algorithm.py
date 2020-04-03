"""
Bike network optimisation algorithm with networkit.
"""
import osmnx as ox
from helper.algorithm_helper import *
from helper.logger_helper import *
from copy import deepcopy
import time


def calc_trips(nkG, edge_dict, trips_dict):
    """
    Calculates the shortest paths for all trips and sets all corresponding
    information in the trip_dict
    :param nkG: graph to calculate the s.p. in.
    :type nkG: networkit graph
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
        shortest_paths = get_all_shortest_paths(nkG, source)
    # Set all information to trip_info and edge_info
        for trip, trip_info in trips_dict.items():
            if trip[0] == source:
                if not trip_info['nodes'] == shortest_paths[trip[1]]:
                    delete_load(edge_dict, {trip: trip_info})
                    trip_info['nodes'] = shortest_paths[trip[1]]
                    trip_info['edges'] = get_trip_edges(edge_dict,
                                                        trip_info['nodes'])
                    for e in trip_info['edges']:
                        edge_dict[e]['trips'] += [trip]
                        edge_dict[e]['load'] += trip_info['nbr of trips']
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
    return trips_dict, edge_dict


def edit_edge(nkG, edge_dict, edge):
    """
    Edits "felt length" of given edge  in the edge_dict and "length" in G.
    Length change is done corresponding to the street type of the edge.
    :param nkG: Graph.
    :type nkG: networkit graph
    :param edge_dict: Dictionary with all information about the edges of G.
    :type edge_dict: dict of dicts
    :param edge: Edge to edit.
    :type edge: tuple of integers
    :return: Updated G and edge_dict.
    :rtype: networkx graph and  dict of dicts
    """
    edge_dict[edge]['bike lane'] = not edge_dict[edge]['bike lane']
    if edge_dict[edge]['bike lane'] == False:
        edge_dict[edge]['felt length'] *= edge_dict[edge]['penalty']
    else: 
        edge_dict[edge]['felt length'] /= edge_dict[edge]['penalty']
    nkG.setWeight(edge[0], edge[1], edge_dict[edge]['felt length'])
    return nkG, edge_dict


def edit_network(nkG, nkG_edited, edge_dict, trips_dict, nk2nx_nodes,
                 nk2nx_edges, street_cost, starttime, logfile, place,
                 minmode, rev, total_budget=10000, build_method = 'Monte Carlo', 
                 w=0.9, cost_method = 'equal'):
    """
    Edits the least loaded unedited edge until no unedited edges are left.
    :param nkG: Graph.
    :type nkG: networkit graph
    :param nkG_edited: Graph that can be edited.
    :type nkG_edited: networkit graph
    :param edge_dict: Dictionary of edges of G. {edge: edge_info}
    :type edge_dict: dcit of dicts
    :param trips_dict: Dictionary with al information about the trips.
    :type trips_dict: dict of dicts
    :param nk2nx_nodes: Dictionary that maps nk nodes to nx nodes.
    :type nk2nx_nodes: dict
    :param nk2nx_edges: Dictionary that maps nk edges to nx edges.
    :type nk2nx_edges: dict
    :param street_cost: Dictionary with construction cost of street types
    :type street_cost: dict
    :param starttime: Time the script started. For logging only.
    :type starttime: timestamp
    :param logfile: Location of the log file
    :type logfile: str
    :param place: name of the network
    :type place: str
    :param minmode: Which minmode should be chosen
    :type minmode: int
    :param rev: If true, builds up bike lanes, not removes.
    :type rev: bool
    :return: data array
    :rtype: numpy array
    """
    # Initial calculation
    print('Initial calculation started.')
    calc_trips(nkG, edge_dict, trips_dict)
    print('Initial calculation ended.')

    # Initialise lists
    # total_cost is the cost of building bike paths in the network -- if we
    # start with an empty network, it is zero, otherwise it is the cost of
    # building every bike path in the network
    if rev:
        total_cost = [0]
    else:
        total_cost = [get_total_cost(edge_dict, street_cost, True, cost_method)]
    bike_lane_perc = [bike_lane_percentage(edge_dict)]
    total_real_distance_traveled = [total_len_on_types(trips_dict, 'real')]
    total_felt_distance_traveled = [total_len_on_types(trips_dict, 'felt')]
    nbr_on_street = [nbr_of_trips_on_street(trips_dict)]
    len_saved = [0]
    edited_edges = []
    edited_edges_nx = []
    edge_action = []
    cc_n, cc_size = get_connected_bike_components(nkG_edited)
    nbr_of_cbc = [cc_n]
    gcbc_size = [max(cc_size.values(), default=0)]

    # when we have found the optimal bike network satisfying the budget
    # constraints, set this to True
    budget_found = False

    K = 100
    log_idx = 0
    iter_log_counter = 0    #log every 100 iterations
    iter_log_nr = 0

    #initial building/removal of paths
    while (rev==(get_total_cost(edge_dict, street_cost,False,cost_method) < (2-w)*total_budget)) \
           or (rev == (get_total_cost(edge_dict, street_cost, False,cost_method) < w*total_budget)):
        #condition for the while loop:
        # if rev:
        #     #build bike paths till cost above (2-w)*budget
        #     condition = (get_total_cost(edge_dict, cost, bike_lanes_everywhere= False) < (2-w)*total_budget)
        # else:
        #     #remove bike paths till costs below w*budget
        #     condition = (get_total_cost(edge_dict, cost, bike_lanes_everywhere= False) < w*total_budget)
        
        # Calculate minimal loaded unedited edge:
        min_loaded_edge = get_minimal_loaded_edge(edge_dict, trips_dict,
                                                  minmode=minmode, rev=rev)


        # EDITING THE CHOSEN EDGE
        
        edited_edges.append(min_loaded_edge)
        edited_edges_nx.append(get_nx_edge(min_loaded_edge, nk2nx_edges))
        if rev: action = True
        else: action = False
        edge_action.append(action)
        remove_isolated_nodes(nkG_edited)
        # Calculate len of all trips running over min loaded edge.
        len_before = get_len_of_trips_over_edge(min_loaded_edge, edge_dict,
                                                trips_dict)
        # Calculate cost of "adding" bike lane
        this_edge_cost = get_cost(min_loaded_edge, edge_dict, street_cost, cost_method)
        if rev:
            # building a bike path -> increase total cost
            total_cost.append(total_cost[-1] + this_edge_cost)
        else:
            # removing a bike path -> decrease total cost
            total_cost.append(total_cost[-1] - this_edge_cost)
        # Edit minimal loaded edge and update edge_dict.
        edit_edge(nkG, edge_dict, min_loaded_edge)
        # Get all trips affected by editing the edge
        if rev:
            trips_recalc = deepcopy(trips_dict)
        else:
            trips_recalc = {trip: trips_dict[trip] for trip
                            in edge_dict[min_loaded_edge]['trips']}

        # CONSEQUENCES FOR THE NETWORK (don't change anything here)
        # Recalculate all affected trips and update their information.
        calc_trips(nkG, edge_dict, trips_recalc)
        trips_dict.update(trips_recalc)
        # Calculate length saved if not editing this edge.
        len_after = get_len_of_trips_over_edge(min_loaded_edge, edge_dict,
                                               trips_dict)
        len_saved.append(len_before - len_after)
        # Store all important data
        bike_lane_perc.append(bike_lane_percentage(edge_dict))
        total_real_distance_traveled.append(total_len_on_types(trips_dict,
                                                               'real'))
        total_felt_distance_traveled.append(total_len_on_types(trips_dict,
                                                               'felt'))
        nbr_on_street.append(nbr_of_trips_on_street(trips_dict))
        cc_n, cc_size = get_connected_bike_components(nkG_edited)
        nbr_of_cbc.append(cc_n)
        gcbc_size.append(max(cc_size.values(), default=0))


        # LOGGING (don't change anything here)
        iter_log_counter = (iter_log_counter +1)%K
        if iter_log_counter == 0:
            iter_log_nr += 1
            next_log = bike_lane_perc[-1]
            

            log_to_file(file=logfile, txt='Reached {0:3.2f} BLP'
                        .format(next_log), stamptime=time.localtime(),
                        start=starttime, end=time.time(), stamp=True,
                        difference=True)
            data = np.array([edited_edges, edited_edges_nx, total_cost,
                             bike_lane_perc, total_real_distance_traveled,
                             total_felt_distance_traveled, nbr_on_street,
                             len_saved, nbr_of_cbc, gcbc_size, edge_action])
            loc = 'data/algorithm/output/{0:}_data_mode_{1:d}{2:}_{3:02d}.npy'\
                .format(place, rev, minmode, iter_log_nr)
            mes = 'Saved at BLP {0:} as {1:}_data_mode_{2:d}{3:}_{4:02d}.npy'\
                .format(next_log, place, rev, minmode, iter_log_nr)
            save_data(loc, data, logfile, mes)


    rang = 0
    iter_edge_counter = 0
    iter_log_counter = 0
    iter_log_nr_loop1 = iter_log_nr
    run_times_loop = int(0.05 * len(edge_dict))
    
    #HERE implement the loop for the conditional building/removing of bike paths
    while True:

        """ HERE STARTS OUR JOB """
        
        
        
        # CHOOSING THE NEXT EDGES TO BE MODIFIED...!
        #decide whether to build a lane or not
        budget_decision = decide_building(total_budget, w, edge_dict, street_cost, cost_method)

        # Calculate minimal loaded unedited edge:
        min_loaded_edge = get_minimal_loaded_edge(edge_dict, trips_dict,
                                                  minmode=minmode, rev=False)
        # Calculate maximal loaded street
        max_loaded_street = get_minimal_loaded_edge(edge_dict, trips_dict,
                                                      minmode=minmode, rev=True)
        
        #  Choose Method
        if build_method == 'Monte Carlo':
            #action = ...               #bool: True-build, False-remove
            if budget_decision:
                chosen_edge = max_loaded_street
                action = True
            else:
                chosen_edge = min_loaded_edge
                action = False
        if (iter_log_nr-iter_log_nr_loop1)*K > run_times_loop:
            break

        if build_method == 'MFT' and (total_budget > get_total_cost(edge_dict,
                                        street_cost, False, cost_method)):
            most_frequented_trip = get_most_travelled_trip(trips_dict, rang)
            if iter_edge_counter == len(get_trip_edges(edge_dict, most_frequented_trip)):
                rang += 1
                iter_edge_counter = 0
            else: iter_edge_counter += 1

            sorted_edges = sort_edges_of_trip(most_frequented_trip, edge_dict, minmode, rev=True)
            chosen_edge = sorted_edges[iter_edge_counter]
            action = True
        if build_method == 'MFT' and (total_budget < get_total_cost(edge_dict,
                                        street_cost, False, cost_method)):
            break 
        
        #if build_method == 'Best BA':
            ##method to chose best bikeability
            #ba = [1 -(i -min(trdt['all'])) /(max(trdt['all']) -min(trdt['all'])) for i in trdt['all']]
            
        # EDITING THE CHOSEN EDGE
	
        if chosen_edge == 'We are done!':
            break
        
        edited_edges.append(chosen_edge)
        edited_edges_nx.append(get_nx_edge(chosen_edge, nk2nx_edges))
        edge_action.append(action)
        remove_isolated_nodes(nkG_edited)
        # Calculate len of all trips running over min loaded edge.
        len_before = get_len_of_trips_over_edge(chosen_edge, edge_dict,
                                                trips_dict)
        
        #print(edge_dict[chosen_edge]['bike lane'])

        # Edit minimal chosen edge and update edge_dict.
        edit_edge(nkG, edge_dict, chosen_edge)
        
        #print(edge_dict[chosen_edge]['bike lane'])





        """ HERE ENDS OUR JOB """


        # CONSEQUENCES FOR THE NETWORK
        # Calculate cost of "adding" bike lane
        this_edge_cost = get_cost(chosen_edge, edge_dict, street_cost, cost_method)
        if action == True:
            # building a bike path -> increase total cost
            total_cost.append(total_cost[-1] + this_edge_cost)
        elif action == False:
            # removing a bike path -> decrease total cost
            total_cost.append(total_cost[-1] - this_edge_cost)
        # Get all trips affected by editing the edge
        if rev:
            trips_recalc = deepcopy(trips_dict)
        else:
            trips_recalc = {trip: trips_dict[trip] for trip
                            in edge_dict[min_loaded_edge]['trips']}
        # Recalculate all affected trips and update their information.
        calc_trips(nkG, edge_dict, trips_recalc)
        trips_dict.update(trips_recalc)
        # Calculate length saved if not editing this edge.
        len_after = get_len_of_trips_over_edge(min_loaded_edge, edge_dict,
                                               trips_dict)
        len_saved.append(len_before - len_after)
        # Store all important data
        bike_lane_perc.append(bike_lane_percentage(edge_dict))
        total_real_distance_traveled.append(total_len_on_types(trips_dict,
                                                               'real'))
        total_felt_distance_traveled.append(total_len_on_types(trips_dict,
                                                               'felt'))
        nbr_on_street.append(nbr_of_trips_on_street(trips_dict))
        cc_n, cc_size = get_connected_bike_components(nkG_edited)
        nbr_of_cbc.append(cc_n)
        gcbc_size.append(max(cc_size.values(), default=0))


        # LOGGING (don't change anything here)
        iter_log_counter = (iter_log_counter +1)%K
        if iter_log_counter == 0:
            next_log = bike_lane_perc[-1]
            iter_log_nr += 1
            
            log_to_file(file=logfile, txt='Reached {0:3.2f} BLP'
                        .format(next_log), stamptime=time.localtime(),
                        start=starttime, end=time.time(), stamp=True,
                        difference=True)
            data = np.array([edited_edges, edited_edges_nx, total_cost,
                             bike_lane_perc, total_real_distance_traveled,
                             total_felt_distance_traveled, nbr_on_street,
                             len_saved, nbr_of_cbc, gcbc_size, edge_action])
            loc = 'data/algorithm/output/{0:}_data_mode_{1:d}{2:}_{3:02d}.npy'\
                .format(place, rev, minmode, iter_log_nr)
            mes = 'Saved at BLP {0:} as {1:}_data_mode_{2:d}{3:}_{4:02d}.npy'\
                .format(next_log, place, rev, minmode, iter_log_nr)
            save_data(loc, data, logfile, mes)


    # Save data of this run to data array
    data = np.array([edited_edges, edited_edges_nx, total_cost, bike_lane_perc,
                     total_real_distance_traveled,
                     total_felt_distance_traveled, nbr_on_street, len_saved,
                     nbr_of_cbc, gcbc_size, edge_action])
    return data


def run_simulation(place, logfile, mode):
    # Start date and time for logging.
    sd = time.localtime()
    starttime = time.time()

    rev = mode[0]
    minmode = mode[1]
    total_budget = mode[2]
    build_method = mode[3]
    w = mode[4]
    cost_method = mode[5]

    logfile = logfile + '_{:d}{:}.txt'.format(rev, minmode)

    log_to_file(logfile, 'Started optimising {} with minmode {} and reversed '
                         '{}'.format(place.capitalize(), minmode, rev),
                start=sd, stamp=False, difference=False)

    nxG = ox.load_graphml('{}.graphml'.format(place),
                          folder='data/algorithm/input', node_type=int)
    nxG = nx.Graph(nxG.to_undirected())
    print('Simulating "{}" with {} nodes and {} edges.'
          .format(place.capitalize(), len(nxG.nodes), len(nxG.edges)))

    trip_nbrs_nx = np.load('data/algorithm/input/{}_demand.npy'.format(place),
                           allow_pickle=True)[0]
    print('Number of trips: {}'.format(sum(trip_nbrs_nx.values())))
    # Exclude round trips
    trip_nbrs_nx = {trip_id: nbr_of_trips for trip_id, nbr_of_trips
                    in trip_nbrs_nx.items() if not trip_id[0] == trip_id[1]}
    print('Number of trips, round trips excluded: {}'.
          format(sum(trip_nbrs_nx.values())))

    # Convert networkx graph into network kit graph
    nkG = nk.nxadapter.nx2nk(nxG, weightAttr='length')
    nkG_edited = nk.nxadapter.nx2nk(nxG, weightAttr='length')
    nkG.removeSelfLoops()
    nkG_edited.removeSelfLoops()

    nx2nk_nodes = {list(nxG.nodes)[n]: n for n in range(len(list(nxG.nodes)))}
    nk2nx_nodes = {v: k for k, v in nx2nk_nodes.items()}
    nx2nk_edges = {(e[0], e[1]): (nx2nk_nodes[e[0]], nx2nk_nodes[e[1]])
                   for e in list(nxG.edges)}
    nk2nx_edges = {v: k for k, v in nx2nk_edges.items()}

    trip_nbrs_nk = {(nx2nk_nodes[k[0]], nx2nk_nodes[k[1]]): v
                    for k, v in trip_nbrs_nx.items()}

    # All street types in network
    street_types = get_all_street_types_cleaned(nxG)
    # Add bike lanes
    len_on_type = {t: 0 for t in street_types}
    len_on_type['primary'] = 0
    len_on_type['bike lane'] = 0

    # Set penalties for different street types
    penalties = {'primary': 7, 'secondary': 2.4, 'tertiary': 1.4,
                 'residential': 1.1}
    if rev:
        penalties = {k: 1 / v for k, v in penalties.items()}

    # Set cost for different street types
    street_cost = {'primary': 1, 'secondary': 1, 'tertiary': 1,
                   'residential': 1, 'cost per trip': 0.001}

    trips_dict = {t_id: {'nbr of trips': nbr_of_trips, 'nodes': [],
                         'edges': [], 'length real': 0, 'length felt': 0,
                         'real length on types': len_on_type,
                         'felt length on types': len_on_type,
                         'on street': False}
                  for t_id, nbr_of_trips in trip_nbrs_nk.items()}
    edge_dict = {edge: {'felt length': get_street_length(nxG, edge,
                                                         nk2nx_edges),
                        'real length': get_street_length(nxG, edge,
                                                         nk2nx_edges),
                        'street type': get_street_type_cleaned(nxG, edge,
                                                               nk2nx_edges),
                        'penalty': penalties[
                            get_street_type_cleaned(nxG, edge, nk2nx_edges)],
                        'speed limit': get_speed_limit(nxG, edge, nk2nx_edges),
                        'bike lane': not rev, 'load': 0, 'trips': []}
                 for edge in nkG.edges()}

    if rev:
        for edge, edge_info in edge_dict.items():
            edge_info['felt length'] *= 1 / edge_info['penalty']
            nkG.setWeight(edge[0], edge[1], edge_info['felt length'])
            
    # Calculate data
    data = edit_network(nkG, nkG_edited, edge_dict, trips_dict, nk2nx_nodes,
                        nk2nx_edges, street_cost, starttime, logfile, place,
                        minmode, rev, total_budget, build_method, w, cost_method)

    np.save('data/algorithm/output/{:s}_data_mode_{:d}{:d}.npy'
            .format(place, rev, minmode), data)

    # Print computation time to console and write it to the log.
    log_to_file(logfile, 'Finished optimising {0:s}o'
                .format(place.capitalize()), stamptime=time.localtime(),
                start=starttime, end=time.time(), stamp=True, difference=True)
