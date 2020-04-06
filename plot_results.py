"""

"""
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from matplotlib.ticker import AutoMinorLocator
from matplotlib.legend_handler import HandlerTuple
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import matplotlib.lines as mlines
import numpy as np
import osmnx as ox
import networkx as nx
import itertools as it
from math import ceil

from helper.current_state_helper import calc_current_state, \
    get_street_type_cleaned

from post_processing import load_graph, calculate_bikeability, load_data


def len_of_bikepath_by_type(ee, G, rev=False):
    """
    Calculates the length of the bike paths along street types during
    calculation.
    :param ee: list of edited edges.
    :param G: networkx graph
    :param rev: if building set it True
    :return: 
    """
    street_types = ['primary', 'secondary', 'tertiary', 'residential']
    total_len = {k: 0 for k in street_types}
    for e in G.edges():
        st = get_street_type_cleaned(G, e)
        total_len[st] += G[e[0]][e[1]][0]['length']
    len_fraction = {k: [0] for k in street_types}
    if not rev:
        ee = list(reversed(ee))
    for e in ee:
        st = get_street_type_cleaned(G, e)
        len_before = len_fraction[st][-1]
        len_fraction[st].append(len_before + G[e[0]][e[1]][0]['length'] /
                                total_len[st])
        for s in [s for s in street_types if s != st]:
            len_fraction[s].append(len_fraction[s][-1])
    return len_fraction


def coord_transf(x, y, xmin=-0.05, xmax=1.05, ymin=-0.05, ymax=1.05):
    """
    Coordination transformation vor h and vlines. From data to axis
    coordinates.
    :param x: data x
    :param y: data y
    :param xmin: axis x min
    :param xmax: axis x max
    :param ymin: axis y min
    :param ymax: axis y max
    :return: axis x, axis y
    """
    return (x - xmin) / (xmax - xmin), (y - ymin) / (ymax - ymin)


def plot_algorithm(place, mode, file_format='png',
                   slice_by='iteration'):
    """
    Plot evolution of bike lanes, in the order the algorithm added / removed
    them.
    :param place: name of the city, will only be used for determining .npy file
    name
    :param mode: mode of the simulation, will only be used for determining .npy
    file name
    :param file_format: what to save the resulting plots as
    :param slice_by: Strategy for choosing when to plot. If
    select_by='iteration', every nth iteration will be plotted, such that in
    total 100 plots result. If select_by='bike lane delta', plotted iterations
    will be chosen such that between plots, roughly the same amount of bike lane
    have been added or removed.
    """
    G = load_graph(place, mode)
    data = load_data(place, mode)
    edited_edges_nx = data['edited edges nx']
    bike_lane_perc = data['bike lane perc']
    action = data['edge action']
    num_iters = len(edited_edges_nx)
    # the 'bike lane' attribute is one of the following here:
    # 'added' -- bike lane was added between plots
    # 'removed' -- bike lane was removed between plots
    # 'not present' -- no bike lane on this edge
    # 'present' -- bike lane on this edge, did not change between plots
    nx.set_edge_attributes(G, 'present', 'bike lane')

    if slice_by == 'iteration':
        plot_at = np.linspace(0, num_iters - 1, num=101, dtype=np.int64)

    color_dict = {
        'present': '#0000FF',
        'not present': '#999999',
        'added': '#00FF00',
        'removed': '#FF0000'
    }

    for i, idx in enumerate(plot_at):
        print("Iter {}!".format(i))
        last_idx = plot_at[i - 1] if i > 0 else 0
        # changes in edited edges
        ee_changes = edited_edges_nx[last_idx:idx]
        # actions for these edited edges
        ac_changes = action[last_idx:idx]

        # 'added' -> 'present', 'removed' -> 'not present'
        # because G is a multigraph, it would be more accurate to also iterate
        # over the edge keys (because there might be multiple edges from node i
        # to node j) -- however, edited_edges_nx does not save keys anyways, so
        # we only ever edit edges with key 0
        for u, v, lane in G.edges(data='bike lane'):
            if lane == 'added':
                G.edges[u, v, 0]['bike lane'] = 'present'
            if lane == 'removed':
                G.edges[u, v, 0]['bike lane'] = 'not present'
        # update graph
        for changed_edge, changed_edge_action in zip(ee_changes, ac_changes):
            if changed_edge_action:
                G.edges[(*changed_edge, 0)]['bike lane'] = 'added'
            else:
                G.edges[(*changed_edge, 0)]['bike lane'] = 'removed'

        edge_color = [color_dict[data] for u, v, data
                      in G.edges.data('bike lane')]
        fig, ax = ox.plot_graph(G, edge_color=edge_color, fig_height=6,
                                fig_width=6, dpi=300, show=False, close=False)
        fig.suptitle('Iteration: {}'.format(idx), fontsize='x-large')
        plt.savefig('plots/evolution/{}-iter-{:04d}-mode-{:d}{}{}{}.{}'
                    .format(place, i,
                            mode[0], mode[1], mode[3], mode[5],
                            file_format))
        plt.close(fig)


def plot_edited_edges(G, place, edited_edges, bike_lane_perc, node_size,
                      rev=False, minmode=1, file_format='png'):
    """
    Plots evolution of edited edges.
    :param G: networkx graph
    :param place: name of the city
    :param edited_edges: list of edited edges
    :param bike_lane_perc: list of bike lane perc
    :param node_size: list of node sizes
    :param rev: if building set it True
    :param minmode: load weighting mode
    :param file_format: foramt of the pictures
    :return: None
    """
    if rev:
        nx.set_edge_attributes(G, False, 'bike lane')
        ee = edited_edges
        blp = bike_lane_perc
    else:
        nx.set_edge_attributes(G, False, 'bike lane')
        ee = list(reversed(edited_edges))
        blp = bike_lane_perc

    plots = np.linspace(0, 1, 101)
    for i, j in enumerate(plots):
        idx = next(x for x, val in enumerate(blp) if val >= j)
        ee_cut = ee[:idx]
        for edge in ee_cut:
            G[edge[0]][edge[1]][0]['bike lane'] = True
            G[edge[1]][edge[0]][0]['bike lane'] = True
        ec = ['#0000FF' if data['bike lane'] else '#999999' for
              u, v, data in G.edges(keys=False, data=True)]
        fig, ax = ox.plot_graph(G, node_size=node_size, node_color='C0',
                                edge_color=ec, fig_height=6, fig_width=6,
                                node_zorder=3, dpi=300, show=False,
                                close=False)
        fig.suptitle('Bike Lane Percentage: {0:.0%}'.format(blp[idx]),
                     fontsize='x-large')
        plt.savefig('plots/evolution/{0:s}-edited-mode-{1:d}{2:}-{3:03d}'
                    '.{4:s}'.format(place, rev, minmode, i, file_format),
                    format='png')
        plt.close(fig)


def total_distance_traveled_list(total_dist, total_dist_now, rev):
    """
    Normalises the total distance driven structured by street type
    :param total_dist: unnormalised total distance
    :param total_dist_now: total distance given by current state
    :param rev: if building set it True
    :return: normalised total distance dicts
    """
    if rev:
        s = 0
        e = -1
    else:
        s = -1
        e = 0
    dist = {}
    dist_now = {}

    # On all
    on_all = [i['total length on all'] for i in total_dist]
    dist_now['all'] = total_dist_now['total length on all'] / on_all[s]
    dist['all'] = [x / on_all[s] for x in on_all]
    # On streets w/o bike paths
    on_street = [i['total length on street'] for i in total_dist]
    dist_now['street'] = total_dist_now['total length on street'] / \
                         on_street[s]
    dist['street'] = [x / on_street[s] for x in on_street]
    # On primary
    on_primary = [i['total length on primary'] for i in total_dist]
    dist_now['primary'] = total_dist_now['total length on primary'] / \
                          on_primary[s]
    dist['primary'] = [x / on_primary[s] for x in on_primary]
    # On secondary
    on_secondary = [i['total length on secondary'] for i in total_dist]
    dist_now['secondary'] = total_dist_now['total length on secondary'] / \
                            on_secondary[s]
    dist['secondary'] = [x / on_secondary[s] for x in on_secondary]
    # On tertiary
    on_tertiary = [i['total length on tertiary'] for i in total_dist]
    dist_now['tertiary'] = total_dist_now['total length on tertiary'] / \
                           on_tertiary[s]
    dist['tertiary'] = [x / on_tertiary[s] for x in on_tertiary]
    # On residential
    on_residential = [i['total length on residential'] for i in total_dist]
    dist_now['residential'] = total_dist_now['total length on residential'] / \
                              on_residential[s]
    dist['residential'] = [x / on_residential[s] for x in on_residential]
    # On bike paths
    on_bike = [i['total length on bike lanes'] for i in total_dist]
    dist_now['bike lanes'] = total_dist_now['total length on bike lanes'] / \
                        on_bike[e]
    dist['bike lanes'] = [x / on_bike[e] for x in on_bike]
    if not rev:
        for st, len_on_st in dist.items():
            dist[st] = list(reversed(len_on_st))
    return dist, dist_now


def get_total_cost(cost, cost_now, rev):
    """
    Calculates and normalises total cost
    :param cost: list of cost per step
    :param cost_now: cost for current state
    :param rev: if building set it True
    :return: total cost for algorithm and current state
    """
    if not rev:
        cost = list(reversed(cost))  # costs per step
    total_cost = [sum(cost[:i]) for i in range(1, len(cost) + 1)]
    cost_now = cost_now / total_cost[-1]
    total_cost = [i / total_cost[-1] for i in total_cost]
    return total_cost, cost_now


def get_end(tdt, tdt_now, rev):
    """
    Returns the step at which bikeability reaches 1.
    :param tdt: total distance driven raw
    :param tdt_now: total distance driven current state raw
    :param rev: if building set it True
    :return: step at which ba=1
    """
    tdt, tdt_now = total_distance_traveled_list(tdt, tdt_now, rev)
    ba = [1 - (i - min(tdt['all'])) / (max(tdt['all']) - min(tdt['all']))
          for i in tdt['all']]
    return next(x for x, val in enumerate(ba) if val >= 1)


def plot_used_nodes(G, trip_nbrs, stations, place, mode):
    """
    Plots usage of nodes
    :param G: networkx graph
    :param trip_nbrs: number of trips dict
    :param stations: stations set
    :param place: name of the city
    :param mode:
    :return: none
    """
    rev = mode[0]
    minmode = mode[1]

    nodes = {n: 0 for n in G.nodes()}
    for s_node in G.nodes():
        for e_node in G.nodes():
            if (s_node, e_node) in trip_nbrs:
                nodes[s_node] += trip_nbrs[(s_node, e_node)]
                nodes[e_node] += trip_nbrs[(s_node, e_node)]

    max_n = max(nodes.values())
    n_rel = {key: ceil((value / max_n) * 100) for key, value in nodes.items()}
    ns = [100 if n in stations else 0 for n in G.nodes()]

    for n in G.nodes():
        if n not in stations:
            n_rel[n] = 101

    cmap_name = 'cool'
    cmap = plt.cm.get_cmap(cmap_name)
    cmap = ['#999999'] + \
           [rgb2hex(cmap(n)) for n in np.linspace(1, 0, 100, endpoint=False)] \
           + ['#ffffff']
    color_n = [cmap[v] for k, v in n_rel.items()]

    fig, ax = ox.plot_graph(G, fig_height=15, fig_width=15, dpi=300,
                            edge_linewidth=2, node_color=color_n,
                            node_size=ns, node_zorder=3, show=False,
                            close=False)
    # ax.set_title('Nodes used as stations in {}'.format(place))
    sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(cmap_name),
                               norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    cbaxes = fig.add_axes([0.1, 0.075, 0.8, 0.03])
    cbar = fig.colorbar(sm, orientation='horizontal', cax=cbaxes)
    cbar.ax.tick_params(axis='x', labelsize=18)
    cbar.ax.set_xlabel('normalised usage of stations', fontsize=24)
    fig.suptitle('Nodes used as Stations in {}'.format(place.capitalize()),
                 fontsize=30)
    plt.savefig('plots/{:}_stations_used.png'.format(place),
                format='png')

    plt.close('all')
    # plt.show()


def plot_mode(data, data_now, nxG, place, stations, trip_nbrs, mode, end):
    """
    Plots the observed values for one mode.
    :param data: data given by algorithm
    :param data_now: data given for current state
    :param nxG: networkx graph
    :param place: name of the city
    :param stations: set of stations
    :param trip_nbrs:
    :param mode: mode to plot
    :param end: plot to this algorithm step.
    :return: None
    """
    rev = mode[0]
    minmode = mode[1]

    bp_now = data_now[0]
    cost_now = data_now[1]
    blp_now = data_now[2]
    trdt_now = data_now[3]
    tfdt_now = data_now[4]
    nos_now = data_now[5]

    # edited_edges = data[0]
    edited_edges_nx = data[1]
    cost = data[2]
    bike_lane_perc = data[3]
    total_real_distance_traveled = data[4]
    total_felt_distance_traveled = data[5]
    nbr_on_street = data[6]
    # len_saved = data[7]
    # nbr_of_cbc = data[8]
    # gcbc_size = data[9]

    trdt, trdt_now = total_distance_traveled_list(total_real_distance_traveled,
                                         trdt_now, rev)
    tfdt, tfdt_now = total_distance_traveled_list(total_felt_distance_traveled,
                                         tfdt_now, rev)

    bl_st = len_of_bikepath_by_type(edited_edges_nx, nxG, rev)
    bl_st_now = len_of_bikepath_by_type(bp_now, nxG, rev)
    bl_st_now = {st: length[-1] for st, length in bl_st_now.items()}

    if rev:
        blp = bike_lane_perc
    else:
        blp = list(reversed(bike_lane_perc))
    ba = [1 - (i - min(trdt['all'])) / (max(trdt['all']) - min(trdt['all']))
          for i in trdt['all']]
    ba_now = 1 - (trdt_now['all'] - min(trdt['all'])) / \
             (max(trdt['all']) - min(trdt['all']))

    if rev:
        nos = [x / max(nbr_on_street) for x in nbr_on_street]
    else:
        nos = list(reversed([x / max(nbr_on_street) for x in nbr_on_street]))
    nos_now = nos_now / max(nbr_on_street)
    los = trdt['street']
    los_now = trdt_now['street']

    trdt_st = {st: len_on_st for st, len_on_st in trdt.items()
               if st not in ['street', 'all']}
    trdt_st_now = {st: len_on_st for st, len_on_st in trdt_now.items()
                   if st not in ['street', 'all']}

    blp_cut = [i / blp[end] for i in blp[:end]]
    blp_now = blp_now / blp[end]

    blp_x = min(blp_cut, key=lambda x: abs(x-blp_now))
    blp_idx = next(x for x, val in enumerate(blp_cut) if val == blp_x)
    ba_y = ba[blp_idx]

    cost_y = min(cost[:end], key=lambda x: abs(x-cost_now))
    cost_idx = next(x for x, val in enumerate(cost[:end]) if val == cost_y)
    cost_x = blp_cut[cost_idx]

    nos_y = min(nos[:end], key=lambda x: abs(x-nos_now))
    nos_idx = next(x for x, val in enumerate(nos[:end]) if val == nos_y)
    nos_x = blp_cut[nos_idx]

    los_y = min(los[:end], key=lambda x: abs(x-los_now))
    los_idx = next(x for x, val in enumerate(los[:end]) if val == los_y)
    los_x = blp_cut[los_idx]

    cut = next(x for x, val in enumerate(ba) if val >= 1)
    total_cost, cost_now = get_total_cost(cost, cost_now, rev)

    cost_now = cost_now / total_cost[end]
    # gcbc_size_normed = [i / max(gcbc_size) for i in reversed(gcbc_size)]

    ns = [30 if n in stations else 0 for n in nxG.nodes()]

    print('Mode: {:d}{:d}, ba=1 after: {:d}, blp at ba=1: {:3.2f}, '
          'blp at cut: {:3.2f}, blp big roads: {:3.2f}, edges: {:}'
          .format(rev, minmode, cut, blp[cut], blp[end], blp_now * blp[end],
                  len(edited_edges_nx)))

    # Plotting
    fig1, ax1 = plt.subplots(dpi=300)
    ax12 = ax1.twinx()
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax12.set_ylim(-0.05, 1.05)

    c_ba = 'C0'

    ax1.plot(blp_cut, ba[:end], c=c_ba, label='bikeability')
    ax1.plot(blp_now, ba_now, c=c_ba, marker='D')
    xmax, ymax = coord_transf(blp_now, max([ba_y, ba_now]))
    ax1.axvline(x=blp_now, ymax=ymax, ymin=0, c=c_ba, ls='--', alpha=0.5)
    ax1.axhline(y=ba_now, xmax=xmax, xmin=0, c=c_ba, ls='--', alpha=0.5)
    ax1.axhline(y=ba_y, xmax=xmax, xmin=0, c=c_ba, ls='--', alpha=0.5)

    ax1.set_ylabel('bikeability', fontsize=12, color=c_ba)
    ax1.tick_params(axis='y', labelsize=12, labelcolor=c_ba)
    ax1.yaxis.set_minor_locator(AutoMinorLocator())

    c_cost = 'C8'

    ax12.plot(blp_cut, [x / total_cost[end] for x in total_cost[:end]],
              c=c_cost, label='total cost')
    ax12.plot(blp_now, cost_now, c=c_cost, marker='s')
    xmin, ymax = coord_transf(min(blp_now, cost_x), cost_now)
    ax1.axvline(x=blp_now, ymax=ymax, ymin=0, c=c_cost, ls='--', alpha=0.5)
    ax1.axhline(y=cost_now, xmax=1, xmin=xmin, c=c_cost, ls='--', alpha=0.5)
    ax1.axhline(y=cost_y, xmax=xmax, xmin=0, c=c_cost, ls='--', alpha=0.5)

    ax12.set_ylabel('cost', fontsize=12, color=c_cost)
    ax12.tick_params(axis='y', labelsize=12, labelcolor=c_cost)
    ax12.yaxis.set_minor_locator(AutoMinorLocator())

    ax1.set_xlabel('normalised fraction of bike paths', fontsize=12)
    ax1.set_title('Bikeability and Cost', fontsize=14)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(axis='x', labelsize=12)

    handles = ax1.get_legend_handles_labels()[0]
    handles.append(ax12.get_legend_handles_labels()[0][0])
    ax1.legend(handles=handles, loc='lower right')

    plt.savefig('plots/{:}_ba_tc_mode_{:d}{:}.png'.format(place, rev, minmode),
                format='png')

    ax1ins = zoomed_inset_axes(ax1, 3.5, loc=1)
    x1, x2, y1, y2 = round(blp_now - 0.05, 2), round(blp_now + 0.05, 2), \
                     round(ba_now - 0.03, 2), round(ba_y + 0.03, 2)
    ax1ins.plot(blp_cut, ba[:end])
    ax1ins.plot(blp_now, ba_now, c=c_ba, marker='D')
    xmax, ymax = coord_transf(blp_now, max([ba_y, ba_now]), xmin=x1, xmax=x2,
                              ymin=y1, ymax=y2)
    ax1ins.axvline(x=blp_now, ymax=ymax, ymin=0, c=c_ba, ls='--', alpha=0.5)
    ax1ins.axhline(y=ba_now, xmax=xmax, xmin=0, c=c_ba, ls='--', alpha=0.5)
    ax1ins.axhline(y=ba_y, xmax=xmax, xmin=0, c=c_ba, ls='--', alpha=0.5)

    ax1ins.set_xlim(x1, x2)
    ax1ins.set_ylim(y1, y2)
    ax1ins.tick_params(axis='y', labelsize=8, labelcolor=c_ba)
    ax1ins.tick_params(axis='x', labelsize=8)
    ax1ins.yaxis.set_minor_locator(AutoMinorLocator())
    ax1ins.xaxis.set_minor_locator(AutoMinorLocator())
    # ax1ins.set_yticklabels(labels=ax1ins.get_yticklabels(), visible=False)
    # ax1ins.set_xticklabels(labels=ax1ins.get_xticklabels(), visible=False)
    mark_inset(ax1, ax1ins, loc1=2, loc2=3, fc="none", ec="0.7")

    plt.savefig('plots/{:}_ba_tc_zoom_mode_{:d}{:}.png'.format(place, rev,
                                                               minmode),
                format='png')

    fig2, ax2 = plt.subplots(dpi=300)
    ax22 = ax2.twinx()
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax22.set_ylim(-0.05, 1.05)

    c_nos = 'C1'
    c_los = 'm'

    p1, = ax2.plot(blp_cut, los[:end], label='length', c=c_los)
    ax2.plot(blp_now, los_now, c=c_los, marker='8')
    xmax, ymax = coord_transf(max(blp_now, los_x), los_now)
    ax2.axvline(x=blp_now, ymax=ymax, ymin=0, c=c_los, ls='--', alpha=0.5)
    ax2.axhline(y=los_now, xmax=xmax, xmin=0, c=c_los, ls='--', alpha=0.5)
    ax2.axvline(x=los_x, ymax=ymax, ymin=0, c=c_los, ls='--', alpha=0.5)

    ax2.set_ylabel('length of trips', fontsize=12, color=c_los)
    ax2.tick_params(axis='y', labelsize=12, labelcolor=c_los)
    ax2.yaxis.set_minor_locator(AutoMinorLocator())

    p2, = ax22.plot(blp_cut, nos[:end], label='trips', c=c_nos)
    ax22.plot(blp_now, nos_now, c=c_nos, marker='v')
    xmin, ymax = coord_transf(min(blp_now, nos_x), nos_now)
    ax22.axvline(x=blp_now, ymax=ymax, ymin=0, c=c_nos, ls='--', alpha=0.5)
    ax22.axhline(y=nos_now, xmax=1, xmin=xmin, c=c_nos, ls='--', alpha=0.5)
    ax22.axvline(x=nos_x, ymax=ymax, ymin=0, c=c_nos, ls='--', alpha=0.5)

    ax22.set_ylabel('number of trips', fontsize=12, color=c_nos)
    ax22.tick_params(axis='y', labelsize=12, labelcolor=c_nos)
    ax22.yaxis.set_minor_locator(AutoMinorLocator())

    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.tick_params(axis='x', labelsize=12)
    ax2.set_xlabel('normalised fraction of bike paths', fontsize=12)
    ax2.set_title('Number of Trips and Length on Street', fontsize=14)
    ax2.legend([p1, p2], [l.get_label() for l in [p1, p2]])
    plt.savefig('plots/{:}_trips_on_street_mode_{:d}{:}.png'.format(place, rev,
                                                                    minmode),
                format='png')

    fig3, ax3 = plt.subplots(dpi=300)
    ax3.set_xlim(-0.05, 1.05)
    ax3.set_ylim(-0.05, 1.15)

    c_st = {'primary': 'darkblue', 'secondary': 'darkgreen',
            'tertiary': 'darkcyan', 'residential': 'darkorange',
            'bike lanes': 'gold'}
    m_st = {'primary': 'p', 'secondary': 'p', 'tertiary': 'p',
            'residential': 'p', 'bike lanes': 'P'}

    for st, len_on_st in trdt_st_now.items():
        xmax, ymax = coord_transf(blp_now, len_on_st, ymax=1.15)
        ax3.axvline(x=blp_now, ymax=ymax, ymin=0, c=c_st[st], ls='--',
                    alpha=0.5)
        ax3.axhline(y=len_on_st, xmax=xmax, xmin=0, c=c_st[st], ls='--',
                    alpha=0.5)

    for st, len_on_st in trdt_st.items():
        ax3.plot(blp_cut, len_on_st[:end], c=c_st[st], label=st)
    for st, len_on_st in trdt_st.items():
        ax3.plot(blp_now, trdt_now[st], c=c_st[st], marker=m_st[st])

    ax3.set_xlabel('% of bike paths by length', fontsize=12)
    ax3.set_ylabel('length', fontsize=12)
    ax3.set_title('Length on Street'.format(rev, minmode), fontsize=14)
    ax3.tick_params(axis='x', labelsize=12)
    ax3.tick_params(axis='y', labelsize=12)
    ax3.xaxis.set_minor_locator(AutoMinorLocator())
    ax3.yaxis.set_minor_locator(AutoMinorLocator())
    ax3.legend()
    plt.savefig('plots/{:}_len_on_street_mode_{:d}{:}.png'
                .format(place, rev, minmode), format='png')

    fig4, ax4 = plt.subplots(dpi=300)
    ax4.set_xlim(-0.05, 1.05)
    ax4.set_ylim(-0.05, 1.05)

    for st, len_by_st in bl_st.items():
        ax4.plot(blp_cut, [x / len_by_st[end] for x in len_by_st[:end]],
                 c=c_st[st], label='{}'.format(st))

    ax4.axvline(x=blp[cut]/blp[end], c='#999999', ls='--', alpha=0.5)

    ax4.set_xlabel('normalised fraction of bike paths', fontsize=12)
    ax4.set_ylabel('length', fontsize=12)
    ax4.set_title('Length of Bike Paths along Streets'
                  .format(rev, minmode), fontsize=14)
    ax4.tick_params(axis='x', labelsize=12)
    ax4.tick_params(axis='y', labelsize=12)
    ax4.xaxis.set_minor_locator(AutoMinorLocator())
    ax4.yaxis.set_minor_locator(AutoMinorLocator())
    ax4.legend()
    plt.savefig('plots/{:}_len_bl_mode_{:d}{:}.png'
                .format(place, rev, minmode), format='png')

    plot_used_nodes(nxG, trip_nbrs, stations, place, mode)

    if (not rev) and (minmode == 1):
        plot_edited_edges(nxG, place, edited_edges_nx, blp, ns, rev, minmode,
                          file_format='png')

    # plt.show()
    plt.close('all')

    return blp_cut, ba[:end], total_cost[:end], nos[:end], los[:end], \
           blp_now, ba_now, cost_now, nos_now, los_now


def compare_modes(place, label, blp, ba, cost, nos, los, blp_now, ba_now,
                  cost_now, nos_now, los_now, color):
    """
    Compares the modes for one city
    :param place: city name
    :param label:
    :param blp:
    :param ba:
    :param cost:
    :param nos:
    :param los:
    :param blp_now:
    :param ba_now:
    :param cost_now:
    :param nos_now:
    :param los_now:
    :param color:
    :return:
    """
    fig1, ax1 = plt.subplots(dpi=300)
    ax1.set_xlabel('normalised fraction of bike paths', fontsize=12)
    ax1.set_ylabel('bikeability', fontsize=12)
    ax1.set_title('Bikeability of {}'.format(place.capitalize()),
                  fontsize=14)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.set_ylim(bottom=-0.05, top=1.05)

    fig2, ax2 = plt.subplots(dpi=300)
    ax2.set_xlabel('normalised fraction of bike paths', fontsize=12)
    ax2.set_ylabel('integrated bikeability', fontsize=12)
    ax2.set_title('Integrated Bikeability of {}'.format(place.capitalize()),
                  fontsize=14)
    ax2.tick_params(axis='x', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.set_ylim(bottom=-0.05, top=1.05)

    fig3, ax3 = plt.subplots(dpi=300)
    ax3.set_xlabel('normalised fraction of bike paths', fontsize=12)
    ax3.set_ylabel('ba per cost', fontsize=12)
    ax3.set_title('CBA of {}'.format(place.capitalize()), fontsize=14)
    ax3.tick_params(axis='x', labelsize=12)
    ax3.tick_params(axis='y', labelsize=12)
    ax3.xaxis.set_minor_locator(AutoMinorLocator())
    ax3.yaxis.set_minor_locator(AutoMinorLocator())
    ax3.set_ylim(bottom=-0.05, top=1.05)

    fig4, ax4 = plt.subplots(dpi=300)
    ax42 = ax4.twinx()
    ax4.set_xlim(-0.05, 1.05)
    ax4.set_ylim(-0.05, 1.05)
    ax42.set_ylim(-0.05, 1.05)

    ax4.set_ylabel('length on street', fontsize=12, color='orangered')
    ax4.tick_params(axis='y', labelsize=12, labelcolor='orangered')
    ax4.yaxis.set_minor_locator(AutoMinorLocator())

    ax42.set_ylabel('number of trips', fontsize=12, color='mediumblue')
    ax42.tick_params(axis='y', labelsize=12, labelcolor='mediumblue')
    ax42.yaxis.set_minor_locator(AutoMinorLocator())

    ax4.set_title('Trips and Length on Street in {}'
                  .format(place.capitalize()), fontsize=14)
    ax4.set_xlabel('normalised fraction of bike paths', fontsize=12)
    ax4.tick_params(axis='x', labelsize=12)
    ax4.xaxis.set_minor_locator(AutoMinorLocator())

    ax4_hand = {}

    for m in blp.keys():
        bikeab = [np.trapz(ba[m][:idx], blp[m][:idx]) for idx in
                  range(len(blp[m]))]
        cba = [bikeab[idx] / cost[m][idx] for idx in
               range(len(blp[m]))]
        ax1.plot(blp[m], ba[m], color=color[m], label=label[m])
        ax2.plot(blp[m], bikeab, color=color[m],  label=label[m])
        ax3.plot(blp[m], cba, color=color[m], label=label[m])
        space = round(len(blp[m]) / 20)
        ax4.plot(blp[m], nos[m], color=color[m], marker='v',  markevery=space,
                 label=label[m])
        ax42.plot(blp[m], los[m], color=color[m], marker='8', markevery=space,
                  label=label[m])
        ax4_hand[m] = mlines.Line2D([], [], color=color[m], label=label[m])

    ax1.plot(blp_now, ba_now, c='#999999', marker='D')
    xmax, ymax = coord_transf(blp_now, ba_now)
    ax1.axvline(x=blp_now, ymax=ymax, ymin=0, c='#999999', ls=':', alpha=0.5)
    ax1.axhline(y=ba_now, xmax=xmax, xmin=0, c='#999999', ls=':', alpha=0.5)

    ax4.plot(blp_now, nos_now, c='#999999', marker='v')
    ax4.plot(blp_now, los_now, c='#999999', marker='8')
    xmax, ymax = coord_transf(blp_now, nos_now)
    ax4.axvline(x=blp_now, ymax=ymax, ymin=0, c='#999999', ls=':', alpha=0.5)
    ax4.axhline(y=nos_now, xmax=1, xmin=xmax, c='#999999', ls=':', alpha=0.5)
    ax4.axhline(y=los_now, xmax=xmax, xmin=0, c='#999999', ls=':', alpha=0.5)

    l_keys_r = []
    l_keys_b = []
    for mode, l_key in ax4_hand.items():
        if not mode[0]:
            l_keys_r.append(l_key)
        else:
            l_keys_b.append(l_key)

    l_keys = [tuple(l_keys_r)] + [tuple(l_keys_b)]
    l_labels = ['Removing', 'Building']

    ax1.legend(l_keys, l_labels, numpoints=1,
               handler_map={tuple: HandlerTuple(ndivide=None)})
    ax2.legend(l_keys, l_labels, numpoints=1,
               handler_map={tuple: HandlerTuple(ndivide=None)})
    ax3.legend(l_keys, l_labels, numpoints=1,
               handler_map={tuple: HandlerTuple(ndivide=None)})

    l_keys.append(mlines.Line2D([], [], color='k', marker='v', label='trips'))
    l_labels.append('trips')
    l_keys.append(mlines.Line2D([], [], color='k', marker='8', label='length'))
    l_labels.append('length')
    ax4.legend(l_keys, l_labels, numpoints=1,
               handler_map={tuple: HandlerTuple(ndivide=None)}, loc=1)

    fig1.savefig('plots/{}-1.png'.format(place), format='png')
    fig2.savefig('plots/{}-2.png'.format(place), format='png')
    fig3.savefig('plots/{}-3.png'.format(place), format='png')
    fig4.savefig('plots/{}-4.png'.format(place), format='png')

    plt.close('all')
    # plt.show()


def plot_city(place, modes):
    """
    Plot all data for one city.
    :param place: name of the city
    :param modes: modes of the algorithm
    :return:
    """
    trip_nbrs = np.load('data/algorithm/input/{}_demand.npy'.format(place),
                        allow_pickle=True)[0]
    trip_nbrs_re = {trip_id: nbr_of_trips for trip_id, nbr_of_trips
                    in trip_nbrs.items() if not trip_id[0] == trip_id[1]}
    trips = sum(trip_nbrs.values())
    trips_re = sum(trip_nbrs_re.values())
    utrips = len(trip_nbrs.keys())
    utrips_re = len(trip_nbrs_re.keys())

    stations = [station for trip_id, nbr_of_trips in trip_nbrs.items() for
                station in trip_id]
    stations = set(stations)

    print('Place: {}, stations: {},trips: {} (rt excl.: {}), '
          'unique trips: {} (rt excl. {})'
          .format(place, len(stations), trips, trips_re, utrips, utrips_re))

    nxG = ox.load_graphml('{}.graphml'.format(place),
                          folder='data/algorithm/input', node_type=int)
    nxG = nxG.to_undirected()

    data_now = calc_current_state(nxG, trip_nbrs)

    data = {}
    for m in modes:
        data[m] = np.load('data/algorithm/output/{:}_data_mode_{:d}{:}{}{}.npy'
                          .format(place, m[0], m[1], mode[3], mode[5]),
                          allow_pickle=True)

    end = max([get_end(d[4], data_now[3], m[0]) for m, d in data.items()])
    print('Cut after: ', end)

    c_norm = ['darkblue', 'mediumblue', 'cornflowerblue']
    c_rev = ['red', 'orangered', 'indianred']

    c = {}
    for m, d in data.items():
        if m[0]:
            c[m] = c_rev[m[1]]
        else:
            c[m] = c_norm[m[1]]

    blp = {}  # bike lane percentage
    ba = {}  # bikeability
    cost = {}
    nos = {}
    los = {}
    blp_now, ba_now, cost_now, nos_now, los_now = 0, 0, 0, 0, 0

    for m, d in data.items():
        blp[m], ba[m], cost[m], nos[m], los[m], blp_now, ba_now, cost_now, \
        nos_now, los_now = \
            plot_mode(d, data_now, nxG, place, stations, trip_nbrs, m, end)

    save = [blp, ba, cost, nos, los]
    np.save('data/plot/ba_comp_{}.npy'.format(place), save)

    label = {m: 'Removing' if not m[0] else 'Building' for m in modes}
    compare_modes(place, label, blp, ba, cost, nos, los, blp_now, ba_now,
                  cost_now, nos_now, los_now, color=c)


def compare_cities(cities, mode, color):
    """
    Compare cities for given mode
    :param cities: list of cities
    :param mode: mode to compare
    :param color: colors for plots
    :return: None
    """
    rev = mode[0]
    minmode = mode[1]

    blp = {}
    ba = {}
    cost = {}
    nos = {}
    los = {}

    for city in cities:
        data = np.load('data/plot/ba_comp_{}.npy'.format(city),
                       allow_pickle=True)
        blp[city] = data[0][mode]
        ba[city] = data[1][mode]
        cost[city] = data[2][mode]
        nos[city] = data[3][mode]
        los[city] = data[4][mode]

    fig1, ax1 = plt.subplots(dpi=300)
    ax1.set_xlabel('normalised fraction of bike paths', fontsize=12)
    ax1.set_ylabel('bikeability', fontsize=12)
    ax1.set_title('Comparison of Bikeabilities', fontsize=14)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.set_ylim(bottom=-0.05, top=1.05)

    fig2, ax2 = plt.subplots(dpi=300)
    ax2.set_xlabel('normalised fraction of bike paths', fontsize=12)
    ax2.set_ylabel('integrated bikeability', fontsize=12)
    ax2.set_title('Comparison of Integrated Bikeabilities', fontsize=14)
    ax2.tick_params(axis='x', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.set_ylim(bottom=-0.05, top=1.05)

    fig3, ax3 = plt.subplots(dpi=300)
    ax32 = ax3.twinx()
    ax3.set_xlim(-0.05, 1.05)
    ax3.set_ylim(-0.05, 1.05)
    ax32.set_ylim(-0.05, 1.05)

    ax3.set_ylabel('number of trips', fontsize=12)
    ax3.tick_params(axis='y', labelsize=12)
    ax3.yaxis.set_minor_locator(AutoMinorLocator())

    ax32.set_ylabel('length on street', fontsize=12)
    ax32.tick_params(axis='y', labelsize=12)
    ax32.yaxis.set_minor_locator(AutoMinorLocator())

    ax3.set_title('Comaprison of Trips and Length on Street', fontsize=14)
    ax3.set_xlabel('normalised fraction of bike paths', fontsize=12)
    ax3.tick_params(axis='x', labelsize=12)
    ax3.xaxis.set_minor_locator(AutoMinorLocator())

    ax3_hand = {}

    for city in cities:
        bikeab = [np.trapz(ba[city][:idx], blp[city][:idx]) for idx in
                  range(len(blp[city]))]
        ax1.plot(blp[city], ba[city], color=color[city],
                 label='{:s}'.format(city.capitalize()))
        ax2.plot(blp[city], bikeab, color=color[city],
                 label='{}'.format(city.capitalize()))
        space = round(len(blp[city]) / 25)

        ax3.plot(blp[city], nos[city], marker='v', markevery=space,
                 color=color[city])
        ax32.plot(blp[city], los[city], marker='8', markevery=space,
                  color=color[city])
        ax3_hand[city] = mlines.Line2D([], [], color=color[city])

    l_keys = [l_key for city, l_key in ax3_hand.items()]
    l_cities = [city.capitalize() for city, l_key in ax3_hand.items()]
    l_keys.append(mlines.Line2D([], [], color='k', marker='v', label='trips'))
    l_cities.append('trips')
    l_keys.append(mlines.Line2D([], [], color='k', marker='8', label='length'))
    l_cities.append('length')

    ax1.legend(loc='lower right')
    ax2.legend(loc='lower right')
    ax3.legend(l_keys, l_cities, numpoints=1,
               handler_map={tuple: HandlerTuple(ndivide=None)})

    fig1.savefig('plots/comparison-1-{:d}{:}.png'.format(rev, minmode))
    fig2.savefig('plots/comparison-2-{:d}{:}.png'.format(rev, minmode))
    fig3.savefig('plots/comparison-3-{:d}{:}.png'.format(rev, minmode))

    plt.close('all')
    # plt.show()


def format_mode(mode):
    flags = []
    if mode[0]:
        flags.append('rev')
    flags.append('minmode ' + str(mode[1]))  # minmode
    # how to display build methods
    bm_dict = {
        0: 'Monte Carlo',
        1: 'MFT',
        2: 'heat'
    }
    flags.append(bm_dict[mode[3]])
    cost_dict = {
        0: 'equal cost',
        1: 'weighted cost'
    }
    flags.append(cost_dict[mode[5]])
    return ', '.join(flags)


def generic_mini_plot(data, modes, x_label, y_label, save=True):
    """
    Do a not-too-fancy plot of data and modes.
    :param x_label: the index for data to use for the x axis
    :param y_label: the index for data to use for the y axis
    :return: fig, ax
    """
    fig, ax = plt.subplots()
    for m in modes:
        ax.plot(data[m][x_label], data[m][y_label], label=format_mode(m), alpha=0.8)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    if save:
        fname = ('plots/' +
                 x_label.replace(' ', '_') + '-' +
                 y_label.replace(' ', '_') + '.png')
        fig.savefig(fname)
        plt.close(fig)
    return fig, ax


def plot_city_mini(place, modes):
    """
    :param modes: list of modes
    """
    # data is a dictionary: mode -> data for mode
    data = {}
    for m in modes:
        data[m] = load_data(place, m)

    # cost - bikeability - plot
    generic_mini_plot(data, modes, 'total cost', 'bikeability')
    generic_mini_plot(data, modes, 'bike lane perc', 'bikeability')
    generic_mini_plot(data, modes, 'iteration', 'bikeability')
    generic_mini_plot(data, modes, 'iteration', 'total cost')
    generic_mini_plot(data, modes, 'total cost', 'total real length on bike lanes')
    generic_mini_plot(data, modes, 'total cost', 'total felt length on bike lanes')


def main():
    places = 'hh_part'
    # rev: False=Removing, True=Building
    minmodes = [1]
    rev = [False]
    #budget choice
    total_budget = [20000]

    # method choices
    build_method = [0,1]
    w = [0.9]
    cost_method = [0]


    modes = list(it.product(rev, minmodes, total_budget, build_method, 
                                   w, cost_method))


    plot_city_mini(places, modes)
    plot_algorithm(places, modes[0], file_format='png',slice_by='iteration')

    #for city in places:
     #   plot_city(city, modes)

    #colors = ['royalblue', 'orangered']
    #comp_color = {city: colors[idx] for idx, city in enumerate(places)}
    #for m in modes:
     #   compare_cities(places, m, comp_color)


if __name__ == '__main__':
    main()
