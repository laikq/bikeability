import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
import numpy as np
import json
import geog
from math import ceil
from shapely.geometry import Point, Polygon
import pandas as pd


def read_csv(path, delim):
    """
    Reads the csv given by path. Delimiter of csv can be chosen by delim.
    All column headers ar converted to lower case.
    :param path: path to load csv from
    :type path: str
    :param delim: delimiter of csv
    :type delim: str
    :return: data frame
    :rtype: pandas DataFrame
    """
    df = pd.read_csv(path, delimiter=delim)
    df.columns = map(str.lower, df.columns)
    return df


def write_csv(df, path):
    """
    Writes given data frame to csv.
    :param df: data frame
    :type df: pandas DataFrame
    :param path: path to save
    :type path: str
    :return: None
    """
    df.to_csv(path, index=False)


def get_circle_from_point(lat, long, radius, n_points=20):
    """
    Returns a circle around a lat/long point with given radius.
    :param lat: latitude of point
    :type lat: float
    :param long: longitude of point
    :type long: float
    :param radius: radius of the circle
    :type radius: float
    :param n_points: number of sides of the polygon
    :type n_points: int
    :return: circle (polygon)
    :rtype: shapely Polygon
    """
    p = Point([long, lat])
    angles = np.linspace(0, 360, n_points)
    polygon = geog.propagate(p, angles, radius)
    return Polygon(polygon)


def get_lat_long_trips(path_to_trips, polygon=None):
    """
    Returns five lists. The first stores the number of cyclists on this trip,
    the second the start latitude, the third the start longitude,
    the fourth the end latitude, the fifth the end longitude.
    An index corresponds to the same trip in each list.
    :param path_to_trips: path to the compacted trips csv.
    :type path_to_trips: str
    :param polygon: If only trips inside a polygon should be considered,
     pass it here.
    :type polygon: Shapely Polygon
    :return: number of trips, start lat, start long, end lat, end long
    :rtype: list
    """
    trips = read_csv(path_to_trips, delim=',')

    if polygon is None:
        start_lat = list(trips['start latitude'])
        start_long = list(trips['start longitude'])
        end_lat = list(trips['end latitude'])
        end_long = list(trips['end longitude'])
        nbr_of_trips = list(trips['number of trips'])
        return nbr_of_trips, start_lat, start_long, end_lat, end_long
    else:
        trips['start in polygon'] = \
            trips[['start latitude', 'start longitude']].apply(
                lambda row: polygon.intersects(Point(row['start longitude'],
                                                     row['start latitude'])),
                axis=1)
        trips['end in polygon'] = \
            trips[['end latitude', 'end longitude']].apply(
                lambda row: polygon.intersects(Point(row['end longitude'],
                                                     row['end latitude'])),
                axis=1)
        trips['in polygon'] = trips[['start in polygon', 'end in polygon']].\
            apply(lambda row: row['start in polygon'] and row['end in polygon'],
                  axis=1)
        start_lat = list(trips.loc[trips['in polygon']]['start latitude'])
        start_long = list(trips.loc[trips['in polygon']]['start longitude'])
        end_lat = list(trips.loc[trips['in polygon']]['end latitude'])
        end_long = list(trips.loc[trips['in polygon']]['end longitude'])
        nbr_of_trips = list(trips.loc[trips['in polygon']]['number of trips'])
        return nbr_of_trips, start_lat, start_long, end_lat, end_long


def get_bbox_of_trips(path_to_trips, polygon=None):
    """
    Returns the bbox of the trips given by path_to_trips.
    :param path_to_trips: path to the compacted trips csv.
    :type path_to_trips: str
    :param polygon: If only trips inside a polygon should be considered,
     pass it here.
    :type polygon: Shapely Polygon
    :return: list of bbox [north, south, east, west]
    :rtype: list
    """
    trips_used, start_lat, start_long, end_lat, end_long = \
        get_lat_long_trips(path_to_trips, polygon)
    north = max(start_lat + end_lat) + 0.005
    south = min(start_lat + end_lat) - 0.005
    east = max(start_long + end_long) + 0.01
    west = min(start_long + end_long) - 0.01
    return [north, south, east, west]


def load_trips(G, path_to_trips, polygon=None):
    """
    Loads the trips and maps lat/long of start and end station to node in
    graph G.
    :param G: graph used vor lat/long to node mapping

    :param path_to_trips: path to the compacted trips csv.
    :type path_to_trips: str
    :param polygon: If only trips inside a polygon should be considered,
     pass it here.
    :type polygon: Shapely Polygon
    :return: dict with trip info and set of stations used.
    trip_nbrs structure: key=(origin node, end node), value=# of cyclists
    """
    nn_method = 'kdtree'

    nbr_of_trips, start_lat, start_long, end_lat, end_long = \
        get_lat_long_trips(path_to_trips, polygon)

    start_nodes = list(ox.get_nearest_nodes(G, start_long, start_lat,
                                            method=nn_method))
    end_nodes = list(ox.get_nearest_nodes(G, end_long, end_lat,
                                          method=nn_method))

    trip_nbrs = {}
    for trip in range(len(nbr_of_trips)):
        trip_nbrs[(int(start_nodes[trip]), int(end_nodes[trip]))] = \
            int(nbr_of_trips[trip])

    stations = set()
    for k, v in trip_nbrs.items():
        stations.add(k[0])
        stations.add(k[1])

    print('Number of trips: {}'.format(sum(trip_nbrs.values())))
    return trip_nbrs, stations


def plot_used_nodes(G, trip_nbrs, stations, place, save, width=20, height=20,
                    dpi=300):
    """
    Plots usage of nodes in graph G. trip_nbrs and stations should be
    structured as returned from load_trips().
    :param G: graph to plot in.
    :type G: networkx graph
    :param trip_nbrs: trips to plot the usage of.
    :type trip_nbrs: dict
    :param stations: set of stations.
    :type stations: set
    :param place: name of the city/place you are plotting.
    :type place: str
    :param save: save name for the plot.
    :type save: str
    :param width: width of the plot.
    :type width: int or float
    :param height: height opf the plot.
    :type height: int or float
    :param dpi: dpi of the plot.
    :type dpi: int
    :return: None
    """
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

    fig, ax = ox.plot_graph(G, fig_height=height, fig_width=width, dpi=dpi,
                            edge_linewidth=2, node_color=color_n,
                            node_size=ns, node_zorder=3, show=False,
                            close=False)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(cmap_name),
                               norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    cbaxes = fig.add_axes([0.1, 0.075, 0.8, 0.03])
    cbar = fig.colorbar(sm, orientation='horizontal', cax=cbaxes)
    cbar.ax.tick_params(axis='x', labelsize=18)
    cbar.ax.set_xlabel('normalised usage of stations', fontsize=24)
    fig.suptitle('Nodes used as Stations in {}'.format(place.capitalize()),
                 fontsize=30)
    plt.savefig('plots/{}.png'.format(save), format='png')

    # plt.close('all')
    plt.show()


def get_polygon_from_json(path_to_json):
    """
    Reads json at path. json should be downloaded via http://geojson.io/.
    :param path_to_json: file path to json.
    :type path_to_json: str
    :return: Polygon given by json
    :rtype: Shapely polygon
    """
    with open(path_to_json) as j_file:
        data = json.load(j_file)
    coordinates = data['features'][0]['geometry']['coordinates'][0]
    coordinates = [(item[0], item[1]) for item in coordinates]
    polygon = Polygon(coordinates)
    return polygon


def get_polygon_from_bbox(bbox):
    """
    Returns the Polygon resembled by the given bbox.
    :param bbox: bbox [north, south, east, west]
    :type bbox: list
    :return: Polygon of bbox
    :rtype: Shapely Polygon
    """
    north, south, east, west = bbox
    corners = [(east, north), (west, north), (west,south), (east, south)]
    polygon = Polygon(corners)
    return polygon


def main():
    print("Please start via a city specific script.")


if __name__ == '__main__':
    main()
