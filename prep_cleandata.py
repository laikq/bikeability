"""
This script downloads graphs needed and prepares the cleaned data for the
algorithm.
If you encounter errors while downloading a map by place, check on
https://nominatim.openstreetmap.org/ if your which_result is correct.
If you want to change the area (polygon) used take a look at the following
website: http://geojson.io/
"""
from helper.data_maps_helper import *
from helper.data_prep_helper import *


def prep_city(place, which_result, save, by = "polygon"):
    """
    Prepares all the data needed for the algorithm.
    :param place: nominatim search for your place
    :type place: str
    :param which_result: row of exact result
    :type which_result: int
    :param save: save name for graph and trips
    :type save: str
    :param by: mode of building the graph
    :type by: str out of ["bbox", "polygon", "name"]
    :return: none
    """
    
    #Speicherort für trips_csv
    trips_csv = 'data/csv/{}_trips_compact.csv'.format(save)
    #Speicherort für Polygon
    path_to_polygon = 'data/polygon/{}.json'.format(save)
    
    if by == "bbox":
        # Get bounding box of trips
        print('Getting bbox of trips.')
        bbox = get_bbox_of_trips(trips_csv)
        
        # Download map given by bbox
        print('Downloading map given by bbox.')
        G_b = download_map(bbox, by_bbox=True, by_name=False, by_polygon=False)
        
        # Loading trips inside bbox
        print('Mapping stations and calculation trips on map given by bbox')
        trips_b, stations_b = load_trips(G_b, trips_csv)
        
        # Colour all used nodes
        print('Plotting used nodes on graph given by bbox.')
        plot_used_nodes(G_b, trips_b, stations_b, save, '{}_bbox'.format(save))
        #Save data
        ox.save_graphml(G_b, filename='{}_bbox.graphml'.format(save),
                    folder='data/algorithm/input')
        np.save('data/algorithm/input/{}_bbox_demand.npy'.format(save), [trips_b])


    if by == "name":
        place = [place, which_result]
        
        # Download whole map of the city
        print('Downloading complete map of city')
        G_c = download_map(place, by_bbox=False, by_name=True, by_polygon=False)
        
        # Loading trips inside whole map
        print('Mapping stations and calculation trips on complete map.')
        trips_c, stations_c = load_trips(G_c, trips_csv)
        
        # Colour all used nodes
        print('Plotting used nodes on complete city.')
        plot_used_nodes(G_c, trips_c, stations_c, save.upper(),'{}_city'.format(save))
        #Save data
        ox.save_graphml(G_c, filename='{}_city.graphml'.format(save),
                         folder='data/algorithm/input')
        np.save('data/algorithm/input/{}_city_demand.npy'.format(save), [trips_c])

    
    if by == "polygon":
        # Download cropped map (polygon)
        polygon = get_polygon_from_json(path_to_polygon)
        
        print('Downloading polygon.')
        G = download_map(polygon, by_bbox=False, by_name=False, by_polygon=True)

        # Loading trips inside the polygon
        print('Mapping stations and calculation trips in polygon.')
        trips, stations = load_trips(G, trips_csv, polygon=polygon)

        # Colour all used nodes
        print('Plotting used nodes in polygon.')
        plot_used_nodes(G, trips, stations, save.upper(), save)
        #Save data
        ox.save_graphml(G, filename='{}.graphml'.format(save),
                    folder='data/algorithm/input')
        np.save('data/algorithm/input/{}_demand.npy'.format(save), [trips])        
    
    


def main():
    place = 'Hamburg, Deutschland'
    which_result = 2
    save = 'hh_part'

    prep_city(place, which_result, save, "polygon")


if __name__ == '__main__':
    main()

