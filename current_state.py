from helper.current_state_helper import *


def main():
    place = 'oslo'
    trip_nbrs = np.load('data/{}_demand.npy'.format(place),
                        allow_pickle=True)[0]
    total_trips = sum(trip_nbrs.values())
    data = calc_current_state(place, trip_nbrs)


if __name__ == '__main__':
    main()
