# bikeability
This repo contains the pyhton skripts and most of the data necessary to run the bikeability improvement algorithm for Hamburg.

How To:
1. Clone the repo to your computer.
2. Install the python packages from requirements.txt, be careful networkit hast to be version 5.x NOT 6.x!
If you aren't keen on blowing up your memory consumption, ignore step 3.
3. Download "Buchungen Call a Bike (Stand 05/2017)" from https://data.deutschebahn.com/dataset/data-call-a-bike, extract it, rename it to db_trips_all.csv and place it in the data/csv folder. Run clean_rawdata.py. Be aware: If you have less than 16GB of memory it is likely that it will not work.
4. Run prep_cleandata.py to prepare the data needed for the algorithm.
5. Run run_algorithm.py to start the algorithm. If you don't want to use all your CPU cores for the edit the processes variable.
6. Run plot_results.py to look plot the results.

Common Problems:
- If you encounter errors while downloading a map by place, check on https://nominatim.openstreetmap.org/ if your which_result is correct.

Comments:
- If you want to change the area (polygon) used take a look at the following
website: http://geojson.io/
