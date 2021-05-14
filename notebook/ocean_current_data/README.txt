OCEAN CURRENT DATA FROM KOLUMBO 2019
-produced by Greg Burgess (gburgess@mit.edu)

-Data from 4 separate dives, labeled dive A, dive E, dive F, and dive G respectively

-For each dive, there are two CSV files. A full data set as well as a time averaged data set.

Full Data Format

Row 1: Time in seconds since start of dive
Row 2: Depth of Vehicle in m (+ down)
Row 3: X position of vehicle in Local Mission Coordinates (LMC) in [m] origin in UTM for each dive provided below
Row 4: Y position of vehicle in LMC in [m}
Row 5: List of Ocean Current "column" data with following format. (True Depth[m], Depth bin [m], ocean current velocity in N direction [m/s], ocean current velocity in E direction [m/s])

Avg Data Format
Col 1: Depth [m]
Col 2: averaged velocity in N direction [m/s]
Col 3: averaged velocity in E direction [m/s]
Col 4: averaged velocity in D direction [m/s] *will be all zero* assumed to be negligible

Origin for each dive in UTMs (Northings and Eastings in meters)

*All in Zone 35 or Zone 'S'

dive A: 
N: 4043736.76
E: 364321.93


dive E:
N: 4041898.49
E: 363197.22


dive F:
N: 4042040.61
E: 363635.53


dive G: 
N: 4042103.23 
E: 363161.35

