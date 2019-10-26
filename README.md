# Online Prediction of Switching Graph Labelings with Cluster Specialists

This is the code accompanying the experimental section of the paper appearing in NeurIPS 2019. 
 
## Data Collection & Cleaning
Data was gathered using the following bash command 
> curl https://data.cityofchicago.org/api/id/eq45-8inv.json?\$select=\`id\`,\`timestamp\`,\`station_name\`,\`address\`,\`total_docks\`,\`docks_in_service\`,\`available_docks\`,\`available_bikes\`,\`percent_full\`,\`status\`,\`latitude\`,\`longitude\`,\`location\`,\`record\`\&\$order=\`timestamp\`\+DESC\&\$limit={xxxxxx\&\$offset={yyyyyy} >> {/path/to/your/file}.json

where :
* xxxxxx -> limit sets the number of entries (~2700000 for a month of data)

* yyyyyy -> offset sets the number of entries to skip from the current time (e.g. 0 will get the latest data)

The data was cleaned and the graph was built from the stations in "/data/Data Cleaning.ipynb". The graphs are stored as .npy
files. The labelings produced after cleaning are stored in "/data/mini_{train/test/total}_station_ten_minute_states.json".
These files are imported in tuning and the main experiments.


## Tuning Parameters
Tuning of the parameters was done in "/clusterspecialists/tuning/parallel_tuning.py"

## Main experiments
The algorithms are contained in "/clusterspecialists/algorithms/" and the main experiments are in "/clusterspecialists/main.py"

The main.py experiments require:
 
 * the output from the data cleaning (i.e., station to node maps as json files, and graphs as .npy files)
 * parameters selected from tuning (lines 249--251 in main.py)

