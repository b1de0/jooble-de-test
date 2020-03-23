# jooble-de-test
Jooble employment test solution for Data Engineer.

Using Dask is justified only in case of a large amount of data (number of files, file size).

# Structure 
app/tools/parsing.py - for read and prepare data \
app/tools/stat.py - transformers for data \
app/main.py - run all process

# Releases
v1.0 - Simple pandas, functions for processing

v1.1 - Simple pandas, processing from parent class with an identical interface

v1.2 - Dask dataframe with parallelization depending on the number of cores. 
You can uncoment 2 services in docker-compose.yml and see calculation plain and progress in real time.

v1.3 (planning) - Dask, Pipeline builder for processing.

# Run
bash run.sh
