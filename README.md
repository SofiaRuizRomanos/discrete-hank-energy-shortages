# Instruction for running this code

## 1. Requirements

Install all packages in the `requirements.txt` file using:

```bash
pip install -r requirements.txt
```

Install graphviz at https://graphviz.org/download/ (the mode of installation depends on the type of computer used). Graphviz produces the DAGs, but is not strictly necessary to solve the mode. 

## 2. Files
Run the file `solve_model.ipynb` to solve the stationary and dynamic models and produce the figures. This jupyter notbeeok automatically calls the files `model_functions.py` (which contains the different blocks of the economy) and `ss_analysis_functions.py` (which computes the performance of the model relative to the stationary target statistics). 

The folder `Pieroni_data` contains results saved from the original matlab files used to compute the figures. 

The folder `figures` contains all figures from the paper.

## 3. Running the brute force search

The command 

```bash
python approximate_target_statistics.py
```

creates the results in the `brute_force_results` folder with seperate folders for `solutions`, i.e. results that fall within the target range, `best_solutions` i.e. results that are as close as possible to the target range and `rejected_solutions`, results that are far outside the target range.Every time that the command above will be run again, it is recommended to erase the folders containing the results to make space for new results by using the following command:

```bash
./erase_run.sh      
```

The `brute_force_test_cache.pkl` pickle file is automatically made/updated every time the command is run, and stores the results of previous runs to avoid running them twice. If a change is made to the main code, it is best to delete this file.
