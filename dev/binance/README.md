signal generation is handled in the signal_generator.py file.
- implement some simple signals which could be used for factor investing
- signals can be combined and optimized

data processing is handled in the data_processor.py file.
- takes pd df of a single pair and calls the signal generator to get the feature df and target df

loading history is so far handled in a ipynb file. This needs to be moved to a data_loader.py file. 
So it can be integrated more efficiently into the main program.


