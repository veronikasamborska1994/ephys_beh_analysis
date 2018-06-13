

import plotting as pl
import data_import as di


# Importing files from Two task Set up 

#two_tasks = di.Experiment('/Users/veronikasamborska/Desktop/2018-12-12-Reversal_learning/2018-03-30-two_tasks_reversal_learning')

final_pilot = di.Experiment ('/Users/veronikasamborska/Desktop/2018-12-12-Reversal_learning/data_pilot3')

# Plotting trials to reversal 
FT = True
pl.trials_till_reversal_plot(final_pilot)

