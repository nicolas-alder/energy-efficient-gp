
from src.FlexGP import GaussianProcessRegressor, utils
from multiprocessing import Pool
import sys
import traceback
import random

class NullWriter(object):
    def write(self, arg):
        pass



def run_experiment(config):
    original_stdout = sys.stdout
    sys.stdout = NullWriter()
    print(config)
    print("Start Experiment: " + config["name"])
    try:
       GP = GaussianProcessRegressor(config)
       GP.fit()
       GP.predict(testset=False)
       GP.predict(testset=True)
       GP.eval()
    except:
       GP.failed(traceback)
       print("Error, skipping experiment ...")


    sys.stdout = original_stdout

if __name__ == '__main__':
    experiments = utils.load_config_files()
    print("Experiments: " + str(len(experiments)))

    # Erstellen Sie einen Pool von 8 Prozessen
    with Pool(2) as p:
        random.shuffle(experiments)
        p.map(run_experiment, experiments)


"""
import src.FlexGP.utils
from src.FlexGP import *


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   experiments = utils.load_config_files()
   print("Experiments: " + str(len(experiments)))

   for config in experiments:
      print(config)
      print("Start Experiment: " + config["name"])
      GP = GaussianProcessRegressor(config)
      GP.fit()
      GP.predict(testset=False)
      GP.predict(testset=True)
      GP.eval()
      try:
         GP = GaussianProcessRegressor(config)
         GP.fit()
         GP.predict(testset=False)
         GP.predict(testset=True)
         GP.eval()
      except:
         GP.failed()
         src.FlexGP.utils.reset_wrapped_functions()
         print("Error, skipping experiment ...")
         continue


"""


