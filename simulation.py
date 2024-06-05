import os
import yaml
import pandas as pd
import numpy as np
import io
import cv2
import time
import cv2
import functools 
import colormap
import ast
import numpy as np
import scipy.interpolate
import yaml

# load parameters
parameter_file = "parameters.yml"

parameters = {}
with open(parameter_file, "r") as file:
    parameters = yaml.load(file, Loader=yaml.FullLoader)

#max_number_of_cultures = 255

class Simulation_data:
    def __init__(self):

        # Constant fields
        self.fertility_map = None
        self.population_diffusivity_map = None   

        # Time dependant fields 
        self.population = None
        self.technological_level = None
        # self.total_culture = np.zeros((num_rows, num_cols))
        # self.political_stage = np.zeros((num_rows, num_cols))
        # self.culture_vector = np.zeros((num_rows, num_cols,max_number_of_cultures))

        # Constants
        self.natural_growth = None

        # Functions
        self.fertility_per_technology_level = None

    def reset_simulation(self):
        # Load image
        map_path = parameters['geographics']['map_path']  # Path to your map image
        self.map_img = cv2.imread(map_path)
        self.map_img = cv2.cvtColor(self.map_img, cv2.COLOR_BGR2RGB)

        # Compute geographical dependant coefficient for fertility and population diffusivity
        geographics_params = [ {**geo_param, "RGB" : ast.literal_eval(geo_param['RGB']) } for geo_param in  parameters['geographics']['zones']]
        geographics_params = [ {**geo_param, "color" : geo_param['RGB'][0]+256*geo_param['RGB'][1]+(256^2)*geo_param['RGB'][2] } for geo_param in  geographics_params]

        color_values = [x['color'] for x in geographics_params]
        fertility_values = [x['Fertility'] for x in geographics_params]
        diffusivity_values = [x['Population diffusivity'] for x in geographics_params]

        fertility_polynomial_coefs = scipy.interpolate.lagrange(color_values, fertility_values).coef
        population_diffusivity_polynomial_coefs = scipy.interpolate.lagrange(color_values, diffusivity_values).coef

        X = self.map_img[:,:,0].astype(np.int32)+256*self.map_img[:,:,1].astype(np.int32)+(256^2)*self.map_img[:,:,2].astype(np.int32)
        fertility_map = np.zeros(X.shape)
        population_diffusivity_map = np.zeros(X.shape)

        for i in range(max(len(fertility_polynomial_coefs),len(fertility_polynomial_coefs))):
            fertility_map = np.multiply(fertility_map,X) + fertility_polynomial_coefs[i]
            population_diffusivity_map = np.multiply(population_diffusivity_map,X)  + population_diffusivity_polynomial_coefs[i]

        self.fertility_map = fertility_map
        self.population_diffusivity_map = population_diffusivity_map

        # Load constants
        self.natural_growth = parameters['demographics']['natural_growth']

        # Functions
        self.fertility_per_technology_level = pd.DataFrame(parameters['fertility_per_technology_level'])       