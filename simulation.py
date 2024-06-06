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
from matplotlib import pyplot as plt

# load parameters
parameter_file = "parameters.yml"

parameters = {}
with open(parameter_file, "r") as file:
    parameters = yaml.load(file, Loader=yaml.FullLoader)

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

        # Variables
        self.year = None

        # Functions
        self.fertility_per_technology_level = None

    def set_max_population(self):
        self.Pmax = self.ha_per_px*self.fertility_per_year[self.fertility_per_year['year']==self.year].iloc[0]["max_population_per_ha"]*self.fertility_map

    def reset_simulation(self):
        # Load image
        map_path = parameters['geographics']['map_path']  # Path to your map image
        self.map_img = cv2.imread(map_path)
        self.map_img = cv2.cvtColor(self.map_img, cv2.COLOR_BGR2RGB)
        self.map_height, self.map_width, _ = self.map_img.shape

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
        self.ha_per_px = parameters['geographics']['ha_per_px']
        self.start_year = parameters['time']['start_year']
        self.end_year = parameters['time']['end_year']
        self.time_step = parameters['time']['time_step']

        # Functions
        self.fertility_per_technology_level = pd.DataFrame(parameters['fertility_per_technology_level'])
        years = sorted(list(dict.fromkeys(self.fertility_per_technology_level['year'].tolist()+list(range(self.start_year, self.end_year + 1 ,1)))))
        self.fertility_per_year=pd.merge(pd.DataFrame({ 'year' : years,}),self.fertility_per_technology_level[['year','max_population_per_ha']],on="year",how="left").interpolate(method='linear')     

        #initialisation
        self.year = self.start_year
        self.population = np.zeros(self.map_img[:,:,0].shape)
        ## TODO parametrize population start
        self.population[800,800] = 1
        self.set_max_population()

    def view_field(self,field_name):
        X = getattr(self, field_name)

        match field_name:
            case "population":
                max_X = np.max(self.Pmax)
            case _ :
                max_X = np.max(X)

        grayscale_array = np.array(255*X/max_X).astype('uint8')
        overlay = np.zeros(self.map_img.shape)
        overlay[:,:,0]= self.map_img[:,:,0]* (1-grayscale_array/255.0) + grayscale_array/255.0* grayscale_array
        overlay[:,:,1]= self.map_img[:,:,1]* (1-grayscale_array/255.0) #+ grayscale_array/255.0* grayscale_array
        overlay[:,:,2]= self.map_img[:,:,2]* (1-grayscale_array/255.0) #+ grayscale_array/255.0* grayscale_array

        img = overlay.astype('uint8')
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,40)
        fontScale              = 1.5
        fontColor              = (255,255,255)
        thickness              = 5
        lineType               = 1

        time = self.year-self.start_year

        img = cv2.putText(img,"Year="+f"{time:04}"+ ", Max "+field_name+"="+"{:.0f}".format(np.max(X)), 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)

        cv2.imwrite("img/"+field_name+"_"+f"{time:04}"+".png",cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        return img
    
    def plot_field(self,field_name):
        plt.imshow(self.view_field(field_name))
        plt.show()

    
    def iterate(self):
        self.set_max_population()
        diffusion = scipy.ndimage.laplace(self.population)*self.population_diffusivity_map/100
        diffusion[diffusion>10*self.natural_growth] = 10*self.natural_growth
        diffusion[diffusion<-10*self.natural_growth] = -10*self.natural_growth
        dP = self.natural_growth*self.population*(1-self.population/self.Pmax) + diffusion
        self.population += dP*self.time_step
        self.population[self.population < 0] = 1e-4
        self.year += self.time_step