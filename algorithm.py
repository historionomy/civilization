import numpy as np
import sys
from scipy.ndimage import convolve

MODE = "DEBUG"

def get_array_means(array, grid_size, num_cols, num_rows):

    # this function get the mean value of the array input in every cell of pixels of dimension 
    # grid_size, and returns the associated array of means

    n, m, c = array.shape

    # let us remove pixels from borders beyond multiple of grid_size
    array = array[:(n // grid_size * grid_size), :(m // grid_size * grid_size), :]

    # let us compute averages by grid_size blocks
    reshaped_array = array.reshape(n, m // grid_size, grid_size, c)
    averaged_array = reshaped_array.mean(axis=2)
    n, m, c = averaged_array.shape
    reshaped_array = averaged_array.reshape(n // grid_size, grid_size, m, c)
    averaged_array = reshaped_array.mean(axis=1)

    ### option for simply sampling the array every grid_size
    # mid_grid_size = grid_size // 2
    # mean_array = array[mid_grid_size::grid_size, mid_grid_size::grid_size, :]
    # return mean_array

    return averaged_array

def calculate_orangeness(rgb):
    """Calculate the orangeness of an RGB color."""
    orange = np.array([1.0, 0.647, 0.0])  # RGB for orange, normalized to [0, 1]
    distance = np.linalg.norm(rgb - orange)
    max_distance = np.linalg.norm(np.array([1.0, 1.0, 1.0]) - orange)
    return 1 - (distance / max_distance)

def calculate_greenness(rgb):
    """Calculate the greenness of an RGB color."""
    medium_green = np.array([0.0, 0.5, 0.0])  # RGB for medium green, normalized to [0, 1]
    distance = np.linalg.norm(rgb - medium_green)
    max_distance = np.linalg.norm(np.array([1.0, 1.0, 1.0]) - medium_green)
    return 1 - (distance / max_distance)

class Grid:
    def __init__(self, num_rows, num_cols, map_img, grid_size, parameters):
        self.display = True
        self.current_grid_size = grid_size
        self.num_rows = num_rows
        self.num_cols = num_cols

        num_grid_rows = int(num_rows * grid_size)
        num_grid_cols = int(num_cols * grid_size)
        map_img = map_img[:num_grid_rows, :num_grid_cols, :]

        self.soil_color = get_array_means(map_img, grid_size, num_cols, num_rows) / 255

        # set technological stages at Paleolithic (tech stage zero)
        self.technology = np.zeros((num_rows, num_cols), dtype=int)
        self.culture = np.random.rand(num_rows, num_cols, 4)
        
        self.politics = np.zeros((num_rows, num_cols), dtype=int)
        self.prestige = np.zeros((num_rows, num_cols), dtype=float)

        # extract sea cells
        sea_blue_mask = self.soil_color[:, :, 2] <= parameters['geographics']['sea_blue_cutoff']
        sea_green_mask = self.soil_color[:, :, 1] >= parameters['geographics']['sea_green_cutoff']
        sea_red_mask = self.soil_color[:, :, 0] >= parameters['geographics']['sea_red_cutoff']
        # self.sea = np.multiply(sea_blue_mask.astype(float), sea_green_mask.astype(float))
        self.sea = sea_red_mask.astype(float)

        # compute soil components
        self.soil = {
            "mediterranean" : np.apply_along_axis(calculate_orangeness, 2, self.soil_color),
            "forest" : np.apply_along_axis(calculate_greenness, 2, self.soil_color)
        }

        # initialize population randomly
        self.population = np.random.rand(num_rows, num_cols)

        # set population on sea at zero
        self.population = np.multiply(self.population, self.sea)

        # set culture on sea at zero
        self.culture = np.multiply(self.culture, np.stack([self.sea] * 4, axis=-1))

        # compute population roof at Paleolithic age
        self.max_population = np.zeros((num_rows, num_cols), dtype=float)
        for soil in parameters['geographics']['soil_types']:
            population_per_soil = self.soil[soil] * parameters['geographics']['fertility_per_technology_level']['paleolithic'][soil + "_fertility"]
            self.max_population += population_per_soil

        # set max population on sea at zero
        self.max_population = np.multiply(self.max_population, self.sea)


    def timestep(self, parameters, time_step):

        # update fonction for each time step

        time_space = time_step * self.current_grid_size ** 2
        cell_size = float(self.current_grid_size ** 2)

        # update population roof according to technology
        self.max_population = np.zeros((self.num_rows, self.num_cols), dtype=float)
        for soil in parameters['geographics']['soil_types']:
            for tech_index in range(len(parameters['technology']['technological_stages'])):
                tech_name = parameters['technology']['technological_stages'][tech_index]
                tech_mask = self.technology == tech_index
                tech_mask = tech_mask.astype(float)
                self.max_population += tech_mask * self.soil[soil] * parameters['geographics']['fertility_per_technology_level'][tech_name][soil + "_fertility"]

        # population increase
        self.population = self.population * np.exp(parameters['demographics']['natural_growth'] * time_step)
        if MODE == "DEBUG":
            print(f'population : {self.population[5,5]}')
            print(f'population max : {self.max_population[5,5]}')
        self.population_excess = np.maximum(self.population - self.max_population, 0) * cell_size
        if MODE == "DEBUG":
            print(f'population excess : {self.population_excess[5,5]}')

        # migrations

        # Define the kernel for distributing population to 4 neighbors
        demo_kernel = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]]) / 4.0

        # distribute excess population to neighbors
        population_migration = convolve(self.population_excess, demo_kernel, mode='constant', cval=0.0)
        if MODE == "DEBUG":
            print(f'population migration : {population_migration[5,5]}')

        endogenous_demo_ratio = self.population / (self.population + population_migration / cell_size + 0.0001) # we add an epsilon to the denominator to avoid division by zero

        # Define the kernel for distributing culture to 4 neighbors
        culture_kernel = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])

        # define cultural changes due to migrations
        # 1st version : new culture is simply demographically ponderated average of migrant + autochtone
        for channel in range(3): # loop through color channels
            culture_migration = convolve(self.culture[:, :, channel], culture_kernel, mode='constant', cval=0.0)
            self.culture[:, :, channel] = population_migration * culture_migration * (1 - endogenous_demo_ratio) + self.population * self.culture[:, :, channel] * endogenous_demo_ratio
            self.culture[:, :, channel] = culture_migration
            # ensure boundaries of culture
            self.culture[:, :, channel] = np.minimum(1, self.culture[:, :, channel])
            self.culture[:, :, channel] = np.maximum(0, self.culture[:, :, channel])
    
        # population clipped after applying migrations
        self.population = np.minimum(self.population + population_migration, self.max_population)
    
        ### tech progress
        #############
        # TO DEBUG
        #############
        # tech_contagion_kernel = np.array([[0, 1, 0],
        #                 [1, 0, 1],
        #                 [0, 1, 0]])
        
        # tech_contagion_potential = convolve(self.technology, tech_contagion_kernel, mode='constant', cval=0.0)

        # # get the map of tech coefficients according to the political stage
        # tech_coefficients = np.array([p['tech_coefficient'] for p in parameters['politics']])
        # tech_coefficients = tech_coefficients[self.politics]

        # tech_progress_potential = parameters['technology']['tech_progress_population_coefficient'] * self.population
        # + parameters['technology']['tech_progress_political_coefficient'] * tech_coefficients
        # + parameters['technology']['tech_progress_contagion_coefficient'] * tech_contagion_potential

        # # let us check if arctan(tech_progress_potential + random) > pi/4
        # tech_progress_potential = np.arctan(tech_progress_potential + np.random.rand(*tech_progress_potential.shape))
        # tech_progress_potential = tech_progress_potential > np.pi/4
        # tech_progress_potential = tech_progress_potential.astype(int)

        # # add progress where it happened
        # self.technology += tech_progress_potential
        # # clip maximal progress
        # self.technology = np.minimum(self.technology, len(parameters['politics']) - 1)

        # ### political progress
        # politics_contagion_kernel = np.array([[0, 1, 0],
        #                 [1, 0, 1],
        #                 [0, 1, 0]])
        
        # politics_contagion_potential = convolve(self.politics, politics_contagion_kernel, mode='constant', cval=0.0)

        # politics_progress_potential = parameters['politics_progress']['politics_progress_population_coefficient'] * self.population
        # + parameters['politics_progress']['politics_progress_contagion_coefficient'] * politics_contagion_potential

        # # let us check if arctan(tech_progress_potential + random) > pi/4
        # politics_progress_potential = np.arctan(politics_progress_potential + np.random.rand(*politics_progress_potential.shape))
        # politics_progress_potential = politics_progress_potential > np.pi/4
        # politics_progress_potential = politics_progress_potential.astype(int)

        # # add political change where it happened
        # self.politics += politics_progress_potential
        # # clip maximal progress
        # self.politics = np.minimum(self.politics, len(parameters['politics']) - 1)

        #############
        # TO DEBUG - END
        #############
