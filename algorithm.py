import numpy as np
from scipy.ndimage import convolve

# Function to get composite value of a grid cell
def get_composite_value(map_img, x, y, grid_size):
    cell = map_img[y:y+grid_size, x:x+grid_size]
    avg_color_per_row = np.average(cell, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return avg_color

def get_array_means(array, grid_size, num_cols, num_rows):

    # this function get the mean value of the array input in every cell of pixels of dimension grid_size, and returns the associated array of means

    # print(array.shape)
    reshaped_array = array.reshape(-1, grid_size, array.shape[2])
    means_first_dim_array = reshaped_array.mean(axis=1)

    # print(means_first_dim_array.shape)

    reshaped_array_second = means_first_dim_array.reshape(grid_size, -1, array.shape[2])
    # print(reshaped_array_second.shape)
    means_second_dim_array = reshaped_array_second.mean(axis=0)

    # print(means_second_dim_array.shape)

    mean_array = means_second_dim_array.reshape(num_rows, num_cols, array.shape[2])

    # print(mean_array.shape)

    return mean_array

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

# # Create a sample 2D array of 4-uplets (RGBA)
# array = np.random.rand(5, 5, 4)  # 5x5 array of RGBA values between 0 and 1

# # Extract the RGB components
# rgb_array = array[:, :, :3]

# # Calculate the orangeness for each cell
# orangeness_array = np.apply_along_axis(calculate_orangeness, 2, rgb_array)

# print("Original RGBA array:\n", array)
# print("Orangeness array:\n", orangeness_array)

class Grid:
    def __init__(self, num_rows, num_cols, map_img, grid_size, parameters):
        self.display = True
        self.current_grid_size = grid_size
        self.num_rows = num_rows
        self.num_cols = num_cols
        # print(map_img.shape)
        # print(num_rows, num_cols)
        # grid_rows = num_rows / grid_size
        # grid_cols = num_cols / grid_size
        # print(grid_rows, grid_cols)
        num_grid_rows = int(num_rows * grid_size)
        num_grid_cols = int(num_cols * grid_size)
        # print(num_grid_rows, num_grid_cols)
        map_img = map_img[:num_grid_rows, :num_grid_cols, :]
        # self.sea = np.zeros((num_rows, num_cols), dtype=int)
        self.soil_color = get_array_means(map_img, grid_size, num_cols, num_rows) / 255
        self.technology = np.zeros((num_rows, num_cols), dtype=int)

        # self.soil = np.zeros((num_rows, num_cols), dtype=int)
        self.culture = np.random.rand(num_rows, num_cols, 4)
        
        self.politics = np.zeros((num_rows, num_cols), dtype=int)
        self.prestige = np.zeros((num_rows, num_cols), dtype=float)

        # extract sea cells
        sea_mask = map_img[:, :, 2] >= parameters['geographics']['sea_blue_cutoff']
        self.sea = sea_mask.astype(int)

        # print(self.soil_color[5,5])
        # compute soil components
        self.soil = {
            "mediterranean" : np.apply_along_axis(calculate_orangeness, 2, self.soil_color),
            "forest" : np.apply_along_axis(calculate_greenness, 2, self.soil_color)
        }

        # print(self.soil)

        # print(self.soil['mediterranean'].shape)

        # initialize population randomly
        self.population = np.random.rand(num_rows, num_cols)

        # compute population roof at Paleolithic age
        self.max_population = np.zeros((num_rows, num_cols), dtype=float)
        for soil in parameters['geographics']['soil_types']:
            # print( self.soil[soil][5,5])
            population_per_soil = self.soil[soil] * parameters['geographics']['fertility_per_technology_level']['paleolithic'][soil + "_fertility"]
            # print(self.max_population.shape)
            # print(population_per_soil)
            self.max_population += population_per_soil

        # self.population = np.zeros((num_rows, num_cols), dtype=float)
        #

        

        # # we fill the map with random population according to soil types
        # initial_population = np.zeros((num_rows, num_cols), dtype=float)
        # for soil in parameters['geographics']['soil_types']:
        #     population_per_soil = self.soil[soil] * self.population * parameters['geographics']['fertility_per_technology_level']['paleolithic'][soil + "_fertility"]
        #     initial_population += population_per_soil

        # self.population = initial_population

        # # population at zero on sea
        # self.population[self.sea == 0] = 0

        # # set technological stages at Paleolithic
        # self.technology = np.zeros((num_rows, num_cols), dtype=int)

    def timestep(self, parameters):

        # update fonction for each time step

        # update population roof according to technology
        self.max_population = np.zeros((self.num_rows, self.num_cols), dtype=float)
        for soil in parameters['geographics']['soil_types']:
            for tech_index in range(len(parameters['technology']['technological_stages'])):
                tech_name = parameters['technology']['technological_stages'][tech_index]
                tech_mask = self.technology == tech_index
                # print(tech_mask[5,5])
                tech_mask = tech_mask.astype(float)
                # print(tech_mask[5,5])
                self.max_population += tech_mask * self.soil[soil] * parameters['geographics']['fertility_per_technology_level'][tech_name][soil + "_fertility"]

        # population increase
        # self.population = self.population * np.exp(parameters['demographics']['natural_growth'])
        print(self.population[5,5])
        # self.population = self.population * parameters['demographics']['natural_growth']
        self.population += 0.1
        print(self.max_population[5,5])
        self.population_excess = np.maximum(self.population - self.max_population, 0)
        print(self.population_excess[5,5])

        # migrations

        # Define the kernel for distributing population to 4 neighbors
        demo_kernel = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]]) / 4.0
        
        # distribute excess population to neighbors
        population_migration = convolve(self.population_excess, demo_kernel, mode='constant', cval=0.0)
        print(population_migration[5,5])

        endogenous_demo_ratio = self.population / (self.population + population_migration + 0.0001) # we add an epsilon to the denominator to avoid division by zero

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
        tech_contagion_kernel = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
        
        tech_contagion_potential = convolve(self.technology, tech_contagion_kernel, mode='constant', cval=0.0)

        # get the map of tech coefficients according to the political stage
        tech_coefficients = np.array([p['tech_coefficient'] for p in parameters['politics']])
        tech_coefficients = tech_coefficients[self.politics]

        tech_progress_potential = parameters['technology']['tech_progress_population_coefficient'] * self.population
        + parameters['technology']['tech_progress_political_coefficient'] * tech_coefficients
        + parameters['technology']['tech_progress_contagion_coefficient'] * tech_contagion_potential

        # let us check if arctan(tech_progress_potential + random) > pi/4
        tech_progress_potential = np.arctan(tech_progress_potential + np.random.rand(*tech_progress_potential.shape))
        tech_progress_potential = tech_progress_potential > np.pi/4
        tech_progress_potential = tech_progress_potential.astype(int)

        # add progress where it happened
        self.technology += tech_progress_potential
        # clip maximal progress
        self.technology = np.minimum(self.technology, len(parameters['politics']) - 1)

        ### political progress
        politics_contagion_kernel = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
        
        politics_contagion_potential = convolve(self.politics, politics_contagion_kernel, mode='constant', cval=0.0)

        politics_progress_potential = parameters['politics_progress']['politics_progress_population_coefficient'] * self.population
        + parameters['politics_progress']['politics_progress_contagion_coefficient'] * politics_contagion_potential

        # let us check if arctan(tech_progress_potential + random) > pi/4
        politics_progress_potential = np.arctan(politics_progress_potential + np.random.rand(*politics_progress_potential.shape))
        politics_progress_potential = politics_progress_potential > np.pi/4
        politics_progress_potential = politics_progress_potential.astype(int)

        # add political change where it happened
        self.politics += politics_progress_potential
        # clip maximal progress
        self.politics = np.minimum(self.politics, len(parameters['politics']) - 1)

# class Cells:
#     def __init__(self, population, soil, culture, technology, politics, prestige):
#         self.population = population
#         self.soil = soil
#         self.culture = culture
#         self.technology = technology
#         self.politics = politics
#         self.prestige = prestige

#     def update_from_neighbors(self, neighbors):
#         mean_value = np.mean([neighbor.population for neighbor in neighbors])
#         self.population += mean_value

# # Function to update cell values based on neighbors
# def update_cells(cells_table, parameters):
#     num_rows = len(cells_table)
#     num_cols = len(cells_table[0])
#     new_table = [[Cells(0.0, 0.0, 0.0, 0.0, 0.0) for _ in range(num_cols)] for _ in range(num_rows)]



#     for i in range(num_rows):
#         for j in range(num_cols):
#             neighbors = []
#             if i > 0:
#                 neighbors.append(cells_table[i-1][j])
#             if i < num_rows - 1:
#                 neighbors.append(cells_table[i+1][j])
#             if j > 0:
#                 neighbors.append(cells_table[i][j-1])
#             if j < num_cols - 1:
#                 neighbors.append(cells_table[i][j+1])

#             new_table[i][j].population = cells_table[i][j].population + np.mean([n.population for n in neighbors])
#             new_table[i][j].soil = cells_table[i][j].soil + np.mean([n.soil for n in neighbors])
#             new_table[i][j].culture = cells_table[i][j].culture + np.mean([n.culture for n in neighbors])
#             new_table[i][j].technology = cells_table[i][j].technology + np.mean([n.technology for n in neighbors])

#     return new_table