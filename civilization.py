import streamlit as st
import os
import yaml
import snowflake.connector
import pandas as pd
import geopandas as gpd
import plotly.express as px
from PIL import Image
import io
import cv2
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates
import time
import copy

from parameters_display import create_parameters_tab
from algorithm import Grid
# from algorithm import update_cells

MODE = "DEBUG"
# MODE = "RUN"

black = (0, 0, 0)
white = (255, 255, 255)

st.set_page_config(layout="wide")

class Cells:
    def __init__(self, f1, f2, f3, f4, f5):
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.f4 = f4
        self.f5 = f5

    def update_from_neighbors(self, neighbors):
        mean_value = np.mean([neighbor.f1 for neighbor in neighbors])
        self.f1 += mean_value

# Load the map of Europe
def load_map():
    map_path = "europe.png"  # Path to your map image
    map_img = cv2.imread(map_path)
    map_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    return map_img

# Function to add grid to the map
def add_grid(map_img, grid_size, color_array):
    grid_img = map_img.copy()
    h, w, _ = map_img.shape
    overlay = grid_img.copy()

    print(color_array.shape)
    print(map_img.shape)

    # Draw the color overlay based on cell values
    map_type = 'scalar'
    if (len(color_array.shape) > 2):
        if (color_array.shape[2] == 4):
            map_type = '3D'
    if map_type == '3D':
        for i in range(len(color_array)):
            for j in range(len(color_array[0])):
                red, green, blue, alpha = color_array[i][j]
                color = (
                    int(np.clip(red, 0, 1) * 255),
                    int(np.clip(green, 0, 1) * 255),
                    int(np.clip(blue, 0, 1) * 255),
                    int(np.clip(alpha, 0, 1) * 255),
                )
                cv2.rectangle(overlay, (j * grid_size, i * grid_size), ((j + 1) * grid_size, (i + 1) * grid_size), color, -1)

        for y in range(0, h, grid_size):
            cv2.line(overlay, (0, y), (w, y), (255, 0, 0), 1)
        for x in range(0, w, grid_size):
            cv2.line(overlay, (x, 0), (x, h), (255, 0, 0), 1)

        alpha = 0.3  # Transparency factor
        grid_img = cv2.addWeighted(overlay, alpha, grid_img, 1 - alpha, 0)

    else:
        if (len(color_array.shape) > 2):
            for i in range(len(color_array)):
                for j in range(len(color_array[0])):
                    depth = int(np.clip(color_array[i][j][0], 0, 1) * 255)
                    color = (depth, 0, 0, depth)
                    cv2.rectangle(overlay, (j * grid_size, i * grid_size), ((j + 1) * grid_size, (i + 1) * grid_size), color, -1)

        else:
            for i in range(len(color_array)):
                for j in range(len(color_array[0])):
                    depth = int(np.clip(color_array[i][j], 0, 1) * 255)
                    color = (depth, 0, 0, depth)
                    cv2.rectangle(overlay, (j * grid_size, i * grid_size), ((j + 1) * grid_size, (i + 1) * grid_size), color, -1)

        for y in range(0, h, grid_size):
            cv2.line(overlay, (0, y), (w, y), black, 1)
        for x in range(0, w, grid_size):
            cv2.line(overlay, (x, 0), (x, h), black, 1)

        alpha = 0.3  # Transparency factor
        grid_img = cv2.addWeighted(overlay, alpha, grid_img, 1 - alpha, 0)

    return grid_img

# load parameters
parameter_file = "parameters.yml"

if 'parameters' not in st.session_state:
    st.session_state['parameters'] = {}
with open(parameter_file, "r") as file:
    st.session_state['parameters'] = yaml.load(file, Loader=yaml.FullLoader)

# load labels per language
content_per_language_file = "labels.yml"

content_translations = {}
with open(content_per_language_file, "r") as file:
    content_translations = yaml.load(file, Loader=yaml.FullLoader)

# Initialize the session state for selected_language
if 'selected_language' not in st.session_state:
    st.session_state['selected_language'] = list(content_translations.keys())[0]

# Initialize the session state for display_filter
if 'display_filter' not in st.session_state:
    st.session_state['display_filter'] = []

# Initialize the session state for grid size
if 'grid_size' not in st.session_state:
    st.session_state['grid_size'] = 1

# Initialize the session state for grid image
if 'grid_img' not in st.session_state:
    st.session_state['grid_img'] = []

default_grid_size = 10
default_counter = -4000

def reset_simulation():
    if st.session_state.reset_simulation_request == True:
        st.session_state.counter = default_counter
        # Create table of Cells with random initialization
        h, w, _ = st.session_state.map_img.shape
        num_rows = h // st.session_state.grid_size
        num_cols = w // st.session_state.grid_size
        st.session_state.cells_table = [[Cells(np.random.rand(), np.random.rand(), np.random.rand(), np.random.rand(), np.random.rand()) for _ in range(num_cols)] for _ in range(num_rows)]

        st.session_state.reset_simulation_request = False


# Main function
def main():

    top_menu = st.columns(3)
    with top_menu[0]:
        st.session_state['selected_language'] = st.selectbox("Language", options=list(content_translations['labels'].keys()))

    language_content = content_translations['labels'].get(st.session_state['selected_language'], "EN")

    st.title(language_content.get("main_title", "Default Title"))

    st.header(language_content.get("sub_title", "Default Header"))

    display_menu = st.columns(3)
    label2display = {
        v: k for k, v in language_content["display"].items()
    }
    with display_menu[0]:
        st.session_state['display_filter'] = st.selectbox(language_content["display_list"] ,options=list(language_content.get("display", {}).values()))
        st.session_state['display_filter'] = label2display[st.session_state['display_filter']]

    # Initialize session state variables
    if 'counter' not in st.session_state:
        st.session_state.counter = default_counter
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'selected_cell' not in st.session_state:
        st.session_state.selected_cell = (0, 0)
    if 'reset_simulation_request' not in st.session_state:
        st.session_state.reset_simulation_request = True        
    if 'grid_size' not in st.session_state:
        st.session_state.grid_size = default_grid_size
    # CODE AUGUSTIN
    if 'map_img' not in st.session_state:
        st.session_state.map_img = load_map()

    # slider for setting Grid size (= map resolution for simulation)
    grid_size = st.sidebar.slider("Grid size", 5, 50, default_grid_size)
    if st.session_state.grid_size != grid_size:
        st.session_state.grid_size = grid_size
        st.session_state.reset_simulation_request = True

    # load map
    map_img = load_map()

    reset_simulation()

    # CODE AUGUSTIN
    # grid_img = add_grid(st.session_state.map_img, st.session_state.grid_size, st.session_state.cells_table)

    # Create table of Cells with random initialization
    h, w, _ = map_img.shape
    num_rows = h // st.session_state['grid_size']
    num_cols = w // st.session_state['grid_size']

    # if grid not instantiated yet, we instantiate it
    if 'grid' not in st.session_state:
        st.session_state.grid = Grid(num_rows, num_cols, map_img, st.session_state['grid_size'], st.session_state['parameters'])

    # if grid is instantiated
    if st.session_state['grid'] is not None:

        # necessary to force redraw the map if grid resolution change
        if st.session_state['grid'].current_grid_size != st.session_state['grid_size']:
            st.session_state.grid = Grid(num_rows, num_cols, map_img, st.session_state['grid_size'], st.session_state['parameters'])

        # this block is for (re)drawing the simulation overlay on the map according to the chosen metric
        if st.session_state['display_filter'] == "population":
            grid_img = add_grid(map_img, st.session_state['grid_size'], st.session_state.grid.population)
        if st.session_state['display_filter'] == "sea":
            grid_img = add_grid(map_img, st.session_state['grid_size'], st.session_state.grid.sea)
        if st.session_state['display_filter'] == "max_population":
            grid_img = add_grid(map_img, st.session_state['grid_size'], st.session_state.grid.max_population)
        if st.session_state['display_filter'] == "culture":
            grid_img = add_grid(map_img, st.session_state['grid_size'], st.session_state.grid.culture)
        if st.session_state['display_filter'] == "soil":
            grid_img = add_grid(map_img, st.session_state['grid_size'], st.session_state.grid.soil_color)
        if st.session_state['display_filter'] == "technology":
            grid_img = add_grid(map_img, st.session_state['grid_size'], st.session_state.grid.technology)
        if st.session_state['display_filter'] == "politics":
            grid_img = add_grid(map_img, st.session_state['grid_size'], st.session_state.grid.politics)

        if 'grid_img' in st.session_state:
            st.session_state['grid_img'] = grid_img

    map_tab, parameters_tab = st.tabs([
        language_content["map_tab"],
        language_content["parameters_tab"]
    ])

    # Map tab
    with map_tab:

        # Counter section
        st.write("### Counter")

        counter_val = st.text_input("Counter", value=st.session_state.counter)
        try:
            counter_val = int(counter_val)
            counter_val = max(-5000, min(counter_val, 2100))
        except ValueError:
            counter_val = -4000
        st.session_state.counter = counter_val

        time_step = st.slider("Time step", min_value=10, max_value=100, value=10)

        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        if col1.button("Play"):
            st.session_state.running = True
        if col2.button("Pause"):
            st.session_state.running = False
        if col3.button("Reset"):
            st.session_state.counter = -4000
            st.session_state.running = False
            st.session_state.reset_simulation_request = True

        st.write(f"Counter value: {st.session_state.counter}")

        # Create two columns with custom widths
        col1, col2 = st.columns([4, 1])  # 4:1 ratio for the column widths

        with col1:
            # Use streamlit_image_coordinates to display the image and capture click coordinates
            coords = streamlit_image_coordinates(
                st.session_state['grid_img'], 
                width=st.session_state['grid_img'].shape[1],
                key="image_coords"
            )
            # CODE AUGUSTIN
            # coords = streamlit_image_coordinates(
            #     grid_img, 
            #     width=1000, 
            #     key="image_coords"
            # )

            # print(st.session_state['grid'].display)
            # if st.session_state['grid'].display is True:
            #     coords = streamlit_image_coordinates(
            #         st.session_state['grid_img'], 
            #         width=1000, 
            #         key="image_coords"
            #     )
            #     st.session_state['grid'].display = False

        with col2:
            if coords:
                
                x, y = coords['x'], coords['y']
                cell_x, cell_y = x // st.session_state['grid_size'], y // st.session_state['grid_size']
                st.session_state.selected_cell = (cell_x, cell_y)

            cell_x, cell_y = st.session_state.selected_cell
            st.write(f"Selected Cell:")
            st.write(f"  absciss: {cell_x} ordinates: {cell_y})")

            # warning : x and y are inverted in st.session_state.grid
            st.write(f"Soil color: {st.session_state.grid.soil_color[cell_y, cell_x]}")
            st.write(f"Population density: {st.session_state.grid.population[cell_y, cell_x]}")
            st.write(f"Maximum population density: {st.session_state.grid.max_population[cell_y, cell_x]}")
            st.write(f"Sea: {st.session_state.grid.sea[cell_y, cell_x]}")
            st.write(f"Culture: {st.session_state.grid.culture[cell_y, cell_x]}")

            if not st.session_state.running:
                # TODO customize according to the metric of the selected cell chosen
                population = st.text_input(language_content["parameters"]["demographics"]["name"], value=st.session_state.grid.population[cell_y, cell_x])
                # culture = st.text_input(language_content["parameters"]["culture"]["name"], value=st.session_state.grid.culture[cell_y, cell_x])
                politics = st.text_input(language_content["parameters"]["politics"]["name"], value=st.session_state.grid.politics[cell_y, cell_x])
                # soil_color = st.text_input(language_content["parameters"]["geographics"]["soil"], value=st.session_state.grid.soil_color[cell_y, cell_x])
                # technology = st.text_input(language_content["parameters"]["technology"]["name"], value=st.session_state.grid.technology[cell_y, cell_x])

                if st.button("Save"):
                    try:
                        st.session_state.grid.population[cell_y, cell_x] = population
                        # st.session_state.grid.culture[cell_y, cell_x] = culture
                        st.session_state.grid.politics[cell_y, cell_x] = politics
                        # st.session_state.grid.soil_color[cell_y, cell_x] = soil_color
                        # st.session_state.grid.technology[cell_y, cell_x] = technology
                    except ValueError:
                        pass  # Ignore invalid input

    # simulation launched : we execute timestep function and redraw the map every second
    if st.session_state.running:
        time.sleep(1)
        st.session_state.counter += time_step
        if 'grid' in st.session_state:
            temp_grid = st.session_state.grid
            temp_grid.timestep(st.session_state['parameters'], time_step)
            temp_grid.display = True
            st.session_state.grid = temp_grid

        st.rerun()

    # parameters edition tab
    with parameters_tab:
        tab_params = copy.deepcopy(st.session_state['parameters'])
        create_parameters_tab(language_content, tab_params)

if __name__ == "__main__":
    main()

