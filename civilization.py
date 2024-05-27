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

# MODE = "debug"
MODE = "run"

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
    map_path = "europe_middle_east.jpg"  # Path to your map image
    map_img = cv2.imread(map_path)
    map_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    return map_img

# Function to add grid to the map
def add_grid(map_img, grid_size, cells_table):
    grid_img = map_img.copy()
    h, w, _ = map_img.shape
    overlay = grid_img.copy()

    # Draw the color overlay based on cell values
    for i in range(len(cells_table)):
        for j in range(len(cells_table[0])):
            cell = cells_table[i][j]
            color = (
                int(np.clip(cell.f1, 0, 1) * 255),
                int(np.clip(cell.f2, 0, 1) * 255),
                int(np.clip(cell.f3, 0, 1) * 255),
                int(np.clip(cell.f4, 0, 1) * 255),
            )
            cv2.rectangle(overlay, (j * grid_size, i * grid_size), ((j + 1) * grid_size, (i + 1) * grid_size), color, -1)

    for y in range(0, h, grid_size):
        cv2.line(overlay, (0, y), (w, y), (255, 0, 0), 1)
    for x in range(0, w, grid_size):
        cv2.line(overlay, (x, 0), (x, h), (255, 0, 0), 1)

    alpha = 0.3  # Transparency factor
    grid_img = cv2.addWeighted(overlay, alpha, grid_img, 1 - alpha, 0)

    return grid_img

# Function to get composite value of a grid cell
def get_composite_value(map_img, x, y, grid_size):
    cell = map_img[y:y+grid_size, x:x+grid_size]
    avg_color_per_row = np.average(cell, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return avg_color

# Function to update cell values based on neighbors
def update_cells(cells_table):
    num_rows = len(cells_table)
    num_cols = len(cells_table[0])
    new_table = [[Cells(0.0, 0.0, 0.0, 0.0, 0.0) for _ in range(num_cols)] for _ in range(num_rows)]

    for i in range(num_rows):
        for j in range(num_cols):
            neighbors = []
            if i > 0:
                neighbors.append(cells_table[i-1][j])
            if i < num_rows - 1:
                neighbors.append(cells_table[i+1][j])
            if j > 0:
                neighbors.append(cells_table[i][j-1])
            if j < num_cols - 1:
                neighbors.append(cells_table[i][j+1])

            new_table[i][j].f1 = cells_table[i][j].f1 + np.mean([n.f1 for n in neighbors])
            new_table[i][j].f2 = cells_table[i][j].f2 + np.mean([n.f2 for n in neighbors])
            new_table[i][j].f3 = cells_table[i][j].f3 + np.mean([n.f3 for n in neighbors])
            new_table[i][j].f4 = cells_table[i][j].f4 + np.mean([n.f4 for n in neighbors])

    return new_table

# load parameters
parameter_file = "parameters.yml"

parameters = {}
with open(parameter_file, "r") as file:
    parameters = yaml.load(file, Loader=yaml.FullLoader)

# Main function
def main():

    st.title("Europe Map with Grid")

    # Initialize session state variables
    if 'counter' not in st.session_state:
        st.session_state.counter = -4000
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'selected_cell' not in st.session_state:
        st.session_state.selected_cell = (0, 0)

    grid_size = st.sidebar.slider("Grid size", 5, 50, 10)
    map_img = load_map()

    # Create table of Cells with random initialization
    h, w, _ = map_img.shape
    num_rows = h // grid_size
    num_cols = w // grid_size
    if 'cells_table' not in st.session_state:
        st.session_state.cells_table = [[Cells(np.random.rand(), np.random.rand(), np.random.rand(), np.random.rand(), np.random.rand()) for _ in range(num_cols)] for _ in range(num_rows)]

    grid_img = add_grid(map_img, grid_size, st.session_state.cells_table)

    # Create two columns with custom widths
    col1, col2 = st.columns([4, 1])  # 4:1 ratio for the column widths

    with col1:
        # Use streamlit_image_coordinates to display the image and capture click coordinates
        coords = streamlit_image_coordinates(
            grid_img, 
            width=1000, 
            key="image_coords"
        )

    with col2:
        if coords:
            x, y = coords['x'], coords['y']
            cell_x, cell_y = x // grid_size, y // grid_size
            st.session_state.selected_cell = (cell_x, cell_y)

        cell_x, cell_y = st.session_state.selected_cell
        st.write(f"Selected Cell: ({cell_x}, {cell_y})")
        selected_cell = st.session_state.cells_table[cell_x][cell_y]

        if not st.session_state.running:
            f1 = st.text_input("f1", value=selected_cell.f1)
            f2 = st.text_input("f2", value=selected_cell.f2)
            f3 = st.text_input("f3", value=selected_cell.f3)
            f4 = st.text_input("f4", value=selected_cell.f4)
            f5 = st.text_input("f5", value=selected_cell.f5)

            if st.button("Save"):
                try:
                    selected_cell.f1 = min(1000, max(0, float(f1)))
                    selected_cell.f2 = min(1000, max(0, float(f2)))
                    selected_cell.f3 = min(1000, max(0, float(f3)))
                    selected_cell.f4 = min(1000, max(0, float(f4)))
                    selected_cell.f5 = min(1000, max(0, float(f5)))
                except ValueError:
                    pass  # Ignore invalid input

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

    st.write(f"Counter value: {st.session_state.counter}")

    if st.session_state.running:
        time.sleep(1)
        st.session_state.counter += time_step
        st.session_state.cells_table = update_cells(st.session_state.cells_table)
        st.rerun()

if __name__ == "__main__":
    main()

