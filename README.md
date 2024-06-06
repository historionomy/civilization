# Civilization

## Prerequisites

First, go to a working directory, and get the sources :

HTTPS mode :

```bash
git clone https://github.com/historionomy/civilization.git
```

SSH mode :

```bash
git clone git@github.com:historionomy/civilization.git
```

Then, move to your directory :

```bash
cd civilization
```

The project needs python >= 3.10 to run locally. You can either use your default python environment to execute it, but we recommand installing a dedicated conda python env :

```bash
conda create --name env_civ python=3.11
```

Then, activate your conda environment :

```bash
conda activate env_civ
```

Install the python packages requirements :

```bash
pip3 install -r requirements.txt
```

If you need to exit your conda environment :

```bash
conda deactivate
```

## Running the app locally

To run the app locally, run :

```bash
streamlit run civilization.py
```

## Code structure

- The function `load_map` in `civilization.py` load the file `europe.png`, as a RGB array
- The Streamlit state variable `st.session_state.grid` stores a `Grid` class object, the `Grid` class being defined in `algorithm.py`
- If the Counter is running, every second, the function `timestep` is applied on `st.session_state.grid`
- The model is executed in the `timestep` function of class `Grid` in file `algorithm.py`

## Algorithm

A detailed overview of the algorithm is available [here](./doc/algorithm_fr.md) in french.
