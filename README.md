# SHARPy for Pazy Wing
*All Pazy simulations are run with ρ=1.225 kg m<sup>-3</sup>.*

## Generating Flutter Speed, Linear System and Perform Time-Series Simulation
First, run `generate_deformed_flutter.py`, the script runs a single-step simulation for each speed defined. You can manuall set the minimum and maximum speeds, `M`, `N`, `Ms`, `alpha`, `skin_on`, and `trailing_edge_weight`. By default, the script stores the simulation outputs in `./output/pazy/output/`, and each setting will get its own directory, which contains the simulations for each velocity. One can extract the linear system from the directory `beam_modal_analysis`, which contains the M, C, and K matrices. After running `generate_deformed_flutter.py`, you can proceed to `pazy_flutter.ipynb` to calculate the flutter speed and perform time-series simulation.

## Time-Series Simulation
In `pazy_flutter.ipynb`, there are two sections, ***SHARPy TS without PDEControl*** and ***SHARPy TS with PDEControl***. The first section runs the time-series simulation for a given speed without control surface controller. This establishes the baseline for Pazy wing motion. The second section runs the time-series simulation for a given speed *with control surface controller*. You can define your control surface span (expressed as fraction of wingspan), control surface chord (expressed as fraction of wing chord), and all the controller parameters.

## Installation
Clone this project with the following:
```
git clone --recursive https://github.com/ianlamfar/sharpy.git <your_custom_folder (optional)>
```
The `--recursive` flag will also clone the xbeam and UVLM dependencies.
After cloning the repo, create the SHARPy environment with one of the `environment_<option>.yml` files provided. Replace `<option>` with one of the following: `linux`, `macos`, `minimal`, or `new`.
```
cd <your_sharpy_folder>/utils
conda env create -n <your_sharpy_env_name> -f environment_<option>.yml
cd ../
conda activate <your_sharpy_env_name>
```
Now build the UVLM and xbeam libraries.
```
mkdir build
cd build
cmake ..
make install -j 8
cd ../
pip install .
```