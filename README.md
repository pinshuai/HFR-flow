# Jupyter notebook
## pre-process in Jupyter notebook

### 0. Model_config
* store model dimension, coordinates
* import geoFramework

### 1. Material_river_face
* generate material ids
* generate river sections

### 2. Initial_head
* import well data
* generate initial head

### 3. River_bc
* import Mass1 data
* calculate river datum & gradients from each river section
* generate .txt files

### 4. PFLOTRAN_input
* generate input deck from PFLOTRAN

## submit jobs to NERSC
### submit_job

## post-process data

### plot_in_Paraview
* download data from NERSC
* plot dataset using Paraview
* convert jpeg to gif

### plot_flux_from_massBalance
* generate flux plot snapshot
* plot net gaining bar plot

### plot_flux_from_river_cells-1
* pre-process data from plot

### plot_flux_from_river_cells-2
* generate bar plot for exchange flux
