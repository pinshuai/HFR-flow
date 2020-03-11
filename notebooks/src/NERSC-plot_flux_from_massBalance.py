#!/usr/bin/env python
# coding: utf-8

# # Generate flux bar plot from Mass Balance file (.dat)

# In[2]:


import sys
sys.version


# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

# import seaborn as sns
# plt.style.use('default')

import h5py as h5
import glob
import matplotlib.gridspec as gridspec
import datetime
from datetime import timedelta
from scipy import interpolate, stats
import os
import re
from natsort import natsorted, ns, natsort_keygen
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry.polygon import Polygon
from scipy.stats.kde import gaussian_kde
import pickle
from textwrap import wrap


# In[2]:


def batch_time_to_delta(origin, x, time_format):
    y = []
    for ix in x:
        temp_y = abs(datetime.strptime(
            ix, time_format) - origin).total_seconds()
        y.append(temp_y)
    y = np.asarray(y)
    return(y)

def batch_delta_to_time(origin, x, time_format, delta_format):
    y = []
    for ix in x:
#         ix = np.asscalar(ix)
        if delta_format == "hours":
            temp_y = origin + timedelta(hours=ix)
        elif delta_format == "days":
            temp_y = origin + timedelta(days=ix)
        elif delta_format == "minutes":
            temp_y = origin + timedelta(minutes=ix)
        elif delta_format == "weeks":
            temp_y = origin + timedelta(weeks=ix)
        elif delta_format == "seconds":
            temp_y = origin + timedelta(seconds=ix)
        elif delta_format == "microseconds":
            temp_y = origin + timedelta(microseconds=ix)
        elif delta_format == "milliseconds":
            temp_y = origin + timedelta(milliseconds=ix)
        else:
            print("Sorry, this naive program only solve single time unit")
        y.append(temp_y.strftime(time_format))
    y = np.asarray(y)
#     y = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in y]
    return(y)


# In[3]:


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)  


# ## I/O files

# In[19]:


# inputs
data_dir = "/global/project/projectdirs/m1800/pin/Reach_scale_model/data/"
fname_usgs_gage = data_dir + "USGS_flow_gh_12472800.csv"
fname_mass1_coord = data_dir + "MASS1/coordinates.csv"
fname_river_middle = data_dir + "river_middle.csv"
fname_river_north = data_dir + "river_north.csv"
fname_river_south = data_dir + "river_south.csv"
fname_river_geo = data_dir + "river_geometry_manual.csv"

case_name =  "HFR_model_100x100x2_cyclic/"

model_dir = "/global/cscratch1/sd/pshuai/" + case_name
fname_model_origin = model_dir + "model_origin.txt"
fname_material_h5 = model_dir + "HFR_material_river.h5"
fname_pflotran_h5 = model_dir + "pflotran_100x100x2_cyclic_2011_2015.h5"
fname_mass_dat = model_dir + "pflotran_100x100x2_cyclic-mas.dat"

model_dir_2 = "/global/cscratch1/sd/pshuai/" + "HFR_model_100x100x2_new_iniH/"
fname_mass_dat_2 = model_dir_2 + "pflotran_100x100x2_6h_bc_new_iniH-mas.dat"

fname_river_bc = "/global/project/projectdirs/m1800/pin/Reach_scale_model/Inputs/river_bc/bc_6h_smooth_032807/"
fname_river_bc_1w = "/global/project/projectdirs/m1800/pin/Reach_scale_model/Inputs/river_bc/bc_1w_smooth_032807/"
# outputs
out_dir = "/global/project/projectdirs/m1800/pin/Reach_scale_model/Outputs/" + case_name
# out_dir_2 = "/global/project/projectdirs/m1800/pin/Reach_scale_model/Outputs/HFR_model_200x200x2_6h_bc/"

fig_dir = "/global/project/projectdirs/m1800/pin/Reach_scale_model/figures/"

fig_river_stage = "/global/project/projectdirs/m1800/pin/Reach_scale_model/Outputs/river_stage/"
# fig_river_stage_6h = out_dir_2 + "river_stage/stage_6h.png"
# fig_river_stage_1w = out_dir + "river_stage/stage_1w.png"
fig_finger_flux = out_dir + "mass_balance/"
fig_net_exchange_bar = out_dir + "mass_balance/net_exchange.png"
fname_net_exchange_txt = out_dir + "mass_balance/net_exchange.txt"
fig_flux_snapshot = out_dir + "mass_balance/"
fig_block_middle_flux_snapshot = out_dir + "flux_block_middle/"
fig_block_north_flux_snapshot = out_dir + "flux_block_north/"
fig_block_south_flux_snapshot = out_dir + "flux_block_south/"

result_dir = "/global/project/projectdirs/m1800/pin/Reach_scale_model/results/" + case_name
fname_flux_array= "/global/project/projectdirs/m1800/pin/Reach_scale_model/results/HFR_model_100x100x2_1w_bc/finger_flux_array.pk"

# fig_finger_flux_6h = out_dir_2 + "finger_flux/"


# In[5]:


date_origin = datetime.datetime.strptime("2007-03-28 12:00:00", "%Y-%m-%d %H:%M:%S")
# model_origin = np.genfromtxt(
#     fname_model_origin, delimiter=" ", skip_header=1)

model_origin = [551600, 104500]
material_file = h5.File(fname_material_h5, "r")


# create sub blocks along river reach

# In[12]:


# middle block
block_middle_coord = np.genfromtxt(fname_river_middle, delimiter=",", skip_header=1)
block_middle_x = [(np.min(block_middle_coord[:, 1]) - model_origin[0]) / 1000,
           (np.max(block_middle_coord[:, 1]) - model_origin[0]) / 1000]

block_middle_y = [(np.min(block_middle_coord[:, 2]) - model_origin[1]) / 1000,
           (np.max(block_middle_coord[:, 2]) - model_origin[1]) / 1000]
# north block
block_north_coord = np.genfromtxt(fname_river_north, delimiter=",", skip_header=1)
block_north_x = [(np.min(block_north_coord[:, 1]) - model_origin[0]) / 1000,
           (np.max(block_north_coord[:, 1]) - model_origin[0]) / 1000]

block_north_y = [(np.min(block_north_coord[:, 2]) - model_origin[1]) / 1000,
           (np.max(block_north_coord[:, 2]) - model_origin[1]) / 1000]
# south block
block_south_coord = np.genfromtxt(fname_river_south, delimiter=",", skip_header=1)
block_south_x = [(np.min(block_south_coord[:, 1]) - model_origin[0]) / 1000,
           (np.max(block_south_coord[:, 1]) - model_origin[0]) / 1000]

block_south_y = [(np.min(block_south_coord[:, 2]) - model_origin[1]) / 1000,
           (np.max(block_south_coord[:, 2]) - model_origin[1]) / 1000]


# ## import USGS gage

# In[6]:


USGS_gage = pd.read_csv(fname_usgs_gage, parse_dates= ['Date'])

# USGS_gage = USGS_gage[(USGS_gage['Date'] >= '2006-1-1') 
#                                 & (USGS_gage['Date'] <= '2018-1-1')]

USGS_gage.rename(columns={'Discharge (ft3/s)': 'Discharge'}, inplace=True)

USGS_gage['Discharge'] = USGS_gage['Discharge']*(0.3048**3) # convert cubic feet/sec to m3/s


# In[18]:


get_ipython().run_line_magic('matplotlib', 'inline')

fig, ax1 = plt.subplots(1, 1) 

USGS_gage.plot(x='Date', y= 'Discharge', ax=ax1, color = 'k', legend = None)

ax1.plot(('1/1/2011', '1/1/2016'), (3409, 3409), '--b')

ax1.set_xlim(['1/1/2011', '1/1/2016'])
ax1.set_ylim([0, 1e4])
ax1.set_ylabel('Discharge ($m^3/s$)')
ax1.set_xlabel('Year')
ax1.tick_params('y', colors='k')

ax1.legend( ['Dam', 'NEXSS'], loc = 'best', frameon = False)

fig.set_size_inches(8, 5)


# In[20]:


fname = out_dir + 'Dam_discharge.png'
fig.savefig(fname, dpi=300)
plt.close(fig)


# In[7]:


USGS_gage['Discharge'].describe()


# In[7]:


discharge_file = open(fname_usgs_gage, "r")
discharge_data = discharge_file.readlines()
# print(discharge_data)

discharge_data = [x.replace('"', "") for x in discharge_data]
discharge_data = [x.split(",") for x in discharge_data[1:]]
discharge_data = [list(filter(None, x)) for x in discharge_data]
discharge_data = np.asarray(discharge_data)
discharge_time = [datetime.strptime(x, "%Y-%m-%d")
                  for x in discharge_data[:, 3]]
discharge_value = discharge_data[:, 4]  # .astype(float)


# ## model dimensions

# In[9]:


# read model dimensions
all_h5 = glob.glob(fname_pflotran_h5)
all_h5 = np.sort(all_h5)

input_h5 = h5.File(all_h5[0], "r")
x_grids = list(input_h5["Coordinates"]['X [m]'])
y_grids = list(input_h5["Coordinates"]['Y [m]'])
z_grids = list(input_h5["Coordinates"]['Z [m]'])
input_h5.close()

# x_grids = np.arange(0, 60001, 100)
# y_grids = np.arange(0, 60001, 100)
# z_grids = np.arange(0, 201, 5)

dx = np.diff(x_grids)
dy = np.diff(y_grids)
dz = np.diff(z_grids)

nx = len(dx)
ny = len(dy)
nz = len(dz)
x = x_grids[0] + np.cumsum(dx) - 0.5 * dx[0]
y = y_grids[0] + np.cumsum(dy) - 0.5 * dy[0]
z = z_grids[0] + np.cumsum(dz) - 0.5 * dz[0]

grids = np.asarray([(x, y, z) for z in range(nz)
                    for y in range(ny) for x in range(nx)])

west_area = dy[0] * dz[0]
east_area = dy[0] * dz[0]
south_area = dx[0] * dz[0]
north_area = dx[0] * dz[0]
top_area = dx[0] * dy[0]
bottom_area = dx[0] * dy[0]


# **read river section information**

# In[10]:


mass1_sections = [s for s, s in enumerate(
    list(material_file["Regions"].keys())) if "Mass1" in s]
group_order = np.argsort(np.asarray(
    [x[6:] for x in mass1_sections]).astype(float))
mass1_sections = [mass1_sections[i] for i in group_order]
nsection = len(mass1_sections)
section_area = []
for isection in mass1_sections:
    faces = list(material_file["Regions"][isection]['Face Ids'])
    iarea = faces.count(1) * west_area + faces.count(2) * east_area +         faces.count(3) * south_area + faces.count(4) * north_area +         faces.count(5) * bottom_area + faces.count(6) * bottom_area
    section_area.append(iarea)
section_area = np.asarray(section_area)


# **read mass1 coordinates**

# In[11]:



section_coord = np.genfromtxt(
    fname_mass1_coord, delimiter=",", skip_header=1)

section_coord[:, 1] = section_coord[:, 1] - model_origin[0]
section_coord[:, 2] = section_coord[:, 2] - model_origin[1]
# subtract last mass1 location to get length for each segment
mass1_length = section_coord[:, 4] - section_coord[-1, 4]


# In[10]:


# add three lines to contour indicating mass1 location
line1 = section_coord[0, 1:3] / 1000
line2 = section_coord[int(len(section_coord[:, 1]) / 2), 1:3] / 1000
line3 = section_coord[-1, 1:3] / 1000
line1_x = [line1[0]] * 2
line1_y = [line1[1] - 5, line1[1] + 5]
line2_x = [line2[0] - 5, line2[0] + 5]
line2_y = [line2[1]] * 2
line3_x = [line3[0] - 5, line3[0] + 5]
line3_y = [line3[1]] * 2


# **read mass balance data**

# In[12]:


mass_file = open(fname_mass_dat, "r")
mass_data = mass_file.readlines()
mass_header = mass_data[0].replace('"', '').split(",")
mass_header = list(filter(None, mass_header))
mass_data = [x.split(" ") for x in mass_data[1:]]
mass_data = [list(filter(None, x)) for x in mass_data]
mass_data = np.asarray(mass_data).astype(float)


# **find columns of desired mass1 colmns**

# In[13]:



# e.g River_40 Water Mass [kg]
mass_index = [i for i, s in enumerate(mass_header) if (
    "River" in s and "Water" in s and "kg]" in s)]
# e.g River_40 Water Mass [kg/h]
flux_index = [i for i, s in enumerate(mass_header) if (
    "River" in s and "Water" in s and "kg/h" in s)]


# In[14]:


# get total river mass/flux across the river bed for each time step (6h)
total_mass = np.sum(mass_data[:, mass_index], axis=1).flatten()
total_flux = np.sum(mass_data[:, flux_index], axis=1).flatten()


# In[15]:


# shift mass when restart happened, new starting mass would shift to zero
diff_total_mass = abs(np.diff(total_mass))
restart_index = np.arange(len(diff_total_mass))[diff_total_mass > 2e10]  # select 2e10 as a threshold

# add mass at restart check point to all after time steps
for i_restart_index in restart_index:
    total_mass[(i_restart_index + 1):len(total_mass)] = total_mass[(
        i_restart_index + 1):len(total_mass)] + total_mass[restart_index]


# In[16]:


restart_index


# # plot river stage
# 

# Mass1 points (40, 186, 332) are used as upstream, middle and downstream river stage.

# In[17]:


start_time = datetime.datetime.strptime("2007-03-28 12:00:00", "%Y-%m-%d %H:%M:%S")

time_ticks = [
    "2011-01-01 00:00:00",
    "2012-01-01 00:00:00",
    "2013-01-01 00:00:00",
    "2014-01-01 00:00:00",
    "2015-01-01 00:00:00",
    "2016-01-01 00:00:00"]
time_ticks = [datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in time_ticks]

selected_ticks = [
    "2012-02-25 00:00:00",
    "2012-04-30 00:00:00",
    "2012-07-04 00:00:00",
    "2012-08-18 00:00:00",
    "2012-10-17 00:00:00"
    ]
selected_ticks = [datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in selected_ticks]


# ## import river stage from generated mass1 datum files

# In[18]:


datum_files = glob.glob(fname_river_bc + "Datum*.txt")

datum_files = natural_sort(datum_files)


# In[19]:


# int(re.findall(r'\d+', datum_files[0])[-1])
# datum_files = natsorted(datum_files) # sort files naturally, i.e. 1, 2, 11, 22, ...


# In[20]:


river_north_datum = pd.read_table(datum_files[0], sep=' ', header=None, names=['time', 'x', 'y', 'wl'])
river_middle_datum = pd.read_table(datum_files[(len(datum_files)-1)//2], sep=' ', header=None, names=['time', 'x', 'y', 'wl'])
river_south_datum = pd.read_table(datum_files[-1], sep=' ', header=None, names=['time', 'x', 'y', 'wl'])

river_north_datum['time'] = batch_delta_to_time(date_origin, river_north_datum['time'], "%Y-%m-%d %H:%M:%S", "seconds")
river_middle_datum['time'] = batch_delta_to_time(date_origin, river_middle_datum['time'], "%Y-%m-%d %H:%M:%S", "seconds")
river_south_datum['time'] = batch_delta_to_time(date_origin, river_south_datum['time'], "%Y-%m-%d %H:%M:%S", "seconds")
# datum_time = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in datum_time]


# In[21]:


river_north_datum['time'] = river_north_datum['time'].apply(pd.to_datetime)
river_middle_datum['time'] = river_middle_datum['time'].apply(pd.to_datetime)
river_south_datum['time'] = river_south_datum['time'].apply(pd.to_datetime)


# ## plot 6h-smooth river stage

# ** river stage from 2011-2016 **

# In[29]:


fig, ax1 = plt.subplots(1, 1)

# river_north_datum.plot(x='time', y='wl',color= 'black', linewidth = 0.5, ax=ax1, legend=None)
river_middle_datum.plot(x='time', y='wl',color= 'black',linewidth = 0.5, ax=ax1, legend=None)
# river_south_datum.plot(x='time', y='wl',color= 'black', linewidth = 0.5, ax=ax1, legend=None)

ax1.set_ylabel('River stage (m)')
ax1.set_xlabel('')
# ax1.set_title('River stage from upstream to downstream')
# ax1.set_aspect("equal", "datalim")
ax1.xaxis.set_major_locator(mdates.YearLocator())
# ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
ax1.xaxis.set_tick_params('labelbottom')

ax1.set_xlim([time_ticks[0], time_ticks[-1]])
plt.setp( ax1.xaxis.get_majorticklabels(), rotation=0, ha = 'center' )

# ax1.legend(handles=legend_elements, loc='best')
fig.tight_layout()
fig.set_size_inches(8, 3.5)


# In[30]:


fig.savefig(fig_river_stage + 'river_stage_m_2011_2016.jpg', dpi=300)
plt.close(fig)


# **plot river stage for individual year**

# In[60]:


fig, ax1 = plt.subplots(1, 1)

# river_north_datum.plot(x='time', y='wl',color= 'black', linewidth = 0.5, ax=ax1, legend=None)
river_middle_datum.plot(x='time', y='wl',color= 'black',linewidth = 0.5, ax=ax1, legend=None)
# river_south_datum.plot(x='time', y='wl',color= 'black', linewidth = 0.5, ax=ax1, legend=None)

## add line for selected time
for itime in selected_ticks:
    x = [itime, itime]
    y = [0, 200]
    ax1.plot(x, y, 'r-')

ax1.set_ylabel('River stage (m)')
ax1.set_xlabel('Year 2011')
# ax1.set_title('River stage from upstream to downstream')
# ax1.set_aspect("equal", "datalim")
# ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.xaxis.set_major_locator(mdates.MonthLocator())
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
ax1.xaxis.set_tick_params('labelbottom')

ax1.set_ylim([109, 117])
ax1.set_xlim([time_ticks[0], time_ticks[1]])
plt.setp( ax1.xaxis.get_majorticklabels(), rotation=0, ha = 'center' )

# ax1.legend(handles=legend_elements, loc='best')
fig.tight_layout()
fig.set_size_inches(8, 3.5)


# In[62]:


fig.savefig(fig_river_stage+ 'river_stage_2011.jpg', dpi=300)
plt.close(fig)


# ## plot 5d-smoothed river stage

# In[20]:


datum_files = glob.glob(fname_river_bc_1w + "Datum*.txt")

# int(re.findall(r'\d+', datum_files[0])[-1])
datum_files = natsorted(datum_files) # sort files naturally, i.e. 1, 2, 11, 22, ...


# In[21]:


river_north_datum_s = pd.read_table(datum_files[0], sep=' ', header=None, names=['time', 'x', 'y', 'wl'])
river_middle_datum_s = pd.read_table(datum_files[(len(datum_files)-1)//2], sep=' ', header=None, names=['time', 'x', 'y', 'wl'])
river_south_datum_s = pd.read_table(datum_files[-1], sep=' ', header=None, names=['time', 'x', 'y', 'wl'])

river_north_datum_s.time = batch_delta_to_time(date_origin, river_north_datum_s.time, "%Y-%m-%d %H:%M:%S", "seconds")
river_middle_datum_s.time = batch_delta_to_time(date_origin, river_middle_datum_s.time, "%Y-%m-%d %H:%M:%S", "seconds")
river_south_datum_s.time = batch_delta_to_time(date_origin, river_south_datum_s.time, "%Y-%m-%d %H:%M:%S", "seconds")
# datum_time = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in datum_time]


# In[22]:


river_north_datum_s['time'] = river_north_datum_s['time'].apply(pd.to_datetime)
river_middle_datum_s['time'] = river_middle_datum_s['time'].apply(pd.to_datetime)
river_south_datum_s['time'] = river_south_datum_s['time'].apply(pd.to_datetime)


# In[34]:


fig, ax1 = plt.subplots(1, 1)

# river_north_datum_s.plot(x='time', y='wl',color= 'black', linewidth = 0.5, ax=ax1, legend=None)
river_middle_datum_s.plot(x='time', y='wl',color= 'black',linewidth = 0.5, ax=ax1, legend=None)
# river_south_datum_s.plot(x='time', y='wl',color= 'black', linewidth = 0.5, ax=ax1, legend=None)

ax1.set_ylabel('River stage (m)')
ax1.set_xlabel('')
# ax1.set_title('River stage from upstream to downstream')
# ax1.set_aspect("equal", "datalim")
ax1.xaxis.set_major_locator(mdates.YearLocator())
# ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
ax1.xaxis.set_tick_params('labelbottom')

ax1.set_xlim([time_ticks[0], time_ticks[-1]])
plt.setp( ax1.xaxis.get_majorticklabels(), rotation=0, ha = 'center' )

# ax1.legend(handles=legend_elements, loc='best')
fig.tight_layout()
fig.set_size_inches(8, 3.5)


# In[35]:


fig.savefig(fig_river_stage+'river_stage_m_2011_2016_s.jpg', dpi=300)
plt.close(fig)


# # plot flux heat map

# ## plot finger flux - first case

# In[22]:


# plot fingerprint plots
simu_time = mass_data[:, 0]
real_time = batch_delta_to_time(
    date_origin, simu_time, "%Y-%m-%d %H:%M:%S", "hours")
real_time = [datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in real_time]
plot_time = [
    "2011-01-01 00:00:00",
    "2016-01-01 00:00:00"]
plot_time = [datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in plot_time]


# In[23]:


time_ticks = [
    "2011-01-01 00:00:00",
    "2012-01-01 00:00:00",
    "2013-01-01 00:00:00",
    "2014-01-01 00:00:00",
    "2015-01-01 00:00:00",
    "2016-01-01 00:00:00"]
time_ticks = [datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in time_ticks]


# In[24]:


flux_array = mass_data[:, flux_index]
flux_array = np.asarray(flux_array) / 1000 * 24
for itime in range(len(real_time)):
    flux_array[itime, :] = flux_array[itime, :] / section_area


# In[26]:


get_ipython().run_line_magic('matplotlib', 'inline')
# plot finger flux
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
fig = plt.figure()

gs = gridspec.GridSpec(2, 1, height_ratios=[2,3])
gs.update(hspace = 0.1) # adjust vertical spacing b/w subplots
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1], sharex=ax1)

# river_north_datum_s.plot(x='time', y='wl',color= 'black', linewidth = 0.5, ax=ax1, legend=None)
river_middle_datum.plot(x='time', y='wl',color= 'black',linewidth = 0.5, ax=ax1, legend=None)
# river_south_datum_s.plot(x='time', y='wl',color= 'black', linewidth = 0.5, ax=ax1, legend=None)
# ax1.set_yticks(np.arange(0, 76, 10))
ax1.set_ylabel('River stage (m)')
ax1.set_xlabel('')
# ax1.set_title('River stage from upstream to downstream')

# plt.xticks(time_ticks, rotation=0, horizontalalignment = "center")
# ax1.set_xticks(time_ticks)

# ax1.set_xlim([time_ticks[0], time_ticks[-1]])


cf1 = ax2.contourf(real_time,
                   0.5 * (mass1_length[1:] + mass1_length[:-1]),
                   -np.transpose(flux_array),
                   cmap=plt.cm.jet,
                   levels=np.arange(-0.1, 0.105, 0.005),
                   extend="both",
                   )
ylabel = '\n'.join(wrap("Distance to downstream end (km)", 20))
ax2.set_ylabel(ylabel)
# ax2.set_title("Exchange flux of weekly boundary")
ax2.set_xticks(time_ticks)
ax2.set_yticks(np.arange(0, 76, 10))
ax2.set_ylim([0, 73])
ax2.set_xlim([time_ticks[0], time_ticks[-1]])

divider = make_axes_locatable(ax2)
cax = divider.new_vertical(size="5%", pad=0.4, pack_start=True)
fig.add_axes(cax)
cb1 = fig.colorbar(cf1, cax=cax, orientation="horizontal")

# cb1 = plt.colorbar(cf1, extend="both", orientation = "horizontal", pad = 0.2)
cb1.ax.set_xlabel("Exchange flux (m/d)", labelpad=0.3)

ax1.minorticks_off()
# fig.suptitle("Finger flux plot for weekly boundary", fontsize = 12)
# fig.tight_layout()
fig.set_size_inches(8, 6)


# In[27]:


fig.savefig(fig_finger_flux + 'finger_flux.jpg', dpi=300, transparent=False)
plt.close(fig)


# ## plot finger flux- 2nd case

# In[24]:


# read mass balance data
mass_file_2 = open(fname_mass_dat_2, "r")
mass_data_2 = mass_file_2.readlines()
mass_header = mass_data_2[0].replace('"', '').split(",")
mass_header = list(filter(None, mass_header))
mass_data_2 = [x.split(" ") for x in mass_data_2[1:]]
mass_data_2 = [list(filter(None, x)) for x in mass_data_2]
mass_data_2 = np.asarray(mass_data_2).astype(float)


# In[25]:


# find columns of desired mass1 colmns
# e.g River_40 Water Mass [kg]
mass_index = [i for i, s in enumerate(mass_header) if (
    "River" in s and "Water" in s and "kg]" in s)]
# e.g River_40 Water Mass [kg/h]
flux_index = [i for i, s in enumerate(mass_header) if (
    "River" in s and "Water" in s and "kg/h" in s)]


# In[26]:


# plot fingerprint plots
simu_time = mass_data_2[:, 0]
real_time_2 = batch_delta_to_time(
    date_origin, simu_time, "%Y-%m-%d %H:%M:%S", "hours")
real_time_2 = [datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in real_time_2]
plot_time = [
    "2011-01-01 00:00:00",
    "2016-01-01 00:00:00"]
plot_time = [datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in plot_time]
time_ticks = [
    "2011-01-01 00:00:00",
    "2012-01-01 00:00:00",
    "2013-01-01 00:00:00",
    "2014-01-01 00:00:00",
    "2015-01-01 00:00:00",
    "2016-01-01 00:00:00"]
time_ticks = [datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in time_ticks]


# In[39]:


flux_array_2 = mass_data_2[:, flux_index]
flux_array_2 = np.asarray(flux_array_2) / 1000 * 24 # convert to m/d
for itime in range(len(real_time_2)):
    flux_array_2[itime, :] = flux_array_2[itime, :] / section_area


# **plot finger flux with river stage**

# In[57]:


get_ipython().run_line_magic('matplotlib', 'inline')
# plot finger flux
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
fig = plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios=[2,3])
gs.update(hspace = 0.1) # adjust vertical spacing b/w subplots
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1], sharex=ax1)

# river_north_datum.plot(x='time', y='wl',color= 'black', linewidth = 0.5, ax=ax1, legend=None)
river_middle_datum.plot(x='time', y='wl',color= 'black',linewidth = 0.5, ax=ax1, legend=None)
# river_south_datum.plot(x='time', y='wl',color= 'black', linewidth = 0.5, ax=ax1, legend=None)

ax1.set_ylabel('River stage (m)')
ax1.set_xlabel('')
# ax1.set_title('River stage from upstream to downstream')

# plt.xticks(time_ticks, rotation=0, horizontalalignment = "center")

# ax1.set_xticks(time_ticks)
# ax1.set_xlim([time_ticks[0], time_ticks[-1]])
# ax1.xaxis.set_major_locator(mdates.YearLocator())
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# ax1.xaxis.set_tick_params('labelbottom')


cf1 = ax2.contourf(real_time_2,
                   0.5 * (mass1_length[1:] + mass1_length[:-1]),
                   -np.transpose(flux_array_2),
                   cmap=plt.cm.jet,
                   levels=np.arange(-0.1, 0.105, 0.005),
                   extend="both",
                   )
# ax2.set_ylabel("")
ylabel = '\n'.join(wrap("Distance to downstream end (km)", 20))
ax2.set_ylabel(ylabel)
# ax2.set_title("Exchange flux of weekly boundary")
ax2.set_xticks(time_ticks)
ax2.set_yticks(np.arange(0, 76, 10))
ax2.set_ylim([0, 73])
ax2.set_xlim([time_ticks[0], time_ticks[-1]])

ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.xaxis.set_tick_params('labelbottom')

divider = make_axes_locatable(ax2)
cax = divider.new_vertical(size="5%", pad=0.4, pack_start=True)
fig.add_axes(cax)
cb1 = fig.colorbar(cf1, cax=cax, orientation="horizontal")

# cb1 = plt.colorbar(cf1, extend="both", orientation = "horizontal", pad = 0.2)
cb1.ax.set_xlabel("Exchange flux (m/d)", labelpad=0.3)

# fig.suptitle("Finger flux plot for hourly boundary", fontsize = 12)
ax1.minorticks_off()
# fig.tight_layout()
fig.set_size_inches(8, 6)


# In[58]:


fig.savefig(fig_finger_flux + 'finger_flux_6h.jpg', dpi=300)
plt.close(fig)


# ## plot net flux difference between 6h and 1w bc

# In[59]:


# flux_array_1w = flux_array[::20] # 1-w bc simulation has 6h output freq. and 6-h bc has 120 h output freq.
flux_array_6h = flux_array_2
flux_array_1w = flux_array


# In[8]:


# with open(fname_flux_array, "wb") as f:
#     pickle.dump((flux_array_6h, flux_array_1w), f)

with open(fname_flux_array, "rb") as f:
    flux_array_6h, flux_array_1w = pickle.load(f)


# In[8]:


np.savetxt(result_dir + "flux_array_6h.csv", flux_array_6h, delimiter=",")


# In[9]:


np.savetxt(result_dir + "flux_array_1w.csv", flux_array_1w, delimiter=",")


# ### use abs flux

# In[31]:


flux_array_diff = (np.abs(flux_array_6h) - np.abs(flux_array_1w))/np.abs(flux_array_1w)*100


# In[32]:


get_ipython().run_line_magic('matplotlib', 'inline')

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

river_north_datum.plot(x='time', y='wl',color= 'black', linewidth = 0.5, ax=ax1, legend=None)
river_middle_datum.plot(x='time', y='wl',color= 'black',linewidth = 0.5, ax=ax1, legend=None)
river_south_datum.plot(x='time', y='wl',color= 'black', linewidth = 0.5, ax=ax1, legend=None)

ax1.set_ylabel('Water Level (m)')
ax1.set_xlabel('')

# gs = gridspec.GridSpec(1, 1)
# fig = plt.figure()
# ax1 = fig.add_subplot(gs[0, 0])
cf1 = ax2.contourf(real_time_2,
                   0.5 * (mass1_length[1:] + mass1_length[:-1]),
                   np.transpose(flux_array_diff),
                   cmap=plt.cm.jet,
                   levels=np.arange(-100, 100, 10),
                   extend="both",
                   vmin = -100, vmax = 100,
                   )

# ax2.imshow(np.transpose(flux_array_diff_num), cmap=cm.gist_rainbow)
# ax2.set_ylabel("")
ax2.set_ylabel("Distance (km)", wrap = True)
ax2.set_xticks(time_ticks)
ax2.set_ylim([0, 73])
ax2.set_xlim([time_ticks[0], time_ticks[-1]])


divider = make_axes_locatable(ax2)
cax = divider.new_vertical(size="5%", pad=0.4, pack_start=True)
fig.add_axes(cax)
cb1 = fig.colorbar(cf1, cax=cax, orientation="horizontal")

# cb1 = plt.colorbar(cf1, extend="both", orientation = "horizontal", pad = 0.2)
cb1.ax.set_xlabel("Abs exchange flux diff (%)", labelpad=0.3)

# fig.suptitle("Finger flux plot for hourly boundary", fontsize = 12)
ax1.minorticks_off()
fig.tight_layout()
fig.set_size_inches(8, 6)



# gs = gridspec.GridSpec(1, 1)
# fig = plt.figure()
# ax1 = fig.add_subplot(gs[0, 0])
# cf1 = ax1.contourf(real_time_2,
#                    0.5 * (mass1_length[1:] + mass1_length[:-1]),
#                    np.transpose(flux_array_diff),
#                    cmap=plt.cm.jet,
#                    levels=np.arange(-100, 100, 10),
#                    extend="both",
#                    vmin = -100, vmax = 100,
#                    )
# ax1.set_ylabel("")
# ax1.set_ylabel("Distance to Downstream Location (km)")
# ax1.set_xticks(time_ticks)
# ax1.set_ylim([0, 73])
# ax1.set_xlim([time_ticks[0], time_ticks[-1]])
# cb1 = plt.colorbar(cf1, extend="both", extendrect = True)
# cb1.ax.set_ylabel("Net exchange flux difference (%)", rotation=270, labelpad=20)

# fig.tight_layout()
# fig.set_size_inches(10, 3.5)


# In[33]:


fname = out_dir + 'mass_balance/finger_flux_abs_diff.jpg'
fig.savefig(fname, dpi=300)
plt.close(fig)


# ### use original flux

# In[128]:


flux_array_diff_sign = (flux_array_6h - flux_array_1w)/flux_array_1w*100


# In[132]:


gs = gridspec.GridSpec(1, 1)
fig = plt.figure()
ax1 = fig.add_subplot(gs[0, 0])
cf1 = ax1.contourf(real_time_2,
                   0.5 * (mass1_length[1:] + mass1_length[:-1]),
                   np.transpose(flux_array_diff_sign),
                   cmap=plt.cm.jet,
                   levels=np.arange(-500, 500, 100),
                   extend="both",
                   )
ax1.set_ylabel("")
ax1.set_ylabel("Distance to Downstream Location (km)")
ax1.set_xticks(time_ticks)
ax1.set_ylim([0, 73])
ax1.set_xlim([time_ticks[0], time_ticks[-1]])
cb1 = plt.colorbar(cf1, extend="both")
cb1.ax.set_ylabel("Net exchange flux difference (%)", rotation=270, labelpad=20)
fig.tight_layout()
fig.set_size_inches(10, 3.5)


# In[135]:


fig.savefig(fig_finger_flux_diff, dpi=300)
plt.close(fig)


# ### use 0, 1 represent flux

# In[9]:


flux_ind_6h = [0 if iflux <0 else 1 for iflux in flux_array_6h.flatten()] # 0--gaining; 1--losing
flux_ind_1w = [0 if iflux <0 else 1 for iflux in flux_array_1w.flatten()] # 0--gaining; 1--losing
flux_ind_6h = np.asarray(flux_ind_6h).reshape(flux_array_6h.shape[0], flux_array_6h.shape[1])
flux_ind_1w = np.asarray(flux_ind_1w).reshape(flux_array_1w.shape[0], flux_array_1w.shape[1])    


# In[10]:


# 1--switch to gaining; -1 -- switch to losing; 0 -- does not change
flux_array_diff_num =  flux_ind_6h - flux_ind_1w 

flux_array_diff_num = flux_array_diff_num.astype('float')

flux_array_diff_num[flux_array_diff_num == 0] =  np.nan


# In[29]:


get_ipython().run_line_magic('matplotlib', 'inline')
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
fig = plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios=[1,1])
gs.update(hspace = 0.1) # adjust vertical spacing b/w subplots
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1], sharex=ax1)

# river_north_datum.plot(x='time', y='wl',color= 'black', linewidth = 0.5, ax=ax1, legend=None)
river_middle_datum.plot(x='time', y='wl',color= 'grey',linewidth = 0.5, ax=ax1, legend=None, label = "base case")
# river_south_datum.plot(x='time', y='wl',color= 'black', linewidth = 0.5, ax=ax1, legend=None)

river_middle_datum_s.plot(x='time', y='wl',color= 'black',linewidth = 0.5, ax=ax1, legend=None, label = "without dam operation")

ax1.set_ylabel('River stage (m)')
ax1.set_xlabel('')
ax1.minorticks_off()
ax1.legend(loc = "upper right", frameon = False)

colors = [(0, 0, 1), (1, 0, 0)]
cm = LinearSegmentedColormap.from_list("mylist", colors, N=2)

cf1 = ax2.contourf(real_time_2 ,
                   0.5 * (mass1_length[1:] + mass1_length[:-1]),
                   -np.transpose(flux_array_diff_num),
                   cmap=cm
#                    colors = ('blue', 'gray', 'red'),
#                    levels=np.arange(-1, 1, 1),
#                    extend="both"
                   )

# ax2.imshow(np.transpose(flux_array_diff_num), cmap=cm.gist_rainbow)
# ax2.set_ylabel("")

ylabel = '\n'.join(wrap("Distance to downstream end (km)", 20))
ax2.set_ylabel(ylabel)
ax2.set_xticks(time_ticks)
ax2.set_ylim([0, 73])
ax2.set_xlim([time_ticks[0], time_ticks[-1]])
# cb1 = plt.colorbar(cf1, extend="both")
# cb1.ax.set_ylabel("Net exchange flux difference", rotation=270, labelpad=20)
# fig.tight_layout()
fig.set_size_inches(8, 6)


# In[31]:


fig.savefig(fig_finger_flux + 'finger_flux_01.png', transparent = True, dpi=300)
plt.close(fig)


# ## plot histgram with PDF

# In[19]:


# n_bins = 50
# bins = [-0.35,-0.25,-0.15,-0.05,0.05,0.15,0.25,0.35]

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=False)

# We can set the number of bins with the `bins` kwarg
# axs[0].hist(-flux_array_6h, bins=bins, facecolor = 'black')
# axs[1].hist(-flux_array_1w, bins=bins, facecolor = 'black')

axs[0].hist(-flux_array_6h.flatten(), np.arange(-0.7, 0.5, 0.02), facecolor = 'b', normed = True)
axs[1].hist(-flux_array_1w.flatten(), np.arange(-0.7, 0.5, 0.02), facecolor = 'g', normed = True)

lnspc = np.linspace(-0.7, 0.7, len(flux_array_6h.flatten()))

# choose distribution funcction
# params = stats.exponpow.fit(-flux_array_6h.flatten(), floc = 0)
# pdf = stats.exponpow.pdf(lnspc, *params)

params = stats.norm.fit(-flux_array_6h.flatten())
pdf = stats.norm.pdf(lnspc, *params)

params2 = stats.norm.fit(-flux_array_1w.flatten())
pdf2 = stats.norm.pdf(lnspc, *params2)
# fit PDF
axs[0].plot(lnspc, pdf)
axs[1].plot(lnspc, pdf2)
# plt.yscale('log')

axs[0].set_title('(a) with daily fluctuations')
axs[1].set_title('(b) without daily fluctuations')
axs[0].set_xlabel('Exchange flux (m/d)')
axs[1].set_xlabel('Exchange flux (m/d)')
axs[0].set_ylabel('Probability')
axs[0].set_xlim([-0.7, 0.7])
axs[1].set_xlim([-0.7, 0.7])


fig.set_size_inches(10, 4)


# In[18]:


# a is your data array
a = flux_array_6h
hist, bins = np.histogram(a, bins=np.arange(-0.7, 0.7, 0.02), normed=False, density= True)
bin_centers = (bins[1:]+bins[:-1])*0.5
plt.plot(bin_centers, hist)


# ### combo histogram with 1 to 1 plot

# In[11]:


fig, (axs, axs2) = plt.subplots(1, 2)

def colorcode(xx, yy):
    color = []
    for x, y in zip(xx, yy):
        if x < 0 and y > 0 :
            icolor = 'red'
#             red_ct += 1
        elif x > 0 and y < 0 :
            icolor = 'blue'
#             blue_ct += 1
        else:
            icolor = 'gray'
        color.append(icolor)
    return color   

# We can set the number of bins with the `bins` kwarg
# axs[0].hist(-flux_array_6h, bins=bins, facecolor = 'black')
# axs[1].hist(-flux_array_1w, bins=bins, facecolor = 'black')

axs.hist(-flux_array_6h.flatten(), np.arange(-0.7, 0.7, 0.02), facecolor = 'none', edgecolor = 'b', histtype='step', log=True, label='base case')
axs.hist(-flux_array_1w.flatten(), np.arange(-0.7, 0.7, 0.02), facecolor = 'none', edgecolor = 'g', histtype='step', log=True, label='without dam operation')

lnspc = np.linspace(-0.7, 0.7, len(flux_array_6h.flatten()))

m, s = stats.norm.fit(-flux_array_6h.flatten())

# axs.yscale('log')
axs.set_xlabel('Exchange flux (m/d)', fontsize=12)
axs.set_ylabel('Counts', fontsize=12)
axs.set_xlim([-0.7, 0.7])
axs.set_ylim([0, 1e7])

axs.legend("upper left", frameon=False)

axs.tick_params(axis = 'both', which = 'major', labelsize = 10)

xx = -flux_array_1w.flatten()
yy = -flux_array_6h.flatten()

axs2.scatter(xx, yy, marker='o', edgecolors = colorcode(xx, yy), facecolors = 'none')

axs2.plot((-1, 1), (0, 0), 'k--')
axs2.plot((0, 0), (-1, 1), 'k--')
axs2.plot((-1,1), (-1,1), 'k-')
# ax.set_title('(a) with daily fluctuations')

axs2.set_xlabel('Exchange flux without dam operation (m/d)', fontsize=12)
axs2.set_ylabel('Exchange flux under base case (m/d)', fontsize=12)
axs2.set_xlim([-0.7, 0.7])
axs2.set_ylim([-0.7, 0.7])
axs2.set_aspect('equal')  

axs2.tick_params(axis = 'both', which = 'major', labelsize = 10)

fig.set_size_inches(12, 5)


# In[37]:


fig.savefig(fig_finger_flux + 'hist1d_combo.png', dpi=300, transparent = True)
plt.close(fig)


# **calculate % of flow reversal**

# In[12]:


red_ct, blue_ct = 0, 0
for x, y in zip(xx, yy):
    if x < 0 and y > 0 :
#         icolor = 'red'
        red_ct += 1
    elif x > 0 and y < 0 :
#         icolor = 'blue'
        blue_ct += 1
    else:
        icolor = 'gray'


# In[16]:


print(red_ct, blue_ct, red_ct/len(xx), blue_ct/len(xx))


# ## plot PDF

# In[54]:



# this create the kernel, given an array it will estimate the probability over that values
kde = gaussian_kde( -flux_array_6h.flatten() )
# these are the values over wich your kernel will be evaluated
dist_space = np.linspace( np.nanmin(-flux_array_6h), np.nanmax(-flux_array_6h), 100 )
# plot the results
plt.plot( dist_space, kde(dist_space) )


# ## count max,min of flux for each year

# In[24]:


flux_2011 = flux_array_6h[98:1558 , :]
flux_2015 = flux_array_6h[5939:7400 , :]


# ## 1 to 1 plot

# In[33]:


fig, ax = plt.subplots(1, 1)

ax.plot(-flux_array_1w.flatten(), -flux_array_6h.flatten(), 'ko', mfc = 'None', alpha = 0.5)

ax.plot((-1,1), (-1,1), 'r-')
# ax.set_title('(a) with daily fluctuations')

ax.set_xlabel('Exchange flux in WR (m/d)', fontsize=12)
ax.set_ylabel('Exchange flux in DR (m/d)', fontsize=12)
ax.set_xlim([-0.7, 0.7])
ax.set_ylim([-0.7, 0.7])
ax.set_aspect('equal')   
ax.tick_params(axis = 'both', which = 'major', labelsize = 10)

# fig.tight_layout()
fig.set_size_inches(6, 5)


# In[34]:


fname = fig_finger_flux + 'flux_one2one.png'
fig.savefig(fname, dpi=300, transparent = True)
plt.close(fig)


# # plot net exchange bar plot

# In[53]:


# calculate net mass
n_segment = len(time_ticks) - 1
sum_mass = np.array([0.] * n_segment)
all_mass = np.array([0.] * n_segment)
for i_segment in range(n_segment):
    select_index = []
    for i_index in range(len(real_time)):
        if (real_time[i_index] >= time_ticks[i_segment] and
                real_time[i_index] < time_ticks[i_segment + 1]):
            select_index.append(i_index)
    sum_mass[i_segment] = total_mass[select_index[-1]] -         total_mass[select_index[0] - 1]
    time_inverval = real_time[select_index[-1]] - real_time[select_index[0] - 1]
    time_scale = 365.25 * 24 * 3600 / time_inverval.total_seconds()
    sum_mass[i_segment] = sum_mass[i_segment] * time_scale / 1000

abs_mass = np.array([0.] * n_segment)
out_mass = np.array([0.] * n_segment)
in_mass = np.array([0.] * n_segment)


# In[54]:


# calculate dam discharge
discharge_flow = np.array([0.] * n_segment)
for i_segment in range(n_segment):
    select_index = []
    for i_index in range(len(discharge_time)):
        if (discharge_time[i_index] >= time_ticks[i_segment] and
                discharge_time[i_index] < time_ticks[i_segment + 1]):
            select_index.append(i_index)
    sum_discharge = sum(np.asarray([discharge_value[i]
                                    for i in select_index]).astype(float))
    sum_discharge = sum_discharge * 3600 * 24 * (0.3048**3)
    # print("{:.5E}".format(sum_discharge))
    discharge_flow[i_segment] = sum_discharge


# In[56]:


## plot bar plots for net gaining volumn
# plot dam discharge
start_year = 2011
fig_name = fig_net_exchange_bar
gs = gridspec.GridSpec(1, 2)
fig = plt.figure()
ax0 = fig.add_subplot(gs[0, 0])
ax0.bar(start_year + np.arange(n_segment), discharge_flow, color="black")
ax0.set_ylim([0, 1.5e11])
ax0.set_xlabel('Time (year)')
ax0.set_ylabel('River Discharge ($m^3$/year)')
ax0.set_title("Priest Rapids Dam Discharge", y=1.05)

# plot net exchange
ax1 = fig.add_subplot(gs[0, 1])
ax1.bar(start_year + np.arange(n_segment), -sum_mass, color="blue")
#ax1.set_ylim([0, 1e8])
ax1.set_xlabel('Time (year)')
ax1.set_ylabel('Exchange flux ($m^3$/year)')
ax1.set_title("River Net Exchange Volume", y=1.05)
fig.tight_layout()
fig.subplots_adjust(left=0.2,
                    right=0.95,
                    bottom=0.08,
                    top=0.85,
                    wspace=0.30,
                    hspace=0.38
                    )
fig.set_size_inches(7, 3)
# fig.savefig(fig_name, dpi=600, transparent=True)


# In[55]:


fig.savefig(fig_name, dpi=600)
plt.close(fig)


# In[63]:


# a = -sum_mass
np.savetxt(fname_net_exchange_txt, -sum_mass, fmt = '%.2e')


# ## import river geometry

# In[43]:


river_geo = pd.read_csv(fname_river_geo)

river_geo['x'] = (river_geo['x'] - model_origin[0])/1000
river_geo['y'] = (river_geo['y'] - model_origin[1])/1000

polygon = Polygon(river_geo.loc[:, ["x", "y"]].values)
river_x,river_y = polygon.exterior.xy


# # plot flux exchange across riverbed snapshots

# In[91]:


## plot exchange flux accross riverbed snapshots
for itime in np.arange(98, 7399, 20):
    print(real_time[itime])
    yx_flux = np.asarray([np.nan] * (ny * nx)).reshape(ny, nx)
    for isection in range(len(mass1_sections)):
        # need minus 1 as python index started with 0
        cell_ids = list(material_file["Regions"]
                        [mass1_sections[isection]]['Cell Ids'])
        cell_ids = (np.asarray(cell_ids) - 1).astype(int)
        xy_cell_index = [grids[i, 0:2] for i in cell_ids]
        xy_cell_index = np.unique(xy_cell_index, axis=0)
        for iindex in range(len(xy_cell_index)):
            yx_flux[xy_cell_index[iindex][1],
                    xy_cell_index[iindex][0]] = flux_array[itime, isection]
    
    gs = gridspec.GridSpec(1, 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(line1_x, line1_y, "black", alpha =0.7)
    ax1.plot(line2_x, line2_y, "black", alpha =0.7)
    ax1.plot(line3_x, line3_y, "black", alpha =0.7)
    cf1 = ax1.imshow(-yx_flux,
                     cmap=plt.cm.jet,
                     origin="lower",
#                      aspect = 'equal',
#                      levels = np.arange(-0.1, 0.1, 0.01),
                     vmin=-0.1,
                     vmax=0.1,
                     extent=[(x[0] - 0.5 * dx[0]) / 1000,
                             (x[-1] + 0.5 * dx[0]) / 1000,
                             (y[0] - 0.5 * dy[0]) / 1000,
                             (y[-1] + 0.5 * dy[0]) / 1000]
                     )
    # plot river geometry
    ax1.plot(river_x, river_y, color='black', alpha=0.7, linewidth=1, solid_capstyle='round', zorder=2)    
    
    ax1.set_xlabel("X (km)")
    ax1.set_ylabel("Y (km)")
#     ax1.set_aspect("equal", "datalim")
    ax1.set_xlim([np.min(x_grids) / 1000, np.max(x_grids) / 1000])
    ax1.set_ylim([np.min(x_grids) / 1000, np.max(x_grids) / 1000])
    cb1 = plt.colorbar(cf1, extend="both")
    cb1.ax.set_ylabel("Exchange flux (m/d)", rotation=270, labelpad=20)
    fig.tight_layout()
    fig.set_size_inches(6, 5)   

    fig_name = fig_flux_snapshot + str(real_time[itime]) + ".png"
    fig.savefig(fig_name, dpi=300)
    plt.close(fig)


# ## plot flux exchange across riverbed snapshots for _north block_

# In[80]:


for itime in range(len(real_time[0:1])):
    print(real_time[itime])
    yx_flux = np.asarray([np.nan] * (ny * nx)).reshape(ny, nx)
    for isection in range(len(mass1_sections)):
        # need minus 1 as python index started with 0
        cell_ids = list(material_file["Regions"]
                        [mass1_sections[isection]]['Cell Ids'])
        cell_ids = (np.asarray(cell_ids) - 1).astype(int)
        xy_cell_index = [grids[i, 0:2] for i in cell_ids]
        xy_cell_index = np.unique(xy_cell_index, axis=0)
        for iindex in range(len(xy_cell_index)):
            yx_flux[xy_cell_index[iindex][1],
                    xy_cell_index[iindex][0]] = flux_array[itime, isection]
    fig_name = fig_block_north_flux_snapshot + str(real_time[itime]) + ".png"
    gs = gridspec.GridSpec(1, 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(gs[0, 0])
    cf1 = ax1.imshow(-yx_flux,
                     cmap=plt.cm.jet,
                     origin="lower",
                     vmin=-0.1,
                     vmax=0.1,
                     extent=[(x[0] - 0.5 * dx[0]) / 1000,
                             (x[-1] + 0.5 * dx[0]) / 1000,
                             (y[0] - 0.5 * dy[0]) / 1000,
                             (y[-1] + 0.5 * dy[0]) / 1000]
                     )
    ax1.axis("off")
    ax1.set_aspect("equal", "datalim")
    ax1.set_xlim(block_north_x)
    ax1.set_ylim(block_north_y)
    # ax1.set_xlabel("Easting (km)")
    # ax1.set_ylabel("Northing (km)")
    fig.tight_layout()
    fig.set_size_inches(6.5, 5.5)
#     fig.savefig(fig_name, dpi=600)
#     plt.close(fig)


# In[81]:


fig.savefig(fig_name, dpi=600)
plt.close(fig)


# ## plot flux exchange across riverbed snapshots for _middle block_

# In[82]:


for itime in range(len(real_time[0:1])):
    print(real_time[itime])
    yx_flux = np.asarray([np.nan] * (ny * nx)).reshape(ny, nx)
    for isection in range(len(mass1_sections)):
        # need minus 1 as python index started with 0
        cell_ids = list(material_file["Regions"]
                        [mass1_sections[isection]]['Cell Ids'])
        cell_ids = (np.asarray(cell_ids) - 1).astype(int)
        xy_cell_index = [grids[i, 0:2] for i in cell_ids]
        xy_cell_index = np.unique(xy_cell_index, axis=0)
        for iindex in range(len(xy_cell_index)):
            yx_flux[xy_cell_index[iindex][1],
                    xy_cell_index[iindex][0]] = flux_array[itime, isection]
    fig_name = fig_block_middle_flux_snapshot + str(real_time[itime]) + ".png"
    gs = gridspec.GridSpec(1, 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(gs[0, 0])
    cf1 = ax1.imshow(-yx_flux,
                     cmap=plt.cm.jet,
                     origin="lower",
                     vmin=-0.1,
                     vmax=0.1,
                     extent=[(x[0] - 0.5 * dx[0]) / 1000,
                             (x[-1] + 0.5 * dx[0]) / 1000,
                             (y[0] - 0.5 * dy[0]) / 1000,
                             (y[-1] + 0.5 * dy[0]) / 1000]
                     )
    ax1.axis("off")
    ax1.set_aspect("equal", "datalim")
    ax1.set_xlim(block_middle_x)
    ax1.set_ylim(block_middle_y)
    # ax1.set_xlabel("Easting (km)")
    # ax1.set_ylabel("Northing (km)")
    fig.tight_layout()
    fig.set_size_inches(6.5, 5.5)
#     fig.savefig(fig_name, dpi=600)
#     plt.close(fig)


# In[83]:


fig.savefig(fig_name, dpi=600)
plt.close(fig)


# ## plot flux exchange across riverbed snapshots for _south block_

# In[84]:


for itime in range(len(real_time[0:1])):
    print(real_time[itime])
    yx_flux = np.asarray([np.nan] * (ny * nx)).reshape(ny, nx)
    for isection in range(len(mass1_sections)):
        # need minus 1 as python index started with 0
        cell_ids = list(material_file["Regions"]
                        [mass1_sections[isection]]['Cell Ids'])
        cell_ids = (np.asarray(cell_ids) - 1).astype(int)
        xy_cell_index = [grids[i, 0:2] for i in cell_ids]
        xy_cell_index = np.unique(xy_cell_index, axis=0)
        for iindex in range(len(xy_cell_index)):
            yx_flux[xy_cell_index[iindex][1],
                    xy_cell_index[iindex][0]] = flux_array[itime, isection]
    fig_name = fig_block_south_flux_snapshot + str(real_time[itime]) + ".png"
    gs = gridspec.GridSpec(1, 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(gs[0, 0])
    cf1 = ax1.imshow(-yx_flux,
                     cmap=plt.cm.jet,
                     origin="lower",
                     vmin=-0.1,
                     vmax=0.1,
                     extent=[(x[0] - 0.5 * dx[0]) / 1000,
                             (x[-1] + 0.5 * dx[0]) / 1000,
                             (y[0] - 0.5 * dy[0]) / 1000,
                             (y[-1] + 0.5 * dy[0]) / 1000]
                     )
    ax1.axis("off")
    ax1.set_aspect("equal", "datalim")
    ax1.set_xlim(block_south_x)
    ax1.set_ylim(block_south_y)
    # ax1.set_xlabel("Easting (km)")
    # ax1.set_ylabel("Northing (km)")
    fig.tight_layout()
    fig.set_size_inches(6.5, 5.5)
#     fig.savefig(fig_name, dpi=600)
#     plt.close(fig)


# In[85]:


fig.savefig(fig_name, dpi=600)
plt.close(fig)

