#!/usr/bin/env python
# coding: utf-8

# In[19]:


# %matplotlib inline
import numpy as np
import h5py as h5
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# from matplotlib.patches import Polygon
# from matplotlib.collections import PatchCollection
from datetime import datetime, timedelta
import pandas as pd
from shapely.geometry.polygon import Polygon
from natsort import natsorted, ns, natsort_keygen
import re
import shapefile as shp
import geopandas as gpd

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

from tqdm.notebook import trange, tqdm


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150


# In[3]:


def batch_delta_to_time(origin, x, time_format, delta_format):
    y = []
    for ix in x:
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
    return(y)


# In[4]:


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)  


# # I/O files

# In[16]:


#input
case_name = "HFR_model_100x100x2_cyclic/"

model_dir = "/global/cscratch1/sd/pshuai/" + case_name
# fname_model_origin = model_dir + "model_origin.txt"
fname_material_h5 = model_dir + "HFR_material_river.h5"
fname_pflotran_h5 = model_dir + "pflotran_100x100x2_cyclic_2011_2015_full.h5"

data_dir = "/global/project/projectdirs/pflotran/pin/Reach_scale_model/data/"
# fname_mass1_coord = data_dir + "MASS1/coordinates.csv"
# fname_hf_shp = data_dir + "hanfordArea.shp"
#output
out_dir = "/global/project/projectdirs/pflotran/pin/Reach_scale_model/Outputs/" + case_name
# fig_wl = out_dir + 'wl/'
fig_tracer = out_dir + "tracer_viri/"


data_dir = "/global/project/projectdirs/pflotran/pin/Reach_scale_model/data/"
fname_river_geo = data_dir + "river_geometry_manual_v2.csv"

fname_hanford_shp = '/global/project/projectdirs/pflotran/pin/Reach_scale_model/data/shapefile/hanfordArea_newCRS.shp'
fname_islands_shp = "/global/project/projectdirs/pflotran/pin/HFR-thermal/data/islands.shp"
# output_dir = "/Users/song884/remote/reach/Outputs/HFR_100x100x5_6h_bc/"
# fig_dir = "/Users/song884/remote/reach/figures/HFR_100x100x5_6h_bc/wl/"
# data_dir = "/Users/song884/remote/reach/data/"


# In[6]:


date_origin = datetime.strptime("2007-03-28 12:00:00", "%Y-%m-%d %H:%M:%S")
# model_origin = np.genfromtxt(
#     fname_model_origin, delimiter=" ", skip_header=1)
model_origin = [551600, 104500]


# ## import mass1 coord

# In[6]:


# ## read mass1 coordinates
# section_coord = np.genfromtxt(
#     fname_mass1_coord, delimiter=",", skip_header=1)
# section_coord[:, 1] = section_coord[:, 1] - model_origin[0]
# section_coord[:, 2] = section_coord[:, 2] - model_origin[1]
# line1 = section_coord[0, 1:3] / 1000
# line2 = section_coord[int(len(section_coord[:, 1]) / 2), 1:3] / 1000
# line3 = section_coord[-1, 1:3] / 1000

# line1_x = [line1[0]] * 2
# line1_y = [line1[1] - 5, line1[1] + 5]
# line2_x = [line2[0] - 5, line2[0] + 5]
# line2_y = [line2[1]] * 2
# line3_x = [line3[0] - 5, line3[0] + 5]
# line3_y = [line3[1]] * 2


# ## import model dimension

# In[7]:


# all_h5 = glob.glob(fname_pflotran_h5) # find all "pflotran*.h5" files
# all_h5 = np.sort(all_h5)

input_h5 = h5.File(fname_pflotran_h5, "r")
groups = list(input_h5.keys()) # create a list with group names
time_index = [s for s, s in enumerate(groups) if "Time:" in s] # enumerate returns its index (index, string)


# In[8]:


# sort time based on scientific value
time_index = sorted(time_index, key = lambda time: float(time[7:18]))

real_time = [str(batch_delta_to_time(date_origin, [float(itime[7:18])], "%Y-%m-%d %H:%M:%S", "hours")[0])
              for itime in time_index]


# In[9]:


x_grids = list(input_h5["Coordinates"]['X [m]'])
y_grids = list(input_h5["Coordinates"]['Y [m]'])
z_grids = list(input_h5["Coordinates"]['Z [m]'])


dx = np.diff(x_grids)
dy = np.diff(y_grids)
dz = np.diff(z_grids)

nx = len(dx)
ny = len(dy)
nz = len(dz)

# x,y,z coordinates at cell center
x = x_grids[0] + np.cumsum(dx) - 0.5 * dx[0]
y = y_grids[0] + np.cumsum(dy) - 0.5 * dy[0]
z = z_grids[0] + np.cumsum(dz) - 0.5 * dz[0]

# create grids--a list of arrays based nx, ny, nz
grids = np.asarray([(x, y, z) for z in range(nz)
                    for y in range(ny) for x in range(nx)])


# ## import river cells

# In[12]:


# # open file for reading
# material_h5 = h5.File(fname_material_h5, "r") 

# # read river cell ids
# river_cells = []
# for i_region in list(material_h5['Regions'].keys()):
#     river_cells = np.append(river_cells, np.asarray(
#         list(material_h5["Regions"][i_region]["Cell Ids"])))
# river_cells = np.unique(river_cells).astype(int)
# river_cells = river_cells - 1  # need minus 1 as python index started with 0
# # label river cells in x-y plane with '1'
# yx_river = np.asarray([np.nan] * (ny * nx)).reshape(ny, nx) # initialize ny*nx array with nan value
# for icell in river_cells:
#     yx_river[grids[icell, 1], grids[icell, 0]] = 1
    
# material_h5.close()


# ## import river geometry

# In[10]:


river_geo = pd.read_csv(fname_river_geo)

# river_geo['x'] = (river_geo['x'] - model_origin[0])/1000
# river_geo['y'] = (river_geo['y'] - model_origin[1])/1000

polygon = Polygon(river_geo.loc[:, ["x", "y"]].values)
river_x,river_y = polygon.exterior.xy


# In[44]:


# sf = shp.Reader(fname_hf_shp)

# plt.figure()
# for shape in sf.shapeRecords():
#     x = [i[0] for i in shape.shape.points[:]]
#     y = [i[1] for i in shape.shape.points[:]]
#     plt.plot(x,y)


# # plot tracer contour

# In[11]:


# north Horn area
block1_x = np.array([569600, 581600])
block1_y = np.array([149500, 156500])


# In[24]:


# %matplotlib inline
# patches = []
# loop over time step
for itime in tqdm(time_index[::]):
    #itime = 6003 # 2011-5-31
#     itime = 6383 # 2011-9-3
#     print(itime)
    # initialize total head
    temp_max_tracer = np.asarray([np.nan] * (ny * nx)).reshape(ny, nx)
    # read pressure
    temp_total_tracer = np.asarray(list(input_h5[itime]["Total_Tracer [M]"]))
    
    temp_sat = np.asarray(list(input_h5[itime]["Liquid_Saturation"]))

    # find the maximum tracer plane
    for ix in range(nx):
        for iy in range(ny):
            sat_index = np.array(np.where(temp_sat[ix, iy, :] >= 1)) # filter out unsat zone
            if sat_index.size > 0:
                max_index = np.argmax(temp_total_tracer[ix, iy, sat_index])
                temp_max_tracer[iy, ix] = temp_total_tracer[ix, iy, max_index] # for contour plot, temp_max_tracer must has shape of (ny, nx)
    
    temp_max_tracer[temp_max_tracer < 0.1] = np.nan
    
    real_itime = batch_delta_to_time(date_origin, [float(
        itime[7:18])], "%Y-%m-%d %H:%M:%S", "hours")
    real_itime = str(real_itime[0])
    
#     print(real_itime)
    ## plot tracer contour

    gs = gridspec.GridSpec(1, 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(gs[0, 0])
       
    cf1 = ax1.contourf(x + model_origin[0] , y + model_origin[1], temp_max_tracer,
                       cmap=plt.cm.viridis,
                       levels=np.arange(0, 1, 0.01),
                       vmin=0,
                       vmax=1,
                       extend="both",
#                        V=np.arange(0, 1, 0.1)
                       )

    # plot river shape
    ax1.plot(river_x, river_y, color='black', alpha=0.7, linewidth=1, solid_capstyle='round', zorder=2)
    
    # plot hanford areas
    hf_shape=gpd.read_file(fname_hanford_shp)
    hf_shape.plot(facecolor = 'none', edgecolor = 'gray', ax=ax1, alpha=1)
    
    #plot island
    island_shape=gpd.read_file(fname_islands_shp)
    island_shape.plot(facecolor = 'none', edgecolor = 'gray', ax=ax1, alpha=0.7)
 
    ax1.set_xlabel("Easting (m)", fontsize=12)
    ax1.set_ylabel("Northing (m)", fontsize=12)

    ax1.set_xlim([np.min(x_grids) + model_origin[0] + 400, np.max(x_grids) + model_origin[0] + 400])
    ax1.set_ylim([np.min(x_grids) + model_origin[1] + 500, np.max(x_grids) + model_origin[1] + 500])
    ax1.set_aspect("equal")
    cb1 = plt.colorbar(cf1, spacing = "uniform", ticks=np.arange(0, 1.1, 0.1))  # ,
    #                           orientation="horizontal", shrink=0.8, aspect=25)
    cb1.ax.set_ylabel("River tracer (-)", rotation=270, labelpad=20, fontsize=12)
    
    ax1.tick_params(axis = 'both', which = 'major', labelsize = 10)

    # this is another inset axes over the main axes
    a = plt.axes([0.2, 0.1, 0.35, 0.35]) #[x0, y0, len_x, len_y]
    a.contourf(x + model_origin[0] , y + model_origin[1], temp_max_tracer,
                       cmap=plt.cm.viridis,
                       levels=np.arange(0, 1, 0.01),
                       vmin=0,
                       vmax=1,
                       extend="both",
#                        V=np.arange(0, 1, 0.1)
                       )
    a.plot(river_x, river_y, color='black', alpha=0.7, linewidth=1, solid_capstyle='round', zorder=2)
    hf_shape.plot(facecolor = 'none', edgecolor = 'gray', ax=a, alpha=1)
    island_shape.plot(facecolor = 'none', edgecolor = 'gray', ax=a, alpha=0.7)
    
    a.tick_params(axis = 'both', left=None, top=None, right=None, bottom=None, 
                  labelleft=None, labeltop=None, labelright=None, labelbottom=None)
    a.set_xlim(block1_x)
    a.set_ylim(block1_y)
    a.set_aspect("equal")    
    mark_inset(ax1, a, loc1=1, loc2=2, fc="none", ec="black")

#     fig.tight_layout()
#     cf3 = ax1.contourf(x / 1000, y / 1000, yx_river, colors="black")
    fig.set_size_inches(8, 6)
    

    fig_name = fig_tracer + real_itime + ".png"
    fig.savefig(fig_name, dpi=300, transparent=False)
    plt.close(fig)   


# In[36]:





# **save tracer plume as .txt output**

# In[17]:


temp_max_tracer_copy = np.nan_to_num(temp_max_tracer)
# temp_max_tracer_copy = np.transpose(temp_max_tracer_copy)

fname = out_dir + 'tracer_' + real_itime + '.txt'
np.savetxt(fname, temp_max_tracer_copy, delimiter=' ') 


# In[19]:


input_h5.close()


# ## plot zoomed in tracer plume

# In[26]:


# north Horn area
block1_x = np.array([561000, 591000])
block1_y = np.array([139000, 158000])

# block1_x = np.array([569600, 581600])
# block1_y = np.array([149500, 156500])
# block1_x = block1_x - model_origin[0]
# block1_y = block1_y - model_origin[1]


# In[12]:


# 300 Area
block2_x = np.array([592600, 597600])
block2_y = np.array([113500, 123500])
block2_x = block2_x - model_origin[0]
block2_y = block2_y - model_origin[1]


# In[34]:


# %matplotlib inline
# patches = []
# loop over time step
for itime in time_index[30:31:]:
    #itime = 6003 # 2011-5-31
#     itime = 6383 # 2011-9-3
#     print(itime)
    # initialize total head
    temp_max_tracer = np.asarray([np.nan] * (ny * nx)).reshape(ny, nx)
    # read pressure
    temp_total_tracer = np.asarray(list(input_h5[itime]["Total_Tracer [M]"]))
    
    temp_sat = np.asarray(list(input_h5[itime]["Liquid_Saturation"]))

    # find the maximum tracer plane
    for ix in range(nx):
        for iy in range(ny):
            sat_index = np.array(np.where(temp_sat[ix, iy, :] >= 1)) # filter out unsat zone
            if sat_index.size > 0:
                max_index = np.argmax(temp_total_tracer[ix, iy, sat_index])
                temp_max_tracer[iy, ix] = temp_total_tracer[ix, iy, max_index] # for contour plot, temp_max_tracer must has shape of (ny, nx)
    
    temp_max_tracer[temp_max_tracer < 0.1] = np.nan
    
    real_itime = batch_delta_to_time(date_origin, [float(
        itime[7:18])], "%Y-%m-%d %H:%M:%S", "hours")
    real_itime = str(real_itime[0])
    
    print(real_itime)
    ## plot tracer contour

    gs = gridspec.GridSpec(1, 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(gs[0, 0])
#     ax1.plot(line1_x, line1_y, "black", alpha =0.7)
#     ax1.plot(line2_x, line2_y, "black", alpha =0.7)
#     ax1.plot(line3_x, line3_y, "black", alpha =0.7)
    
#     ax1.plot(river_x, river_y, color='#6699cc', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
    
    cf1 = ax1.contourf(x + model_origin[0] , y + model_origin[1], temp_max_tracer,
                       cmap=plt.cm.viridis,
                       levels=np.arange(0, 1, 0.01),
                       vmin=0,
                       vmax=1,
                       extend="both",
#                        V=np.arange(0, 1, 0.1)
                       )
#     cf2 = ax1.contour(x / 1000, y / 1000, temp_max_tracer,
#                       colors="grey",
#                       levels=[0.5],
#                       linewidths=1,
#                       vmin=0,
#                       vmax=1)
#     plt.clabel(cf2, inline = True, fmt = '%3.0d', fontsize = 10)

    # plot river shape
    ax1.plot(river_x, river_y, color='black', alpha=0.7, linewidth=1, solid_capstyle='round', zorder=2)
    
    # plot hanford areas
    hf_shape=gpd.read_file(fname_hanford_shp)
    hf_shape.plot(facecolor = 'none', edgecolor = 'gray', ax=ax1, alpha=1)
 
#     ax1.set_xlabel("Easting (m)", fontsize=12)
#     ax1.set_ylabel("Northing (m)", fontsize=12)

    ax1.set_xlim(block1_x)
    ax1.set_ylim(block1_y)
    ax1.set_aspect("equal", "datalim")
#     cb1 = plt.colorbar(cf1, spacing = "uniform", ticks=np.arange(0, 1.1, 0.1))  # ,
#     #                           orientation="horizontal", shrink=0.8, aspect=25)
#     cb1.ax.set_ylabel("River tracer (-)", rotation=270, labelpad=20, fontsize=12)
    
#     ax1.tick_params(axis = 'both', which = 'major', labelsize = 10)
    ax1.tick_params(axis = 'both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
    
   
#     fig.tight_layout()
#     cf3 = ax1.contourf(x / 1000, y / 1000, yx_river, colors="black")
    fig.set_size_inches(4, 4)


# In[22]:


input_h5.close()


# # write contour to .ASC file

# note: need to reverse the tracer array (from top to bottom is from south to north) since ASCII coord is from north to south

# In[18]:


TheFile=open(out_dir + 'tracer_'+ real_itime + '.asc',"w")
TheFile.write("ncols 600\n")
TheFile.write("nrows 600\n")
TheFile.write("xllcorner {}\n".format(model_origin[0]))
TheFile.write("yllcorner {}\n".format(model_origin[1])) 
TheFile.write("cellsize {}\n".format(dx[0])) 
TheFile.write("NODATA_value  0\n")
 
TheFormat="{0} "
 
filename = out_dir + 'tracer_' + real_itime + '.txt'
 
ncols= 600
nrows= 600
 
table = []
data =[]
with open(filename) as my_file:
    for line in my_file: #read line by line
 
         numbers_str = line.split() #split string by " "(space)
        #convert string to floats
         numbers_str_new = ["{0:.2f}".format(float(x)) for x in numbers_str]  #map(float,numbers_str) works too (convert feet to meter with factor of 0.3048)
          
         table.append(numbers_str_new) #store each string line
 
for item in table[::-1]:
    #loop over each line
    for ele in item: #loop over each element in line
#        print(ele) 
        data.append(ele) #store each element one by one 
 
## read into file
for i in range(0, len(data), ncols): #loop over data with stepsize of ncols
#    print(data[i:i+ncols])
    TheFile.write(" ".join(data[i:i + ncols])) #join element in list with space, and write into file
    TheFile.write("\n")#write new line
 
     
TheFile.close()

