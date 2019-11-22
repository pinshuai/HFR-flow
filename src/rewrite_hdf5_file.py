#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py as h5
import glob
import numpy as np
from datetime import datetime, timedelta


# ## I/O file

# In[2]:


#output_dir = "/Users/song884/remote/reach/Outputs/2007_age/"
# output_dir = "/Users/shua784/Paraview/HFR/test_2007_age/"

# output_h5.close()
# output_file = output_dir + "all_age_flux.h5"

case_name = "TH_100x100x2_thermal_newMass/"

model_dir = "/global/cscratch1/sd/pshuai/" + case_name

fname_output_h5 = model_dir + "TH_100x100x2_thermal_all_MASS1-restart-modified-age.h5"
fname_pflotran_h5 = model_dir + "TH_100x100x2_thermal-120h.h5"
fname_h5_old = model_dir + "backup/pflotran_100x100x2_cyclic.h5"
fname_h5_new = model_dir + "pflotran_100x100x2_cyclic.h5"

fname_h5_age = "/global/cscratch1/sd/pshuai/TH-100x100x2-flow_age/TH_100x100x2_flow_age-restart.h5"

fname_h5_thermal = model_dir + "TH_100x100x2_thermal_all_MASS1-restart.h5"


# In[3]:


# view the tree structure of hdf5 file
def print_structure(name, obj):
    print(name)
    
## example: input_h5.visititems(print_structure)


# In[4]:


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


# In[5]:


date_origin = datetime.strptime("2007-03-28 12:00:00", "%Y-%m-%d %H:%M:%S")


# # Combine multiple hdf5 files

# In[46]:


output_h5 = h5.File(fname_output_h5, "w")


# In[47]:


# all_h5 = glob.glob(output_dir + "pflotran*h5")
# all_h5 = np.sort(all_h5)


# **copy the 1st h5 file's groups (`Coordinates`, `Provenance`, `Time: xxx h` ...)**

# In[ ]:


# i_h5 = all_h5[0]
# print(i_h5)
input_h5 = h5.File(fname_h5_old, "r")

groups = list(input_h5.keys())
for i_group in groups:
    print(i_group)
    group_id = output_h5.require_group(i_group)
    datasets = list(input_h5[i_group].keys())
    for i_dataset in datasets:
        input_h5.copy("/" + i_group + "/" + i_dataset,
                      group_id, name=i_dataset)
input_h5.close()


# **copy the 2nd h5 file's group with `Time: xxx h`** 

# In[ ]:


# for i_h5 in all_h5[1:]:
#     print(i_h5)
input_h5 = h5.File(fname_h5_new, "r")

groups = list(input_h5.keys())
groups = [s for s,  s in enumerate(groups) if "Time:" in s]
for i_group in groups:
    print(i_group)
    group_id = output_h5.require_group(i_group)
    datasets = list(input_h5[i_group].keys())
    for i_dataset in datasets:
        input_h5.copy("/" + i_group + "/" + i_dataset,
                      group_id, name=i_dataset)
input_h5.close()


# In[50]:


output_h5.close()


# # Extract output from a single hdf5 file

# In[6]:


input_h5 = h5.File(fname_pflotran_h5, "r")

output_h5 = h5.File(fname_output_h5, "w")


# In[7]:


input_h5.visititems(print_structure)


# In[8]:


groups = list(input_h5.keys())

ts_groups = [s for s,  s in enumerate(groups) if "Time:" in s]

# sort time based on scientific value
ts_groups = sorted(ts_groups, key = lambda time: float(time[7:18]))


# In[9]:


real_time = [str(batch_delta_to_time(date_origin, [float(itime[7:18])], "%Y-%m-%d %H:%M:%S", "hours")[0])
              for itime in ts_groups]


# **copy groups and dsets from `input_h5`**

# **extract every 120h**

# In[11]:


# copy groups before "Time: xxxx h"
for i_group in groups[:2]:
    print(i_group)
    group_id = output_h5.require_group(i_group)
    datasets = list(input_h5[i_group].keys())
    for i_dataset in datasets:
        input_h5.copy("/" + i_group + "/" + i_dataset,
                      group_id, name=i_dataset)

       
# copy each group "Time: xxxx h"
for i_group in ts_groups[::]:
    print(i_group)
    group_id = output_h5.require_group(i_group)
    datasets = list(input_h5[i_group].keys())    
    # select to copy only subset datasets[::2] or the full dataset[::]
    selected_ind = [0,2,4,6,8,9,12,13,15]
    selected_dsets = [datasets[i] for i in selected_ind] 
    for i_dataset in selected_dsets[::]:
        input_h5.copy("/" + i_group + "/" + i_dataset,
                      group_id, name=i_dataset)


# In[12]:


output_h5.close()

input_h5.close()


# # rewrite part of HDF5 file

# **note**: for pflotran restart file, `Pressure` and `Temperature` are stored in the flow side with `Pressure` first ('Checkpoint/PMCSubsurfaceFlow/flow'), P1, T1, P2, T2 ...; `Tracer` and `Age` are stored in the transport side (i.e. 'Checkpoint/PMCSubsurfaceTransport/transport')

# In[6]:


output_h5 = h5.File(fname_output_h5, "w")

input_h5_1 = h5.File(fname_h5_thermal, "r")
input_h5_2 = h5.File(fname_h5_age, "r")


# **copy every dataset in the restart file of TH mode**

# In[7]:


groups = list(input_h5_1.keys())
for i_group in groups:
    print(i_group)
    group_id = output_h5.require_group(i_group)
    datasets = list(input_h5_1[i_group].keys())
    for i_dataset in datasets:
        print(i_dataset)
        input_h5_1.copy("/" + i_group + "/" + i_dataset,
                      group_id, name=i_dataset)


# ## replace initial transport 

# **get `Total_tracer` and `Age_tracer` from age simulation, i.e. the `Primary_variable` under transport group**

# In[8]:


grp_2 = input_h5_2['Checkpoint/PMCSubsurfaceTransport/transport']
dset_2 = grp_2['Primary_Variable'] # Total tracer + age tracer
list(grp_2.keys())


# **replace the original `Primary_Variable` with the new `Primary_Variable` from the age simulation**

# In[9]:


grp_1 = output_h5['Checkpoint/PMCSubsurfaceTransport/transport']
del grp_1['Primary_Variable'] # Total tracer + age tracer
grp_1.create_dataset("Primary_Variable", data=dset_2)


# ## replace initial flow

# **get Pressure from age simulation, i.e. the Primary_variable under flow group**

# In[10]:


grp_3 = input_h5_2['Checkpoint/PMCSubsurfaceFlow/flow']
dset_3 = grp_3['Primary_Variables'] # flow only
list(grp_3.keys())


# **replace the original `Primary_Variables` with the new `Primary_Variables` from the age simulation**

# In[11]:


grp_1 = output_h5['Checkpoint/PMCSubsurfaceFlow/flow']
dset_1 = grp_1['Primary_Variables'] # flow (1) + temperature (2)


# In[12]:


# replace old pressure head with new pressure head
dset_1[::2] = dset_3


# In[13]:


del grp_1['Primary_Variables']
grp_1.create_dataset("Primary_Variables", data=dset_1)


# In[14]:


input_h5_1.close()
input_h5_2.close()
output_h5.close()

