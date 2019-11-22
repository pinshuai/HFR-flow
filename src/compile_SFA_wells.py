#!/usr/bin/env python
# coding: utf-8

# # compile separete csv files into one file with wellnames added

# In[63]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150


# In[61]:


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from glob import glob
from natsort import natsorted
import re
from tqdm.notebook import trange, tqdm


# In[67]:


fname_wells = "/Users/shua784/Dropbox/PNNL/People/Velo/300A_Well_Data_2008-2019/"
fname_well_coord = "/Users/shua784/Dropbox/PNNL/People/Velo/300A_well_coordinates_all.csv"
fname_well_all = fname_wells + "all_wells_2017-2018.csv"


# In[68]:


years= [str(i) for i in np.arange(2017, 2019)]


# In[70]:


df = pd.DataFrame(columns=(['DateTime', 'Temp', 'SpC', 'WL', 'WellName']))

for iyear in tqdm(years[:]):
    files = glob(fname_wells + iyear + '/*.csv')
    files = natsorted(files)

    # ifile = files[0]
    for ifile in files[:]:
        idf = pd.read_csv(ifile, header = 0, names=['DateTime', 'Temp', 'SpC', 'WL'])
        iname = re.sub('_3var.csv', '',re.split('/', ifile)[-1])
        idf['WellName'] = np.repeat(np.array(iname), idf.shape[0], axis=0)

        df = pd.concat([df, idf])

df.to_csv(fname_well_all, index=False)

