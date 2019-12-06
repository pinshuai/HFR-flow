#!/usr/bin/env python
# coding: utf-8

# # upload model inputs and submit jobs to NERSC

# ## upload files to NERSC

# In[ ]:


scp -r ~/Dropbox/PNNL/Projects/Reach_scale_model/Inputs/HFR_model_200x200x2_steady_state/* pshuai@cori.nersc.gov:/global/cscratch1/sd/pshuai/HFR_model_200x200x2_steady_state/.


# ## login to NERSC

# In[ ]:


ssh pshuai@edison.nersc.gov


# ## submit batch job

# In[5]:


get_ipython().run_cell_magic('bash', '', 'ls')


# In[ ]:


#!/bin/bash -l

#SBATCH -A m1800
#SBATCH -N 86
#SBATCH -t 48:00:00
#SBATCH -L SCRATCH  
#SBATCH -J test_0413
#SBATCH --qos regular
#SBATCH --mail-type ALL
#SBATCH --mail-user pin.shuai@pnnl.gov

cd $SLURM_SUBMIT_DIR

srun -n 2048 ../pflotran-edison-flux -pflotranin pflotran_200x200x5_6h_bc_test.in


# # run interactive job

# In[ ]:


## on Edison

salloc -N 1 -q debug -L SCRATCH -t 00:30:00

srun -n 24 ../pflotran-edison -pflotranin pflotran_200x200x2_head_bc.in

salloc -N 2 -q debug -L SCRATCH -t 00:30:00

srun -n 48 ../pflotran-edison -pflotranin pflotran_200x200x2_head_bc.in


# In[ ]:


## on Cori

salloc -N 1 -q debug -C haswell -t 00:30:00 -L SCRATCH

srun -n 32 ../pflotran-cori -pflotranin pflotran_200x200x2_6h_bc_obs.in


# # download files from NERSC

# In[ ]:


scp -r pshuai@cori.nersc.gov:/global/cscratch1/sd/pshuai/HFR_model_100x100x2/* ~/Dropbox/PNNL/Projects/Reach_scale_model/Outputs/HFR_model_100x100x2/.


# In[ ]:


## download from dtn.pnl.gov
scp -r shua784@dtn2.pnl.gov:/pic/dtn2/shua784/HFR_model_100x100x5/pflotran_100x100x5.h5 ~/Paraview/HFR/HFR_100x100x5m/.

