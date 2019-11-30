#!/usr/bin/env python
# coding: utf-8

# # Rename files using R

# `mixedsort`: sort the embeded numbers in correct orders so that 1, 2, 3, ..., 10, 11, 12... instead of 1, 10, 11, 12..., 2, ..., 3, ...
# 
# - to execute in batch mode, save this as `rename.R` and on the command line use `Rscript rename.R`

# In[ ]:


get_ipython().run_line_magic('R', '')
library(gtools)

file.rename(mixedsort(list.files(pattern = "resize*")), paste0("stage", 1:366, ".jpg"))


# In[ ]:


file.rename(list.files(pattern = "resize*"), paste0("stage", sprintf("%03d", 1:366), ".jpg"))


# # manipulate figures using `ImageMagic`

# ## resize images
# 1. use `convert -crop` or `convert -resize` to change image size
# 
# 2. or you can use `mogrify` to batch convert

# In[3]:


#!/bin/bash -l

cd "$cwd/age"

for itime in {1..366}
do

    echo $(printf "%04d" $itime)
    ## crop & resize image
    convert age."$(printf "%04d" $(($itime+5)) )".jpg -crop 2000x2000+600+200     -resize 1200 crop.resize.age.trans."$(printf "%04d" $(($itime+5)) )".jpg
    
done

cd "$cwd/tracer"

for itime in {1..366}
do

    echo $(printf "%04d" $itime)
    ## crop & resize image
    convert tracer."$(printf "%04d" $(($itime+5)) )".jpg -crop 2000x2000+600+200     -resize 1200 crop.resize.tracer.trans."$(printf "%04d" $(($itime+5)) )".jpg
    
done


# In[9]:


#!/bin/bash -l

cd "$cwd/tracer_m"

for itime in {1..366}
do

    echo $(printf "%04d" $itime)
    ## crop & resize image
    convert tracer_m."$(printf "%04d" $(($itime+5)) )".jpg -crop 2000x2000+600+200     -resize 800 crop.resize.tracer_m.trans."$(printf "%04d" $(($itime+5)) )".jpg
    
done

cd "$cwd/tracer_n"

for itime in {1..366}
do

    echo $(printf "%04d" $itime)
    ## crop & resize image
    convert tracer_n."$(printf "%04d" $(($itime+5)) )".jpg -crop 2200x2200+500+200     -resize 800 crop.resize.tracer_n.trans."$(printf "%04d" $(($itime+5)) )".jpg
    
done

cd "$cwd/tracer_s"

for itime in {1..366}
do

    echo $(printf "%04d" $itime)
    ## crop & resize image
    convert tracer_s."$(printf "%04d" $(($itime+5)) )".jpg -crop 1200x2400+1000+10     -resize 800 crop.resize.tracer_s.trans."$(printf "%04d" $(($itime+5)) )".jpg
    
done


# ## generate .gif animation
# 
# use `convert *.jpg .gif` to generate gif animation

# In[10]:


#!/bin/bash -l

cd "$cwd/age"

convert -delay 1 -loop 0 -quality 75 crop.resize.age.trans.*.jpg age.trans.2011.2015.gif

cd "$cwd/tracer"

convert -delay 1 -loop 0 -quality 75 crop.resize.tracer.trans.*.jpg tracer.trans.2011.2015.gif

cd "$cwd/tracer_s"

convert -delay 1 -loop 0 -quality 75 crop.resize.tracer_s.trans.*.jpg tracer_s.trans.2011.2015.gif

cd "$cwd/tracer_n"

convert -delay 1 -loop 0 -quality 75 crop.resize.tracer_n.trans.*.jpg tracer_n.trans.2011.2015.gif

cd "$cwd/tracer_m"

convert -delay 1 -loop 0 -quality 75 crop.resize.tracer_m.trans.*.jpg tracer_m.trans.2011.2015.gif


# In[ ]:


#!/bin/bash -l

# cd "$cwd/age"

convert -delay 1 -loop 0 -quality 75 *.jpg stage.2011.2015.gif


# In[ ]:


# convert jpeg to gif
#!/bin/bash -l

# convert -delay 1 -loop 0 -quality 75 /Users/shua784/Dropbox/PNNL/Projects/Reach_scale_model/Outputs/HFR_model_200x200x5/river_stage/resize*.png -resize 30% /Users/shua784/Dropbox/PNNL/Projects/Reach_scale_model/Outputs/HFR_model_200x200x5/river_stage/river_stage_2011_2015.gif

# convert -delay 1 -loop 0 -quality 75 /Users/shua784/Dropbox/PNNL/Projects/Reach_scale_model/Outputs/HFR_model_200x200x5/age/age*.jpg /Users/shua784/Dropbox/PNNL/Projects/Reach_scale_model/Outputs/HFR_model_200x200x5/age/age_2011_2015.gif

convert -delay 1 -loop 0 -quality 50 /Users/shua784/Dropbox/PNNL/Projects/Reach_scale_model/Outputs/HFR_model_200x200x5/combined_tracer_age_level/tracer*.jpg -resize 50% /Users/shua784/Dropbox/PNNL/Projects/Reach_scale_model/Outputs/HFR_model_200x200x5/combined_tracer_age_level/tracer_age_level_2011_2015.gif


# ## generate mpeg movie

# **common video format: mp4, mpg, avi.**

# In[ ]:


convert -delay 10 -quality 75 *.jpg wl_stage.gif


# ## combine figures
# use `convert -append` (top to bottom ) or `convert +append` (left to right) to combine figures

# In[7]:


# combine two figures
#!/bin/bash -l

cd /Users/shua784/Dropbox/PNNL/Projects/Reach_scale_model/Outputs/HFR_model_200x200x5

for itime in {1..366} ## timesteps from 2011-1-1 to 2015-12-31

do
	echo $(printf "%04d" $itime)

	convert /tracer/tracer."$(printf "%04d" $(($itime+10)) )".jpg /age/age."$(printf "%04d" $(($itime+10)) )".jpg     +append combined_tracer_age/tracer_age."$itime".jpg 

	convert river_stage/sorted/stage"$itime".jpg -resize 200% -gravity Center combined_tracer_age/tracer_age."$itime".jpg     -append combined_tracer_age_level/tracer_age_level."$itime".jpg" 

## convert "/Users/shua784/Dropbox/PNNL/Projects/Reach_scale_model/Outputs/river_stage/sorted/stage_"$itime".jpg" -resize 50% -gravity Center "/Users/shua784/Dropbox/PNNL/Projects/Reach_scale_model/Outputs/test_2007_age/GW_age/sorted/age."$(printf "%04d" $(($itime-1)) )".jpg" -append "/Users/shua784/Dropbox/PNNL/Projects/Reach_scale_model/Outputs/test_2007_age/combined_age_level/age_level."$itime".jpg" 

## convert "/Users/shua784/Dropbox/PNNL/Projects/Reach_scale_model/Outputs/river_stage/sorted/stage_"$itime".jpg" -resize 50% -gravity Center "/Users/shua784/Dropbox/PNNL/Projects/Reach_scale_model/Outputs/test_2007_age/tracer_plume/sorted/tracer."$(printf "%04d" $(($itime-1)) )".jpg" -append "/Users/shua784/Dropbox/PNNL/Projects/Reach_scale_model/Outputs/test_2007_age/combined_tracer_level/tracer_level."$itime".jpg" 

 
done

