
setwd("/Users/shua784/Dropbox/PNNL/Projects/Reach_scale_model/")

rm(list=ls())

library(fields)
library(AtmRay) 
library(maptools)
library(raster)
library(plot3D)
##------------------INPUT-------------------------------##
fname_hanford = "data/geoFramework/hanford.asc"
fname_basalt= "data/geoFramework/top_of_basalt.asc"
fname_ringold_a= "data/geoFramework/ringold_a.asc"
fname_ringold_e= "data/geoFramework/ringold_e.asc"
fname_ringold_lm= "data/geoFramework/ringold_lm.asc"
fname_cold_creek= "data/geoFramework/cold_creek.asc"
fname_taylor_flats= "data/geoFramework/taylor_flats.asc"

fname_river_bath = "data/geoFramework/river_bathymetry_20m.asc"

# fname_test = "data/PascoEast_TopArray.asc"

plot.fig = F


##---------------------OUTPUT---------------------------##
fname_fig.basalt2d = "figures/basalt2d.jpg"
fname_fig.hanford2d = "figures/hanford2d.jpg"
fname_fig.basalt2d.model = "figures/basalt2d_model_250m.jpg"
fname_fig.hanford2d.model = "figures/hanford2d_model_250m.jpg"
fname_fig.ringold_a_2d = "figures/ringold_a_2d.jpg"
fname_fig.ringold_a_2d.model = "figures/ringold_a_2d_model_250m.jpg"
fname_fig.ringold_e_2d = "figures/ringold_e_2d.jpg"
fname_fig.ringold_e_2d.model = "figures/ringold_e_2d_model_250m.jpg"
fname_fig.ringold_lm_2d = "figures/ringold_lm_2d.jpg"
fname_fig.ringold_lm_2d.model = "figures/ringold_lm_2d_model_250m.jpg"
fname_fig.cold_creek_2d = "figures/cold_creek_2d.jpg"
fname_fig.cold_creek_2d.model = "figures/cold_creek_2d_model_250m.jpg"
fname_fig.taylor_flats_2d = "figures/taylor_flats_2d.jpg"
fname_fig.taylor_flats_2d.model = "figures/taylor_flats_2d_model_250m.jpg"
fname_fig.river_bath_2d = "figures/river_bath_2d.jpg"
fname_fig.river_bath_2d.model = "figures/river_bath_2d_model_250m.jpg"
fname_geoFramework.r = "results/geoframework_200x200x5.r"
fname_ascii.r = "results/ascii.r"
fname_model_inputs.r = "results/model_inputs_200x200x5.r"
##---------------------FUNCTIONS---------------------------------##
proj_to_model <- function(origin,angle,coord)
{
  output = array(NA,dim(coord))
  rownames(output) = rownames(coord)
  colnames(output) = colnames(coord)    
  output[,1] = (coord[,1]-origin[1])*cos(angle)+(coord[,2]-origin[2])*sin(angle)
  output[,2] = (coord[,2]-origin[2])*cos(angle)-(coord[,1]-origin[1])*sin(angle)
  return(output)
}   


model_to_proj <- function(origin,angle,coord)
{
  output = array(NA,dim(coord))
  rownames(output) = rownames(coord)
  colnames(output) = colnames(coord)    
  output[,1] = origin[1]+coord[,1]*cos(angle)-coord[,2]*sin(angle)
  output[,2] = origin[2]+coord[,1]*sin(angle)+coord[,2]*cos(angle)
  return(output)
}   

##---------------------Parameters-----------------------------------##
angle = 0

#hanford reach
# model_origin = c(538000, 97000)
model_origin = c(551600, 104500)

xlen = 60*1000 #x boundary length
ylen = 60*1000 #y boundary length

zlen = 200 #z boundary length
# zlen = 100 #z boundary length

# model origin
z0 = 0 
x0 = model_origin[1]
y0 = model_origin[2]

##==================== grid cell size===========================
idx = 200
idy = 200
idz = 5

##----------------------------------------------------------------------------
dx = rep(idx, xlen/idx)#dx=4 m
# dx = c(rev(round(0.5*1.09919838^seq(1,44),3)),rep(0.5,(400-350)/0.5))#refine mesh?? need to modify?
dy = rep(idy, ylen/idy)#dy=4 m
# dz = c(rev(round(0.1*1.09505^seq(1,25),3)),rep(0.1,(108-100)/0.1),round(0.1*1.097^seq(1,11),3))
dz = rep(idz, zlen/idz)

nx = length(dx)
ny = length(dy)
nz = length(dz)

#create x,y,z coordinates for each cell center
x = cumsum(dx)-0.5*dx
y = cumsum(dy)-0.5*dy
# z = 90+cumsum(dz)-0.5*dz
z = z0 + cumsum(dz)-0.5*dz

#min and max x,y,z coord
range_x = c(x[1]-0.5*dx[1], x[length(x)]+0.5*dx[length(x)])
range_y = c(y[1]-0.5*dy[1], y[length(y)]+0.5*dy[length(y)])
range_z = c(z[1]-0.5*dz[1], z[length(z)]+0.5*dz[length(z)])

##dimension of model domain in CRS
west_x = x0
east_x = x0 + xlen
south_y = y0
north_y = y0 + ylen

## use "maptools" package to read ASCII file
# grid <- readAsciiGrid(fname_test, as.image = F) # return an object of class
# df <- data.frame(readAsciiGrid(fname_test))
# names(df)=c("z", "x", "y")
# image(grid)
# summary(grid)
# 
# grid2 <- readAsciiGrid(fname_test, as.image = T) # return a list, and keep NAs

# ## use "raster" package to read ASCII file
# r = raster(fname_test)
# plot(r)

# save(list = c("model_origin", "x", "y", "z", "nx", "ny", "nz", "idx", "idy", "idz", "x0", "y0", "z0", "xlen", "ylen", "zlen"),
#      file = fname_model_inputs.r)
save(list = ls(), file = fname_model_inputs.r)
#======================= load ASCII file and interpolate to model domain ============================


if (!file.exists(fname_ascii.r)) {
  
hanford_data = readAsciiGrid(fname_hanford, as.image = T)
basalt_data = readAsciiGrid(fname_basalt, as.image = T)
cold_creek_data = readAsciiGrid(fname_cold_creek, as.image = T)
taylor_flats_data = readAsciiGrid(fname_taylor_flats, as.image = T)
ringold_e_data = readAsciiGrid(fname_ringold_e, as.image = T)
ringold_lm_data = readAsciiGrid(fname_ringold_lm, as.image = T)
ringold_a_data = readAsciiGrid(fname_ringold_a, as.image = T)
river_bath_data = readAsciiGrid(fname_river_bath, as.image = T)

save(list = c("hanford_data", "basalt_data", "cold_creek_data", "taylor_flats_data", "ringold_e_data", "ringold_lm_data", "ringold_a_data", "river_bath_data"), file = fname_ascii.r)


} else {
  load(fname_ascii.r) ## load into stored geologic data
}
  
  
  
cells_model = expand.grid(x,y) # expand grid to include all x-y coordinates
cells_proj = model_to_proj(model_origin,angle,cells_model) # convert model coord. to proj. coord.

unit_x = sort(as.numeric(names(table(cells_proj[, 1]))))
unit_y = sort(as.numeric(names(table(cells_proj[, 2]))))

cells_hanford = interp.surface(hanford_data, cells_proj) # map surface to model grids
cells_basalt = interp.surface(basalt_data, cells_proj) # map surface to model grids
cells_cold_creek = interp.surface(cold_creek_data, cells_proj)
cells_taylor_flats = interp.surface(taylor_flats_data, cells_proj)
cells_ringold_e = interp.surface(ringold_e_data, cells_proj)
cells_ringold_lm = interp.surface(ringold_lm_data, cells_proj)
cells_ringold_a = interp.surface(ringold_a_data, cells_proj)
cells_river_bath = interp.surface(river_bath_data, cells_proj)

cells_hanford = array(cells_hanford, c(nx, ny))
cells_basalt = array(cells_basalt, c(nx, ny))
cells_cold_creek = array(cells_cold_creek, c(nx, ny))
cells_taylor_flats = array(cells_taylor_flats, c(nx, ny))
cells_ringold_e = array(cells_ringold_e, c(nx, ny))
cells_ringold_lm = array(cells_ringold_lm, c(nx, ny))
cells_ringold_a = array(cells_ringold_a, c(nx, ny))
cells_river_bath = array(cells_river_bath, c(nx, ny))

save(list=ls(), file=fname_geoFramework.r) 

#=================================plot each geologic unit====================================
if (plot.fig) {
  
  
  # open3d()
  # bg3d("white")
  # nbcol = 100
  # color = rev(rainbow(nbcol, start = 0/6, end = 4/6))
  # zcol  = cut(cells_basalt, nbcol)
  
  # persp3d(unit_x, unit_y, cells_basalt, col = color[zcol], aspect = c(1,1,0.2), box= F)
  # persp3d(unit_x, unit_y, cells_hanford, col = color[zcol], aspect = c(1,1,0.2), add = TRUE)
  # persp3d(unit_x, unit_y, cells_hanford, col = "red", aspect = c(1,1,1), add = TRUE)
  
  # persp3d(unit_x, unit_y, cells_basalt, col = "blue", aspect = c(1,1,1), box= F)
  # persp3d(unit_x, unit_y, cells_ringold_lm, col = "red", aspect = c(1,1,1), add = TRUE)
  # persp3d(unit_x, unit_y, cells_ringold_a, col = "green", aspect = c(1,1,1), add = TRUE)
  # persp3d(unit_x, unit_y, cells_ringold_e, col = "yellow", aspect = c(1,1,1), add = TRUE, alpha=0.5)
  # persp3d(unit_x, unit_y, cells_taylor_flats, col = "purple", aspect = c(1,1,1), add = TRUE, alpha=0.5)
  # persp3d(unit_x, unit_y, cells_cold_creek, col = "cyan", aspect = c(1,1,1), add = TRUE, alpha=0.5)
  
  
##---------------------------- plot hanford ------------------------- 
## crop dataset to model region
x_model = which(hanford_data$x >= west_x & hanford_data$x <= east_x)
y_model = which(hanford_data$y >= south_y & hanford_data$y <= north_y)

hanford_data_model = list()
hanford_data_model = list(x = hanford_data$x[x_model], y = hanford_data$y[y_model], z = hanford_data$z[x_model, y_model])


jpeg(fname_fig.hanford2d, width=8,height=8,units='in',res=300,quality=100)
image2D(hanford_data_model, shade=0.2, rasterImage = F, NAcol = "white",
        main = c("DEM_30m"), asp = 1)
dev.off()

jpeg(fname_fig.hanford2d.model, width=8,height=8,units='in',res=300,quality=100)
image2D(z= cells_hanford, x= unit_x, y= unit_y, shade=0.5, rasterImage = F, NAcol = "white", border = NA, resfac = 3,
        main = c("DEM_150m"), asp = 1)
dev.off()

## show the perspective view of the surface plot
# open3d()
# bg3d("white")
# nbcol = 100
# color = rev(rainbow(nbcol, start = 0/6, end = 4/6))
# zcol  = cut(hanford_data_model$z, nbcol)
# persp3d(hanford_data_model$x, hanford_data_model$y, hanford_data_model$z, col = color[zcol], aspect = c(1,1,0.2))
# 


##---------------------------- plot basalt ------------------------- 
## crop dataset to model region
x_model = which(basalt_data$x >= west_x & basalt_data$x <= east_x)
y_model = which(basalt_data$y >= south_y & basalt_data$y <= north_y)

basalt_data_model = list()
basalt_data_model = list(x = basalt_data$x[x_model], y = basalt_data$y[y_model], z = basalt_data$z[x_model, y_model])

jpeg(fname_fig.basalt2d, width=8,height=8,units='in',res=300,quality=100)
image2D(basalt_data_model, shade=0.2, rasterImage = F, NAcol = "white",
        main = c("basalt_30m"), asp = 1)
dev.off()

jpeg(fname_fig.basalt2d.model, width=8,height=8,units='in',res=300,quality=100)
image2D(z= cells_basalt, x= unit_x, y= unit_y, shade=0.2, rasterImage = F, NAcol = "white",
        main = c("basalt_100m"), asp = 1)
dev.off()

##---------------------------- plot ringold ------------------------- 
## crop dataset to model region
x_model = which(ringold_a_data$x >= west_x & ringold_a_data$x <= east_x)
y_model = which(ringold_a_data$y >= south_y & ringold_a_data$y <= north_y)

ringold_a_data_model = list()
ringold_a_data_model = list(x = ringold_a_data$x[x_model], y = ringold_a_data$y[y_model], z = ringold_a_data$z[x_model, y_model])

jpeg(fname_fig.ringold_a_2d, width=8,height=8,units='in',res=300,quality=100)
image2D(ringold_a_data_model, shade=0.2, rasterImage = F, NAcol = "white",
        main = c("ringold_a_30m"), asp = 1)
dev.off()

jpeg(fname_fig.ringold_a_2d.model, width=8,height=8,units='in',res=300,quality=100)
image2D(z= cells_ringold_a, x= unit_x, y= unit_y, shade=0.2, rasterImage = F, NAcol = "white",
        main = c("ringold_a_100m"), asp = 1)
dev.off()

##
x_model = which(ringold_e_data$x >= west_x & ringold_e_data$x <= east_x)
y_model = which(ringold_e_data$y >= south_y & ringold_e_data$y <= north_y)

ringold_e_data_model = list()
ringold_e_data_model = list(x = ringold_e_data$x[x_model], y = ringold_e_data$y[y_model], z = ringold_e_data$z[x_model, y_model])

jpeg(fname_fig.ringold_e_2d, width=8,height=8,units='in',res=300,quality=100)
image2D(ringold_e_data_model, shade=0.2, rasterImage = F, NAcol = "white",
        main = c("ringold_e_30m"), asp = 1)
dev.off()

jpeg(fname_fig.ringold_e_2d.model, width=8,height=8,units='in',res=300,quality=100)
image2D(z= cells_ringold_e, x= unit_x, y= unit_y, shade=0.2, rasterImage = F, NAcol = "white",
        main = c("ringold_e_100m"), asp = 1)
dev.off()

##
x_model = which(ringold_lm_data$x >= west_x & ringold_lm_data$x <= east_x)
y_model = which(ringold_lm_data$y >= south_y & ringold_lm_data$y <= north_y)

ringold_lm_data_model = list()
ringold_lm_data_model = list(x = ringold_lm_data$x[x_model], y = ringold_lm_data$y[y_model], z = ringold_lm_data$z[x_model, y_model])

jpeg(fname_fig.ringold_lm_2d, width=8,height=8,units='in',res=300,quality=100)
image2D(ringold_lm_data_model, shade=0.2, rasterImage = F, NAcol = "white",
        main = c("ringold_lm_30m"), asp = 1)
dev.off()

jpeg(fname_fig.ringold_lm_2d.model, width=8,height=8,units='in',res=300,quality=100)
image2D(z= cells_ringold_lm, x= unit_x, y= unit_y, shade=0.2, rasterImage = F, NAcol = "white",
        main = c("ringold_lm_100m"), asp = 1)
dev.off()


##---------------------------- plot cold creek ------------------------- 
x_model = which(cold_creek_data$x >= west_x & cold_creek_data$x <= east_x)
y_model = which(cold_creek_data$y >= south_y & cold_creek_data$y <= north_y)

cold_creek_data_model = list()
cold_creek_data_model = list(x = cold_creek_data$x[x_model], y = cold_creek_data$y[y_model], z = cold_creek_data$z[x_model, y_model])

jpeg(fname_fig.cold_creek_2d, width=8,height=8,units='in',res=300,quality=100)
image2D(cold_creek_data_model, shade=0.2, rasterImage = F, NAcol = "white",
        main = c("cold_creek_30m"), asp = 1)
dev.off()

jpeg(fname_fig.cold_creek_2d.model, width=8,height=8,units='in',res=300,quality=100)
image2D(z= cells_cold_creek, x= unit_x, y= unit_y, shade=0.2, rasterImage = F, NAcol = "white",
        main = c("cold_creek_100m"), asp = 1)
dev.off()

##---------------------------- plot taylor flats ------------------------- 
x_model = which(taylor_flats_data$x >= west_x & taylor_flats_data$x <= east_x)
y_model = which(taylor_flats_data$y >= south_y & taylor_flats_data$y <= north_y)

taylor_flats_data_model = list()
taylor_flats_data_model = list(x = taylor_flats_data$x[x_model], y = taylor_flats_data$y[y_model], z = taylor_flats_data$z[x_model, y_model])

jpeg(fname_fig.taylor_flats_2d, width=8,height=8,units='in',res=300,quality=100)
image2D(taylor_flats_data_model, shade=0.2, rasterImage = F, NAcol = "white",
        main = c("taylor_flats_30m"), asp = 1)
dev.off()

jpeg(fname_fig.taylor_flats_2d.model, width=8,height=8,units='in',res=300,quality=100)
image2D(z= cells_taylor_flats, x= unit_x, y= unit_y, shade=0.2, rasterImage = F, NAcol = "white",
        main = c("taylor_flats_100m"), asp = 1)
dev.off()

##---------------------------- plot river bathymetry ------------------------- 
x_model = which(river_bath_data$x >= west_x & river_bath_data$x <= east_x)
y_model = which(river_bath_data$y >= south_y & river_bath_data$y <= north_y)

river_bath_data_model = list()
river_bath_data_model = list(x = river_bath_data$x[x_model], y = river_bath_data$y[y_model], z = river_bath_data$z[x_model, y_model])

jpeg(fname_fig.river_bath_2d, width=8,height=8,units='in',res=300,quality=100)
image2D(river_bath_data_model, shade=0.2, rasterImage = F, NAcol = "white",
        main = c("river_bath_20m"), asp = 1)
dev.off()

jpeg(fname_fig.river_bath_2d.model, width=8,height=8,units='in',res=300,quality=100)
image2D(z= cells_river_bath, x= unit_x, y= unit_y, shade=0.2, rasterImage = F, NAcol = "white",
        main = c("river_bath_100m"), asp = 1)
dev.off()

}




