## generate input deck for PFLOTRAN

## use "Run" instead of "Source" to execute the codes!!

setwd("/Users/shua784/Dropbox/PNNL/Projects/Reach_scale_model/")


## -----------INPUTS-----------------##
# fname_WellScreen = "data/Monitoring_Well_Screen_bigdomain.csv"
fname_material.h5 = c("HFR_material_river.h5")
# fname_H.BC.h5 = c("300A_H_BC.h5")
fname_H.initial.h5 = c("HFR_H_Initial.h5")
fname_model_inputs.r = "results/model_inputs_200x200x5.r"
fname_bc_dir = "bc_6h_smooth/"

fname.DatumH = c("DatumH_Mass1_")
fname.Gradient = c("Gradients_Mass1_")
#-----------------OUTPUT--------------------------#
fname_pflotran.in = "Inputs/HFR_model_200x200x5/pflotran_200x200x5_6h_bc.in"
# fname_well_obs_region = "data/well_obs_region.txt"


load(fname_model_inputs.r)
#-------------PARAMETERIZATION------------------#
# start.time = as.POSIXct("2005-03-29 12:00:00",tz="GMT",format="%Y-%m-%d %H:%M:%S")
start.time = as.POSIXct("2007-03-28 12:00:00",tz="GMT",format="%Y-%m-%d %H:%M:%S")
end.time = as.POSIXct("2015-12-31 00:00:00",tz="GMT",format="%Y-%m-%d %H:%M:%S")

grid.n_pts = c(nx, ny, nz)
grid.d_pts = c(idx,  idy,  idz)


# river conductance
# upstr_river_cond = 5.23e-14
# north_river_cond = 6.31e-14
# middle_river_cond = 1.47e-13
# south_river_cond = 1.59e-12

river_cond = 4.65e-13 # from mean(rand_2) conductance

# # project origin
# x0 = 593000
# y0 = 114500
# z0 = 88
model_domain_origin = c(0,0,z0)

a = 0 #rotation
wall.stop = 19.8 # wallclock stop time, h
check.pt = 720 # check point, h

# solver
ts.acceleration = 8
max.ts.cuts = 50

diff.coeff = 1e-9 # diffusion coefficient
# recharge = 1.757e-9 #recharge rate 5.54 cm/yr from Fayer and Walters (1995)

##----------------------- material -------------------------------##
material.list = c("hanford", "cold_creek", "taylor_flats", "ringold_e", "ringold_lm", "ringold_a")
# material properties
Khf = 7000 #m/d from 300A model, William's 2008
hanford = list(ID = 1, porosity = 0.2, tortuosity = 1, perm.x = Khf/(9.5e11), perm.y = Khf/(9.5e11), perm.z = Khf/(9.5e12), cc = "cc1")
# Kcc = 4.8 #m/d from Last et al
Kcc = 100 #m/d from Paul's report
cold_creek = list(ID = 2, porosity = 0.25, tortuosity = 1, perm.x = Kcc/(9.5e11), perm.y = Kcc/(9.5e11), perm.z = Kcc/(9.5e12), cc = "cc1")
Ktf = 1 # m/d from franklin county report
taylor_flats = list(ID = 3, porosity = 0.43, tortuosity = 1, perm.x = Ktf/(9.5e11), perm.y = Ktf/(9.5e11), perm.z = Ktf/(9.5e12), cc = "cc2")
Kre = 40 # m/d from Paul's report
ringold_e = list(ID = 4, porosity = 0.25, tortuosity = 1, perm.x = Kre/(9.5e11), perm.y = Kre/(9.5e11), perm.z = Kre/(9.5e12), cc = "cc1")
Krlm = 1 #m/d from franklin
ringold_lm = list(ID = 5, porosity = 0.43, tortuosity = 1, perm.x = Krlm/(9.5e11), perm.y = Krlm/(9.5e11), perm.z = Krlm/(9.5e12), cc = "cc2")
Kra = 1 # similar to krlm
ringold_a = list(ID = 6, porosity = 0.43, tortuosity = 1, perm.x = Kra/(9.5e11), perm.y = Kra/(9.5e11), perm.z = Kra/(9.5e12), cc = "cc2")

# Khf = 7000 #m/d from 300A model, William's 2008
# hanford = list(ID = 1, porosity = 0.2, tortuosity = 1, perm.x = Khf/(9.5e11), perm.y = Khf/(9.5e11), perm.z = Khf/(9.5e12), cc = "cc1")
# # Kcc = 4.8 #m/d from Last et al
# Kcc = Khf #m/d from Paul's report
# cold_creek = list(ID = 2, porosity = 0.2, tortuosity = 1, perm.x = Kcc/(9.5e11), perm.y = Kcc/(9.5e11), perm.z = Kcc/(9.5e12), cc = "cc1")
# Ktf = Khf # m/d from franklin county report
# taylor_flats = list(ID = 3, porosity = 0.2, tortuosity = 1, perm.x = Ktf/(9.5e11), perm.y = Ktf/(9.5e11), perm.z = Ktf/(9.5e12), cc = "cc1")
# Kre = Khf # m/d from Paul's report
# ringold_e = list(ID = 4, porosity = 0.2, tortuosity = 1, perm.x = Kre/(9.5e11), perm.y = Kre/(9.5e11), perm.z = Kre/(9.5e12), cc = "cc1")
# Krlm = Khf #m/d from franklin
# ringold_lm = list(ID = 5, porosity = 0.2, tortuosity = 1, perm.x = Krlm/(9.5e11), perm.y = Krlm/(9.5e11), perm.z = Krlm/(9.5e12), cc = "cc1")
# Kra = Khf # similar to krlm
# ringold_a = list(ID = 6, porosity = 0.2, tortuosity = 1, perm.x = Kra/(9.5e11), perm.y = Kra/(9.5e11), perm.z = Kra/(9.5e12), cc = "cc1")

# Khf = 7000 #m/d from 300A model, William's 2008
# hanford = list(ID = 1, porosity = 0.2, tortuosity = 1, perm.x = Khf/(9.5e11), perm.y = Khf/(9.5e11), perm.z = Khf/(9.5e12), cc = "cc1")
# # Kcc = 4.8 #m/d from Last et al
# Kcc = 190 #m/d from Paul's report
# cold_creek = list(ID = 2, porosity = 0.2, tortuosity = 1, perm.x = Kcc/(9.5e11), perm.y = Kcc/(9.5e11), perm.z = Kcc/(9.5e12), cc = "cc2")
# Ktf = 7.5 # m/d from franklin county report
# taylor_flats = list(ID = 3, porosity = 0.3, tortuosity = 1, perm.x = Ktf/(9.5e11), perm.y = Ktf/(9.5e11), perm.z = Ktf/(9.5e12), cc = "cc2")
# Kre = 3817 # m/d from Paul's report
# ringold_e = list(ID = 4, porosity = 0.2, tortuosity = 1, perm.x = Kre/(9.5e11), perm.y = Kre/(9.5e11), perm.z = Kre/(9.5e12), cc = "cc1")
# Krlm = 13.8 #m/d from franklin
# ringold_lm = list(ID = 5, porosity = 0.21, tortuosity = 1, perm.x = Krlm/(9.5e11), perm.y = Krlm/(9.5e11), perm.z = Krlm/(9.5e12), cc = "cc2")
# Kra = 13.8 # similar to krlm
# ringold_a = list(ID = 6, porosity = 0.3, tortuosity = 1, perm.x = Kra/(9.5e11), perm.y = Kra/(9.5e11), perm.z = Kra/(9.5e12), cc = "cc2")



K.list = list(hanford = hanford, cold_creek = cold_creek, taylor_flats = taylor_flats, ringold_e = ringold_e, ringold_lm = ringold_lm, ringold_a = ringold_a)
# saturation functions
cc.names = c("cc1", "cc2")

cc1 = list(M = 0.3391, alpha = 7.2727e-4, rsat = 0.16, max.cp = 1e8)
cc2 = list(M = 0.7479, alpha = 1.4319e-4, rsat = 0.1299, max.cp = 1e8)

cc.list = list(cc1 = cc1, cc2 = cc2)

# output opitons
output.ts = 120 # hour
h5.files = 73 # number of hdf5 files to write
obs.ts = 6 # hour

# --------------------------times----------------------------------

time.index = seq(from=start.time,to=end.time,by="1 hour")
ntime = length(time.index)
nhours = ntime # from 2010-02-27 12:00:00 to 2017-07-01 00:00:00
ini.ts = 0.01 #hour
max.ts = 6 #hour

##-----------------------flow region------------------------
# slice.list = c("315","316","317","318","319","bank_1", "bank_2", "320", "bank_3",
#                "321","bank_4", "322","bank_5", "323","bank_6", "324","bank_7", "325", "bank_8", "bank_9", "326", "327",
#                "328", "329")
slice.list = as.character(seq(40, 332, 1))
river_region = paste("Mass1_", slice.list, sep = "")
flow_region = paste("Flow_Mass1_", slice.list, sep = "")
names(river_region)  = flow_region
names(slice.list) = flow_region
## ------------------- add solute tracer----------------------------
solute.list = c("Solute_river")

## condition coupler, each flow region corresponds to one solute
solute.coupler.list = c(rep(solute.list, length(flow_region)))
names(solute.coupler.list)= flow_region

## concentration list should have same number as of solute list
concentration.list = c("Concentration_river")
names(concentration.list) = solute.list

## tracer list should has same num. of solute list
tracer.list = c("Tracer")
names(tracer.list) = concentration.list

## conductance assign same num. as of flow region
cond.list = c(rep(river_cond, length(flow_region)))

names(cond.list) = flow_region


## create river flow bc list

fname.river_datum = paste(fname.DatumH, slice.list, ".txt", sep = "")
fname.river_gradient = paste(fname.Gradient, slice.list, ".txt", sep = "")
names(fname.river_datum)= flow_region
names(fname.river_gradient)= flow_region

# datasets
flow.BC = c("Flow_West", "Flow_South", "Flow_North", "Flow_East")
BC.all = c("BC_East", "BC_West", "BC_South", "BC_North")
BC.list = c("BC_West", "BC_South", "BC_North", "BC_East")
BC.face = c("West", "South", "North", "East")
names(flow.BC) = BC.face
names(BC.list) = flow.BC
max.buff = 200



# model corner coordinates
xrange = c(0, xlen)
yrange = c(0, ylen)
zrange = c(z0, z0+zlen)

# # injection well coordinates
# inj_wellname = c("Well_399-1-23")


##--------------- generate well obs input deck in Pflotran-----------------
# #read well screen info
# Wells = read.csv(fname_WellScreen, header = TRUE, stringsAsFactors=FALSE)
# Wells[,1] = paste("Well_",Wells[,1],sep="")
# rownames(Wells) = Wells[,1]
# names(Wells) = c("wellname", "x", "y", "elev", "screen_top", "screen_bot")
# Well = as.matrix(Wells[,2:6])
# 
# # x=Wells$x
# x=Well[,1]
# # x=Well
# y=Well[,2]
# elev=Well[,3]
# screen_top=Well[,4]
# screen_bot=Well[,5]
# 
# # convert project coord to model coord, and rotate coordinates a-deg
# x = x-x0
# y = y-y0
# xx=x*cos(a)+y*sin(a)
# yy=y*cos(a)-x*sin(a)
# data=c(xx,yy)
# 
# nwell = length(x)
# cells_z = seq((z0+0.5*grid.d_pts[3]), (z0+grid.n_pts[3]*grid.d_pts[3]), grid.d_pts[3])
# n_obs = nwell
# 
# well.list = Wells$wellname
# well.list = c("Well_399-1-10A", "Well_399-1-21A", "Well_399-2-1", "Well_399-2-2","Well_399-2-3", "Well_399-2-32", "Well_399-4-9")

## ===============BEGINNING of FILE=========================
sink(fname_pflotran.in)
cat("\n")
cat("SIMULATION")
cat("\n")
cat("  SIMULATION_TYPE SUBSURFACE")
cat("\n")
cat("  PROCESS_MODELS")
cat("\n")
cat("  SUBSURFACE_FLOW flow")
cat("\n")
cat("  MODE RICHARDS")
cat("\n")
cat("  /")
cat("\n")
cat("  SUBSURFACE_TRANSPORT transport")
cat("\n")
cat("  GLOBAL_IMPLICIT")
cat("\n")
cat("  /")
cat("\n")
cat("/")
cat("\n")
cat("\n")
cat("\n")

cat("CHECKPOINT")
cat("\n")
cat(paste("  PERIODIC TIME"), check.pt, "h")
cat("\n")
cat("/")
cat("\n")
cat("\n")

cat("#  RESTART pflotran_bigplume-restart.chk")
cat("\n")
cat("END")
cat("\n")
cat("\n")

cat("SUBSURFACE")
cat("\n")
cat(paste("WALLCLOCK_STOP"), wall.stop, "h")
cat("\n")
cat("\n")
cat('#=======================chemistry========================')
cat("\n")
cat("CHEMISTRY")
cat("\n")
cat("  PRIMARY_SPECIES")
cat("\n")
for (itracer in tracer.list) {
  cat(paste("   ", itracer))
  cat("\n")
}
cat("    Tracer_Age")
cat("\n")
cat("  /")
cat("\n")
cat("  DATABASE tracer_HFR.dat")
cat("\n")
cat("  ACTIVITY_COEFFICIENTS OFF")
cat("\n")
cat("  OUTPUT")
cat("\n")
cat("    ALL")
cat("\n")
cat("    AGE")
cat("\n")
cat("    TOTAL")
cat("\n")
cat("/")
cat("\n")
cat("END")
cat("\n")

cat('#=======================solver options========================')
cat("\n")
cat("TIMESTEPPER FLOW")
cat("\n")
cat(paste("  TS_ACCELERATION", ts.acceleration))
cat("\n")
cat(paste("  MAX_TS_CUTS", max.ts.cuts))
cat("\n")
cat("END")
cat("\n")
cat("\n")

cat("\n")
cat("TIMESTEPPER TRANSPORT")
cat("\n")
cat(paste("  TS_ACCELERATION", ts.acceleration))
cat("\n")
cat(paste("  MAX_TS_CUTS", max.ts.cuts))
cat("\n")
cat("END")
cat("\n")
cat("\n")

cat("NEWTON_SOLVER FLOW")
cat("\n")
cat("  MAXIT 20")
cat("\n")
cat("  RTOL 1.d-50")
cat("\n")
cat("  ATOL 1.d-50")
cat("\n")
cat("  STOL 1.d-60")
cat("\n")
cat("  ITOL_UPDATE 1.d0")
cat("\n")
cat("END")
cat("\n")
cat("\n")

cat("LINEAR_SOLVER FLOW")
cat("\n")
cat("END")
cat("\n")
cat("\n")

cat("NEWTON_SOLVER TRANSPORT")
cat("\n")
cat("  NO_INFINITY_NORM")
cat("\n")
cat("END")
cat("\n")
cat("\n")

cat("LINEAR_SOLVER TRANSPORT")
cat("\n")
cat("END")
cat("\n")
cat("\n")
cat('#=======================discretization========================')
cat("\n")
cat("GRID")
cat("\n")
cat("  TYPE structured")
cat("\n")
cat("  NXYZ ")
cat(grid.n_pts)
cat("\n")
cat("  ORIGIN ")
cat(model_domain_origin)
cat("\n")
cat("  DXYZ ")
# cat(grid.d_pts)
cat("\n")
cat("  ")
cat(grid.d_pts[1])
cat("\n")
cat("  ")
cat(grid.d_pts[2])
cat("\n")
cat("  ")
cat(grid.d_pts[3])
cat("\n")
cat("/")
cat("\n")
cat("  BOUNDS")
cat("\n")
cat("  ")
cat(xrange[1], yrange[1], zrange[1])
cat("\n")
cat("  ")
cat(xrange[2], yrange[2], zrange[2])
cat("\n")
cat("/")
cat("\n")
cat("END")
cat("\n")

cat('#=======================fluid properties========================')
cat("\n")
cat("FLUID_PROPERTY")
cat("\n")
cat(paste("  DIFFUSION_COEFFICIENT", diff.coeff))
cat("\n")
cat("END")
cat("\n")
cat('#=======================datasets========================')
cat("\n")

# for (iBC in BC.all) {
#   cat(paste("DATASET", iBC))
#   cat("\n")
#   cat(paste("  FILENAME", fname_H.BC.h5))
#   cat("\n")
#   cat(paste("  HDF5_DATASET_NAME", iBC))
#   cat("\n")
#   cat(paste("  MAX_BUFFER_SIZE", max.buff))
#   cat("\n")
#   cat("END")
#   cat("\n")
#   cat("\n")
# }

cat("DATASET Initial_Head")
cat("\n")
cat(paste("  FILENAME", fname_H.initial.h5))
cat("\n")
cat("  HDF5_DATASET_NAME Initial_Head")
cat("\n")
cat("END")
cat("\n")

cat("\n")
cat('#=======================material properties========================')
cat("\n")
for (imaterial in material.list) {
  
  cat(paste("MATERIAL_PROPERTY", imaterial))
  cat("\n")
  cat(paste("  ID"), K.list[[imaterial]]$ID)
  cat("\n")
  cat(paste("  POROSITY"), K.list[[imaterial]]$porosity)
  cat("\n")
  cat(paste("  TORTUOSITY"), K.list[[imaterial]]$tortuosity)
  cat("\n")
  cat(paste("  CHARACTERISTIC_CURVES", K.list[[imaterial]]$cc))
  cat("\n")
  cat("  PERMEABILITY")
  cat("\n")
  cat(paste("  PERM_X"), K.list[[imaterial]]$perm.x)
  cat("\n")
  cat(paste("  PERM_Y"), K.list[[imaterial]]$perm.y)
  cat("\n")
  cat(paste("  PERM_Z"), K.list[[imaterial]]$perm.z)
  cat("\n")
  cat('  /')
  cat("\n")
  cat('END')
  cat("\n")
  cat("\n")
  
}


cat('#=======================saturation fuctions========================')
cat("\n")
for (icc in cc.names) {
  
  cat(paste("CHARACTERISTIC_CURVES", icc))
  cat("\n")
  cat("  SATURATION_FUNCTION VAN_GENUCHTEN")
  cat("\n")
  cat(paste("    M"), cc.list[[icc]]$M)
  cat("\n")
  cat(paste("    ALPHA"), cc.list[[icc]]$alpha)
  cat("\n")
  cat(paste("    LIQUID_RESIDUAL_SATURATION"), cc.list[[icc]]$rsat)
  cat("\n")
  cat(paste("    MAX_CAPILLARY_PRESSURE"), cc.list[[icc]]$max.cp)
  cat("\n")
  cat('  /')
  cat("\n")
  cat("  PERMEABILITY_FUNCTION BURDINE_VG_LIQ")
  cat("\n")
  cat(paste("    M"), cc.list[[icc]]$M)
  cat("\n")
  cat(paste("    LIQUID_RESIDUAL_SATURATION"), cc.list[[icc]]$rsat)
  cat("\n")
  cat('  /')
  cat("\n")
  cat('END')
  cat("\n")
  cat("\n")
}



cat("\n")
cat('#=======================output options========================')
cat("\n")
cat("OUTPUT")
cat("\n")
cat("  SNAPSHOT_FILE")
cat("\n")
cat("  NO_PRINT_INITIAL")
cat("\n")
cat(paste("#  PERIODIC TIME"), output.ts, "h")
cat("\n")
cat("  PERIODIC TIME 1 w between 4 y and 9 y")
cat("\n")
cat(paste("#  FORMAT HDF5 MULTIPLE_FILES TIMES_PER_FILE"), h5.files)
cat("\n")
cat("  FORMAT HDF5 SINGLE_FILE")
cat("\n")
cat('  /')
cat("\n")
cat("\n")
cat("  MASS_BALANCE_FILE")
cat("\n")
cat("  NO_PRINT_INITIAL")
cat("\n")
cat(paste("#  PERIODIC TIME"), output.ts, "h")
cat("\n")
cat("  PERIODIC TIME 1 w between 4 y and 9 y")
cat("\n")
cat('  /')
cat("\n")
cat("\n")
cat("#  OBSERVATION_FILE")
cat("\n")
cat(paste("#  PERIODIC TIME"), obs.ts, "h")
cat("\n")
cat('#  /')
cat("\n")
cat("\n")
cat("#  VELOCITY_AT_CENTER ")
cat("\n")
cat("  VELOCITY_AT_FACE ")
cat("\n")
cat('END')
cat("\n")
cat("\n")
cat('#=======================times========================')
cat("\n")
cat("TIME")
cat("\n")
cat(paste("  FINAL_TIME", nhours, "h"))
cat("\n")
cat(paste("  INITIAL_TIMESTEP_SIZE", ini.ts, "h"))
cat("\n")
cat(paste("  MAXIMUM_TIMESTEP_SIZE", max.ts, "h"))
cat("\n")
cat('END')
cat("\n")
cat("\n")
cat('#=======================regions========================')
cat("\n")
cat('REGION all')
cat('\n')
cat('  COORDINATES')
cat('\n')
cat("  ")
cat(xrange[1], yrange[1], zrange[1])
cat('\n')
cat("  ")
cat(xrange[2], yrange[2], zrange[2])
cat('\n')
cat('  /')
cat('\n')
cat('END')
cat('\n')
cat('\n')

cat('REGION Bottom')
cat('\n')
cat('FACE BOTTOM')
cat('\n')
cat('  COORDINATES')
cat('\n')
cat("  ")
cat(xrange[1], yrange[1], zrange[1])
cat('\n')
cat("  ")
cat(xrange[2], yrange[2], zrange[1])
cat('\n')
cat('  /')
cat('\n')
cat('END')
cat('\n')
cat('\n')

cat('REGION Top')
cat('\n')
cat('FACE TOP')
cat('\n')
cat('  COORDINATES')
cat('\n')
cat("  ")
cat(xrange[1], yrange[1], zrange[2])
cat('\n')
cat("  ")
cat(xrange[2], yrange[2], zrange[2])
cat('\n')
cat('  /')
cat('\n')
cat('END')
cat('\n')
cat('\n')

cat('REGION South')
cat('\n')
cat('FACE SOUTH')
cat('\n')
cat('  COORDINATES')
cat('\n')
cat("  ")
cat(xrange[1], yrange[1], zrange[1])
cat('\n')
cat("  ")
cat(xrange[2], yrange[1], zrange[2])
cat('\n')
cat('  /')
cat('\n')
cat('END')
cat('\n')
cat('\n')

cat('REGION North')
cat('\n')
cat('FACE NORTH')
cat('\n')
cat('  COORDINATES')
cat('\n')
cat("  ")
cat(xrange[1], yrange[2], zrange[1])
cat('\n')
cat("  ")
cat(xrange[2], yrange[2], zrange[2])
cat('\n')
cat('  /')
cat('\n')
cat('END')
cat('\n')
cat('\n')

cat('REGION West')
cat('\n')
cat('FACE WEST')
cat('\n')
cat('  COORDINATES')
cat('\n')
cat("  ")
cat(xrange[1], yrange[1], zrange[1])
cat('\n')
cat("  ")
cat(xrange[1], yrange[2], zrange[2])
cat('\n')
cat('  /')
cat('\n')
cat('END')
cat('\n')
cat('\n')

cat('REGION East')
cat('\n')
cat('FACE EAST')
cat('\n')
cat('  COORDINATES')
cat('\n')
cat("  ")
cat(xrange[2], yrange[1], zrange[1])
cat('\n')
cat("  ")
cat(xrange[2], yrange[2], zrange[2])
cat('\n')
cat('  /')
cat('\n')
cat('END')
cat('\n')
cat('\n')

# cat('REGION Inj_Well')
# cat('\n')
# cat('  COORDINATES')
# cat('\n')
# cat(xx[inj_wellname], yy[inj_wellname], screen_bot[inj_wellname])
# cat('\n')
# cat(xx[inj_wellname], yy[inj_wellname], screen_top[inj_wellname])
# cat('\n')
# cat('  /')
# cat('\n')
# cat('/')
# cat('\n')

# %%%%%%%%%%%%%%%%%%%river regions%%%%%%%%%%%%%%%%%%%%%%%%%%

for (iriver in river_region) {
  cat(paste('REGION ', iriver, sep = '' ))
  cat('\n')
  cat(paste('  FILE ', fname_material.h5))
  cat('\n')
  cat('END')
  cat("\n")
  cat("\n")
}
cat("\n")

# #------------- %put observation wells in "regions" of input deck----------------------
# cat('# Observation wells')
# cat("\n")
# cat("\n")
# 
# for (iwell in well.list) {
#   
#   if (screen_top[iwell] >= z0 & screen_bot[iwell] <= z0+zlen) {
#   z = cells_z[which((cells_z >= screen_bot[iwell]) & (cells_z <= screen_top[iwell]))]
#   nz = length(z)
#   # wellname = rownames(Well)[iwell]
#     for (j in 1:nz) {
#       cat(paste('REGION ', iwell, '_', j, sep = ''))
#       cat('\n')
#       cat('  COORDINATE ');
#       cat(xx[iwell],yy[iwell], z[j])
#       cat('\n')
#       cat('/')
#       cat('\n')
#       cat('\n')
#     }
#   }
# }

cat("#==================flow conditions=========================")
cat("\n")
cat("MINIMUM_HYDROSTATIC_PRESSURE -1.d0")
cat("\n")
cat("\n")

cat("FLOW_CONDITION Initial")
cat("\n")
cat("  TYPE")
cat("\n")
cat("    PRESSURE hydrostatic")
cat("\n")
cat('  /')
cat("\n")
cat("  CYCLIC")
cat("\n")
cat("  DATUM DATASET Initial_Head")
cat("\n")
cat("  PRESSURE 101325")
cat("\n")
cat("END")
cat("\n")
cat("\n")

# cat("FLOW_CONDITION Recharge")
# cat("\n")
# cat("  TYPE")
# cat("\n")
# cat("    FLUX neumann")
# cat("\n")
# cat('/')
# cat("\n")
# cat(paste("    FLUX"), recharge)
# cat("\n")
# cat('/')
# cat("\n")


#%%%%%%%%%%%%%%%%%%%% river regions %%%%%%%%%%%%%%%%%%%%%%%%%
for (iflow in flow_region) {
  cat(paste("FLOW_CONDITION"), iflow)
  cat("\n")
  cat("  TYPE")
  cat("\n")
  cat("  PRESSURE conductance")
  cat("\n")
  cat('  /')
  cat("\n")
  cat(paste("  CONDUCTANCE"), cond.list[iflow])
  cat("\n")
  cat("  CYCLIC")
  cat("\n")
  cat(paste("  DATUM file ", fname_bc_dir, fname.river_datum[iflow], sep = ""))
  cat("\n")
  cat("  PRESSURE 101325")
  cat("\n")
  cat("  GRADIENT")
  cat("\n")
  cat(paste("  PRESSURE file ",fname_bc_dir, fname.river_gradient[iflow], sep = ""))
  cat("\n")
  cat('  /')
  cat("\n")
  cat('/')
  cat("\n")
  cat("\n")
}

# ## flow BCs
# for (iflow.BC in flow.BC) {
#   cat(paste("FLOW_CONDITION"), iflow.BC)
#   cat("\n")
#   cat("  TYPE")
#   cat("\n")
#   cat("  PRESSURE hydrostatic")
#   cat("\n")
#   cat("  /")
#   cat("\n")
#   cat("  CYCLIC")
#   cat("\n")
#   cat(paste("  DATUM DATASET"), BC.list[iflow.BC])
#   cat("\n")
#   cat("  PRESSURE 101325")
#   cat("\n")
#   cat("/")
#   cat("\n")
#   cat("\n")
# }


cat("#==================transport conditions=========================")
cat("\n")
cat("TRANSPORT_CONDITION Initial")
cat("\n")
cat("  TYPE dirichlet_zero_gradient")
cat("\n")
cat("  CONSTRAINT_LIST")
cat("\n")
cat("  ")
cat(paste(0, "Concentration_initial"))
cat("\n")
cat("/")
cat("\n")
cat("END")
cat("\n")

for (isolute in solute.list) {
  cat(paste("TRANSPORT_CONDITION"), isolute)
  cat("\n")
  cat("  TYPE dirichlet")
  cat("\n")
  cat("  CONSTRAINT_LIST")
  cat("\n")
  cat("  ")
  cat(paste(0, concentration.list[isolute]))
  cat("\n")
  cat("/")
  cat("\n")
  cat("END")
  cat("\n")
}

# sink()
cat("#==================constraints=========================")
cat("\n")
for (iconc in concentration.list) {
  cat(paste("CONSTRAINT", iconc))
  cat("\n")
  cat("  CONCENTRATIONS")
  cat("\n")
  for (itracer in tracer.list) {
    if (itracer == tracer.list[iconc]) {
      cat("   ")
      cat(paste(itracer), "1.d0 T")
    } else {
      cat("   ")
      cat(paste(itracer), "1.d-20 T")
    }
    cat("\n")

  }
  cat("    Tracer_Age  1.d-20 T")
  cat("\n")  
  cat("  /")
  cat("\n")
  cat("END")
  cat("\n")
  cat("\n")
}

cat("\n")
cat("CONSTRAINT Concentration_initial")
cat("\n")
cat("  CONCENTRATIONS")
cat("\n")
for (itracer in tracer.list) {
  cat("   ")
  cat(paste(itracer), "1.d-20 T")
  cat("\n")
}
cat("    Tracer_Age  1.d-20 T")
cat("\n")  
cat("  /")
cat("\n")
cat("END")
cat("\n")
cat("\n")

cat("\n")
cat("#==================condition couplers=========================")
# 
cat("\n")
cat("INITIAL_CONDITION")
cat("\n")
cat("  FLOW_CONDITION Initial")
cat("\n")
cat("  TRANSPORT_CONDITION Initial")
cat("\n")
cat("  REGION all")
cat("\n")
cat("END")
cat("\n")
cat("\n")

#%%%%%%%%%%%%%%%%%%%%% River BC coupling %%%%%%%%%%%%%%%%%%%%%%%%%%
for (iflow in flow_region) {
  cat(paste("BOUNDARY_CONDITION River_", slice.list[iflow], sep = ""))
  cat("\n")
  cat(paste("  FLOW_CONDITION"), iflow)
  cat("\n")
  cat(paste("  TRANSPORT_CONDITION"), solute.coupler.list[iflow])
  cat("\n")
  cat(paste("  REGION"), river_region[iflow])
  cat("\n")
  cat("END")
  cat("\n")
  cat("\n")
}

# for (iface in BC.face) {
#   cat(paste("BOUNDARY_CONDITION", iface))
#   cat("\n")
#   cat(paste("  FLOW_CONDITION", flow.BC[iface]))
#   cat("\n")
#   if (iface == "North") {
#     cat(paste("  TRANSPORT_CONDITION", solute.list[5]))
#   } else {
#     cat("  TRANSPORT_CONDITION Initial")
#   }
#   cat("\n")
#   cat(paste("  REGION"), iface) 
#   cat("\n")
#   cat("END")
#   cat("\n")
#   cat("\n")
# }

# for (iface in BC.face) {
#   cat(paste("BOUNDARY_CONDITION", iface))
#   cat("\n")
#   cat(paste("  FLOW_CONDITION", flow.BC[iface]))
#   cat("\n")
#     cat("  TRANSPORT_CONDITION Initial")
#   cat("\n")
#   cat(paste("  REGION"), iface)
#   cat("\n")
#   cat("END")
#   cat("\n")
#   cat("\n")
# }


cat("#==================stratigraphy couplers=========================")
cat("\n")
cat("STRATA")
cat("\n")
cat(paste("  MATERIAL", fname_material.h5))
cat("\n")
cat("END")
cat("\n")
cat("\n")
# cat("#==================observation points=========================")
# cat('\n')
# # % putting in "observation points" in input deck
# for (iwell in well.list){
#   
#   if (screen_top[iwell] >= z0 & screen_bot[iwell] <= z0+zlen) {
#   z = cells_z[which(cells_z >= screen_bot[iwell] & (cells_z <= screen_top[iwell]))]
#   nz = length(z)
#   # wellname = rownames(Well)[iwell]
#     for (j in 1:nz){
#  
#       cat('OBSERVATION')
#       cat('\n')
#       cat(paste('  REGION ', iwell, '_', j, sep = ''))
#       cat('\n')
#       cat('  VELOCITY ')
#       cat('\n')
#       cat('/')
#       cat('\n')
#       cat('\n')
#     }
#   }
# }

cat("#==================END=========================")
cat("\n")
cat("END_SUBSURFACE")
sink()