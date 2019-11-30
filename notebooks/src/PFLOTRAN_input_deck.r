imodel=c("100x100x1")
model_dir = "/global/project/projectdirs/m1800/pin/Reach_scale_model/"
## inputs
fname_WellScreen = paste(model_dir, "data/well_data/HEIS_300A_well_screen.csv", sep = "")

fname_material.h5 = c("HFR_material_river.h5")
fname_H.initial.h5 = c("HFR_H_Initial.h5")
# fname_H.BC.h5 = c("HFR_H_BC.h5")

fname_bc_dir = "bc_6h_smooth/"
fname.DatumH = c("DatumH_Mass1_")
fname.Gradient = c("Gradients_Mass1_")

fname_mass_section = paste(model_dir, "results/HFR_model_", imodel, "/mass_section.txt", sep = "")
## outputs
fname_model_inputs.r = paste(model_dir, "results/HFR_model_", imodel, "/model_inputs.r", sep = "")
fname_pflotran.in = paste(model_dir, "Inputs/HFR_model_", imodel, "/pflotran_", imodel, "_new_iniH.in", sep = "")

is.flowBC = FALSE
has.basalt = TRUE

load(fname_model_inputs.r)
# print(fname_pflotran.in)

#-------------PARAMETERIZATION------------------#
start.time = as.POSIXct("2007-03-28 12:00:00",tz="GMT",format="%Y-%m-%d %H:%M:%S")
end.time = as.POSIXct("2015-12-31 00:00:00",tz="GMT",format="%Y-%m-%d %H:%M:%S")

# --------------------------times----------------------------------
time.index = seq(from=start.time,to=end.time,by="1 hour")
ntime = length(time.index)
nhours = ntime # from 2010-02-27 12:00:00 to 2017-07-01 00:00:00
ini.ts = 0.01 #hour
max.ts = 6 #hour

grid.n_pts = c(nx, ny, nz)
grid.d_pts = c(idx,  idy,  idz)

river_cond = 4.65e-13 # from mean(rand_2) conductance

model_domain_origin = c(0,0,z0)

a = 0 #rotation
wall.stop = 47.8 # wallclock stop time, h
check.pt = 720 # check point, h

# solver
ts.acceleration = 8
max.ts.cuts = 20

diff.coeff = 1e-9 # diffusion coefficient
# recharge = 1.757e-9 #recharge rate 5.54 cm/yr from Fayer and Walters (1995)

##----------------------- material -------------------------------##
if (has.basalt) {
    material.list = c("hanford", "cold_creek", "taylor_flats", "ringold_e", "ringold_lm", "ringold_a", "basalt")
} else {
    material.list = c("hanford", "cold_creek", "taylor_flats", "ringold_e", "ringold_lm", "ringold_a")
}

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

Kb = 1e-3 # from Franklin
basalt = list(ID = 9, porosity = 0.2, tortuosity = 1, perm.x = Kb/(9.5e11), perm.y = Kb/(9.5e11), perm.z = Kb/(9.5e12), cc = "cc1")

K.list = list(hanford = hanford, cold_creek = cold_creek, taylor_flats = taylor_flats, ringold_e = ringold_e, ringold_lm = ringold_lm, 
              ringold_a = ringold_a, basalt =  basalt)
# saturation functions
cc.names = c("cc1", "cc2")

cc1 = list(M = 0.3391, alpha = 7.2727e-4, rsat = 0.16, max.cp = 1e8)
cc2 = list(M = 0.7479, alpha = 1.4319e-4, rsat = 0.1299, max.cp = 1e8)

cc.list = list(cc1 = cc1, cc2 = cc2)

# output opitons
output.ts = 6 # hour
h5.files = 73 # number of hdf5 files to write
obs.ts = 6 # hour

##-----------------------flow region------------------------
mass_section = read.delim(fname_mass_section, header = FALSE, stringsAsFactor = FALSE)
slice.list = as.character(unlist(mass_section))

# slice.list = as.character(seq(40, 332, 1)) 
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
# BC.all = c("BC_East", "BC_West", "BC_South", "BC_North")
BC.list = c("BC_West", "BC_South", "BC_North", "BC_East")
BC.face = c("West", "South", "North", "East")
names(flow.BC) = BC.face
names(BC.list) = flow.BC
max.buff = 200

# model corner coordinates
xrange = c(0, xlen)
yrange = c(0, ylen)
zrange = c(z0, z0+zlen)

#--------------- generate well obs input deck in Pflotran-----------------
#read well screen info
Wells = read.csv(fname_WellScreen, header = TRUE, stringsAsFactors=FALSE)
Wells$WellName = paste("Well_", Wells$WellName, sep="")
# Wells[,1] = paste("Well_", Wells[,1], sep="")
rownames(Wells) = Wells$WellName
# names(Wells) = c("wellname", "x", "y", "elev", "screen_top", "screen_bot")
Well = as.matrix(Wells[,3:7])


Easting = Well[,1]
Northing = Well[,2]
elev = Well[,3]
screen_top = Well[,4]
screen_bot = Well[,5]


# convert project coord to model coord, and rotate coordinates a-deg
x = Easting -x0
y = Northing -y0
xx=x*cos(a)+y*sin(a)
yy=y*cos(a)-x*sin(a)

nwell = length(x)
cells_z = seq((z0+0.5*grid.d_pts[3]), (z0+grid.n_pts[3]*grid.d_pts[3]), grid.d_pts[3])
n_obs = nwell

well.list = Wells$WellName


f <- file(fname_pflotran.in, open = "wt")
cat <- function(...){
    base::cat(..., file=f)
    }
# cat("Hello World\n")


# sink(file = fname_pflotran.in, append = FALSE, type = "output")

cat("## ===============BEGINNING of FILE=========================")
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

cat("#  RESTART pflotran_bigplume-restart.chk\n")
# use result from previoius simulaiton and reset time to zero
cat("#  RESTART \n") 
cat("#  FILENAME pflotran_bigplume-restart.chk \n") 
cat("#  RESET_TO_TIME_ZERO\n")
cat("##  REALIZATION_DEPENDENT\n")
cat("#  END\n")

cat("END\n")
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
cat("#    Tracer_Age")
cat("\n")
cat("  /")
cat("\n")
cat("#  DATABASE tracer_HFR.dat")
cat("\n")
cat("#  ACTIVITY_COEFFICIENTS OFF")
cat("\n")
cat("  OUTPUT")
cat("\n")
cat("    ALL")
cat("\n")
cat("#    AGE")
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
cat("#  MAX_STEPS 100\n")
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
cat("  VERBOSE_ERROR_MESSAGING") ## add detailed error message
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
# cat("  BOUNDS")
# cat("\n")
# cat("  ")
# cat(xrange[1], yrange[1], zrange[1])
# cat("\n")
# cat("  ")
# cat(xrange[2], yrange[2], zrange[2])
# cat("\n")
# cat("/")
# cat("\n")
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

if (is.flowBC) {
    for (iBC in BC.list) {
      cat(paste("DATASET", iBC))
      cat("\n")
      cat(paste("  FILENAME", fname_H.BC.h5))
      cat("\n")
      cat(paste("  HDF5_DATASET_NAME", iBC))
      cat("\n")
      cat(paste("  MAX_BUFFER_SIZE", max.buff))
      cat("\n")
      cat("END")
      cat("\n")
      cat("\n")
    }
}


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
cat("  VARIABLES")
cat("\n")
cat("  LIQUID_PRESSURE")
cat("\n")
cat("  LIQUID_HEAD\n")
cat("  LIQUID_SATURATION")
cat("\n")
cat("  MATERIAL_ID_KLUDGE_FOR_VISIT")  # add composite material ids
cat("\n")
cat("  RESIDUAL\n")
cat('  /')
cat("\n")
cat("\n")
cat("  SNAPSHOT_FILE")
cat("\n")
cat("  NO_PRINT_INITIAL")
cat("\n")
cat(paste("#  PERIODIC TIME"), output.ts, "h")
cat("\n")
cat("  PERIODIC TIME 6 h between 32400 h and 76800 h")
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
cat("  PERIODIC TIME 6 h between 32400 h and 76800 h")
cat("\n")
cat('  /')
cat("\n")
cat("\n")
cat("  OBSERVATION_FILE")
cat("\n")
cat("  NO_PRINT_INITIAL\n")
cat("  PERIODIC TIME 6 h between 32400 h and 76800 h")
cat("\n")
cat('  /')
cat("\n")
cat("\n")
cat("  VELOCITY_AT_CENTER ") # OUTPUT darcy velocity at cell center
cat("\n")
cat("  VELOCITY_AT_FACE ")  # output darcy velocity at cell face
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

#------------- %put observation wells in "regions" of input deck----------------------
cat('# Observation wells')
cat("\n")
cat("\n")

for (iwell in well.list) {
  
  if (screen_top[iwell] >= z0 & screen_bot[iwell] <= z0+zlen & (screen_top[iwell] - screen_bot[iwell]) >= dz[1]) {
    z = cells_z[which((cells_z >= screen_bot[iwell]) & (cells_z <= screen_top[iwell]))]
    nz = length(z)
    # wellname = rownames(Well)[iwell]
    for (j in 1:nz) {
      cat(paste('REGION ', iwell, '_', j, sep = ''))
      cat('\n')
      cat('  COORDINATE ')
      cat(xx[iwell],yy[iwell], z[j])
      cat('\n')
      cat('/')
      cat('\n')
      cat('\n')
    }
  }
}

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
  cat("  INTERPOLATION LINEAR\n")
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

## flow BCs

if (is.flowBC) {
    for (iflow.BC in flow.BC) {
      cat(paste("FLOW_CONDITION"), iflow.BC)
      cat("\n")
      cat("  TYPE")
      cat("\n")
      cat("  PRESSURE hydrostatic")
      cat("\n")
      cat("  /")
      cat("\n")
      cat("  CYCLIC")
      cat("\n")
      cat(paste("  DATUM DATASET"), BC.list[iflow.BC])
      cat("\n")
      cat("  PRESSURE 101325")
      cat("\n")
      cat("/")
      cat("\n")
      cat("\n")
    }
}


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
  cat("  TYPE dirichlet_zero_gradient") # use dirichlet zero gradient when mass is moving in and out of boundary
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
  cat("#    Tracer_Age  1.d-20 T")
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
cat("#    Tracer_Age  1.d-20 T")
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

if (is.flowBC) {
    
    for (iface in BC.face) {
      cat(paste("BOUNDARY_CONDITION", iface))
      cat("\n")
      cat(paste("  FLOW_CONDITION", flow.BC[iface]))
      cat("\n")
        cat("  TRANSPORT_CONDITION Initial")
      cat("\n")
      cat(paste("  REGION"), iface)
      cat("\n")
      cat("END")
      cat("\n")
      cat("\n")
    }
}


cat("#==================stratigraphy couplers=========================")
cat("\n")
cat("STRATA")
cat("\n")
cat(paste("  FILE", fname_material.h5))
cat("\n")
cat("END")
cat("\n")
cat("\n")

cat("#==================observation points=========================")
cat('\n')
# % putting in "observation points" in input deck
for (iwell in well.list){
  
  if (screen_top[iwell] >= z0 & screen_bot[iwell] <= z0+zlen & (screen_top[iwell] - screen_bot[iwell]) >= dz[1]) {
    z = cells_z[which(cells_z >= screen_bot[iwell] & (cells_z <= screen_top[iwell]))]
    nz = length(z)
    # wellname = rownames(Well)[iwell]
    for (j in 1:nz){
      
      cat('OBSERVATION')
      cat('\n')
      cat(paste('  REGION ', iwell, '_', j, sep = ''))
      cat('\n')
      cat('  VELOCITY ')
      cat('\n')
      cat('/')
      cat('\n')
      cat('\n')
    }
  }
}

cat("#==================END=========================")
cat("\n")
cat("END_SUBSURFACE")
# sink()

close(f)
