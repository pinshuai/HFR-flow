## #This file is used for calculating transient boundary conditions
## #using universal kriging 

###cov_model_sets = c('gaussian','wave','exponential','spherical')
###drift_sets = c(0,1)

setwd("/Users/shua784/Dropbox/PNNL/Projects/Reach_scale_model/")

# rm(list=ls())
library(geoR)
library(rhdf5)
library(ggplot2)
# library(gstat)
library(sp)
library(maptools)
library(phylin)
##------------INPUT----------------##
# source("codes/300A_parameters.R")
H5close()
options(geoR.messages=FALSE)
# input_folder = 'data/headdata4krige_Plume_2008-2017/'
fname_geoFramework.r = "results/geoframework_200m.r"
fname_river.geo = "data/river_geometry_manual.csv"

fname_mvAwln = "/Users/shua784/Dropbox/PNNL/People/From_Patrick/SQL/mvAwln.csv"
fname_mvAwln_id = "/Users/shua784/Dropbox/PNNL/People/From_Patrick/SQL/mvAwln_wellID_updated.csv"

fname_manual_wells_ids = "/Users/shua784/Dropbox/PNNL/People/From_Patrick/SQL/HYDRAULIC_HEAD_MV_WellID.csv"
fname_manual_wells = "/Users/shua784/Dropbox/PNNL/People/From_Patrick/SQL/HYDRAULIC_HEAD_MV.csv"

fname_USGS_wells = "/Users/shua784/Dropbox/PNNL/People/from_Erick/Burns_well_data.csv"
fname_USGS_wells_ids = "/Users/shua784/Dropbox/PNNL/People/from_Erick/Burns_well_attributes.csv"

fname_SFA_wells = "/Users/shua784/Dropbox/PNNL/People/Velo/300A_Well_Data/"
fname_SFA_wells_ids = "/Users/shua784/Dropbox/PNNL/People/Velo/300A_well_coord.csv"
fname_SFA_wells_all = "/Users/shua784/Dropbox/PNNL/People/Velo/SFA_all_wells.csv"

is.plot = F

##--------------OUTPUT---------------------##

fname_initial.h5 = "Inputs/HFR_model_200m/HFR_H_Initial_2007_04_01.h5"
# BC.h5 = "Inputs/HFR_H_BC.h5"

# fname_head.bc.r= "results/HFR_head_BC.r"
fname_wells.r = "results/well_compiled_wl_data.r"
# fname_fig.initalH_contour = "figures/initial_head_150m.jpg"
# fname_fig.initialH_krige = "figures/initial_head_krige.jpg"
fname_fig.initialH_idw = "figures/initial_head_200m_2017-04-01.jpg"
fname.selected.wells.df = "results/selected.wells.df_2007-3-28.r"

load(fname_geoFramework.r)

## for grids
grid.x = idx
grid.y = idy
grid.nx = nx
grid.ny = ny
# pred.grid.south = expand.grid(seq(range_x[1]+grid.x/2,range_x[2],grid.x),range_y[1]+grid.y/2) # for South boundary
# pred.grid.north = expand.grid(seq(range_x[1]+grid.x/2,range_x[2],grid.x),range_y[2]-grid.y/2) # for North boundary
# pred.grid.east = expand.grid(range_x[1]+grid.x/2,seq(range_y[1]+grid.y/2,range_y[2],grid.y)) # for East boundary
# pred.grid.west = expand.grid(range_x[2]-grid.x/2,seq(range_y[1]+grid.y/2,range_y[2],grid.y)) # for West boundary
pred.grid.domain = expand.grid(seq(range_x[1]+grid.x/2,range_x[2],grid.x),
                               seq(range_y[1]+grid.y/2,range_y[2],grid.y)) # for domain
# colnames(pred.grid.south)=c('x','y')
# colnames(pred.grid.north)=c('x','y')
# colnames(pred.grid.east)=c('x','y')
# colnames(pred.grid.west)=c('x','y')
colnames(pred.grid.domain)=c('x','y')


## time information
# start.time = as.POSIXct("2010-02-27 12:00:00",tz="GMT",format="%Y-%m-%d %H:%M:%S")
# end.time = as.POSIXct("2010-02-28 23:00:00",tz="GMT",format="%Y-%m-%d %H:%M:%S")

# dt = 3600  ##secs
# times = seq(start.time,end.time,dt)
# ntime = length(times)
# time.id = seq(0,ntime-1,dt/3600)  ##hourly boundary, why start from 0h?

# origin.time = as.POSIXct("2007-12-31 23:00:00",tz="GMT",format="%Y-%m-%d %H:%M:%S") # starting time should be 1 h early than "2008-1-1 0:0:0" to set the right index in folder/headdata4krige_Plume_2008_2017

## BC.south = array(NA,c(ntime,grid.nx))
## BC.north = array(NA,c(ntime,grid.nx))
## BC.east = array(NA,c(ntime,grid.ny))
## BC.west = array(NA,c(ntime,grid.ny))


BC.south = c()
BC.north = c()
BC.east = c()
BC.west = c()
avail.time.id = c()

range.xcoods = c(model_origin[1], model_origin[1] + xlen)
range.ycoods = c(model_origin[2], model_origin[2] + ylen)

# ##==================== read into well data ====================
if (!file.exists(fname_wells.r)) {
  

mvAwln.id = read.csv(fname_mvAwln_id, stringsAsFactors = F)
mvAwln = read.csv(fname_mvAwln, stringsAsFactors = F)
mvAwln.id = transform(mvAwln.id,Easting = as.numeric(Easting),
                      Northing = as.numeric(Northing))
HEIS_auto_wells = subset(mvAwln, select = c("WellName", "WellNumber", "procWaterElevation", "procDate"))
HEIS_auto_wells = transform(HEIS_auto_wells, WellName = as.character(WellName),
                   WellNumber = as.character(WellNumber),
                   procWaterElevation = as.numeric(procWaterElevation),
                   procDate = as.POSIXct(procDate))

manual_wells_ids = read.csv(fname_manual_wells_ids, stringsAsFactors = F)
manual_wells = read.csv(fname_manual_wells, stringsAsFactors = F)
manual_wells = transform(manual_wells, HYD_DATE_TIME_PST = as.POSIXct(HYD_DATE_TIME_PST))

# HEIS_auto_wells = mvAwln
HEIS_auto_wells_ids = mvAwln.id
HEIS_manual_wells = manual_wells
colnames(HEIS_manual_wells)[1:4] = c("WellNumber", "WellName", "procDate", "procWaterElevation")
HEIS_manual_wells_ids = manual_wells_ids

USGS_wells = read.csv(fname_USGS_wells, stringsAsFactors = F)
USGS_wells_ids = read.csv(fname_USGS_wells_ids, stringsAsFactors = F)
USGS_wells_ids = transform(USGS_wells_ids, CP_ID_NUM = as.character(CP_ID_NUM))
USGS_wells = transform(USGS_wells, CP_NUM = as.character(CP_NUM), DATE = as.POSIXct(DATE))
USGS_wells$WLELEVft88 = USGS_wells$WLELEVft88*0.3048 # convert ft to meter
USGS_wells_ids$X_SP_83FT = USGS_wells_ids$X_SP_83FT*0.3048
USGS_wells_ids$Y_SP_83FT = USGS_wells_ids$Y_SP_83FT*0.3048
colnames(USGS_wells)[1:4] = c("WellNumber", "procDate", "Year_fract", "procWaterElevation")
colnames(USGS_wells_ids)[2:4] = c("WellNumber", "Easting", "Northing")

## select USGS wells
USGS_wells_selected.names = USGS_wells_ids$WellNumber[which(USGS_wells_ids$Easting < range.xcoods[2] & USGS_wells_ids$Easting > range.xcoods[1] & 
                                                              USGS_wells_ids$Northing < range.ycoods[2] & USGS_wells_ids$Northing > range.ycoods[1])]
USGS_wells_selected = data.frame(WellName = character(), Easting = numeric(), 
                                 Northing = numeric(), DateTime = as.POSIXct(character()), WL = numeric(), stringsAsFactors = F)

for (iwell in USGS_wells_selected.names) {
  
  manual_well = USGS_wells[which(USGS_wells$WellNumber == iwell), ]
  USGS_wells_selected = rbind(USGS_wells_selected, data.frame(WellNumber = manual_well$WellNumber, WL = manual_well$procWaterElevation,
                                                              DateTime = manual_well$procDate,
                                                              Easting = rep(USGS_wells_ids$Easting[which(USGS_wells_ids$WellNumber == iwell)], length(manual_well$WellNumber)),
                                                              Northing = rep(USGS_wells_ids$Northing[which(USGS_wells_ids$WellNumber == iwell)], length(manual_well$WellNumber)),
                                                              stringsAsFactors = F
  ))
  
  
}


## SFA wells
SFA_wells_ids = read.csv(fname_SFA_wells_ids, stringsAsFactors = F)
colnames(SFA_wells_ids)[2] = c("WellName")
# SFA_wells_list=c("399-1-1_3var.csv")
# iwell = SFA_wells_list

        if (!file.exists(fname_SFA_wells_all)) {
          
        SFA_wells = data.frame(WellName = as.character(), DateTime = as.POSIXct(character()),  Temp = numeric(),
                               Spc = numeric(), WL = numeric(), stringsAsFactors = F)
        SFA_wells_list = list.files(fname_SFA_wells)
        for (iwell in SFA_wells_list) {
          # iwell = "399-1-1_3var.csv"
          iSFA_well = read.csv(paste(fname_SFA_wells, iwell, sep = ""), stringsAsFactors = F)
          # iSFA_well = read.csv(paste(fname_SFA_wells, "399-1-1_3var.csv", sep = ""), stringsAsFactors = F)
        
           colnames(iSFA_well) = c("DateTime", "Temp", "Spc", "WL")
        
           if (iwell %in% c("399-5-1_3var.csv", "399-3-19_3var.csv" ) ) {
             iSFA_well$DateTime = as.POSIXct(iSFA_well$DateTime, format = "%m/%d/%y %H:%M", tz = "GMT") ## time formate must agree with data-column
        
           } else {
           iSFA_well$DateTime = as.POSIXct(iSFA_well$DateTime, format = "%d-%b-%Y %H:%M:%S", tz = "GMT") ## time formate must agree with data-column
           }
        
          id_col = data.frame(WellName = rep(gsub("_3var.csv", "", iwell), dim(iSFA_well)[1]), stringsAsFactors = F)
          iSFA_well = cbind(id_col, iSFA_well)
        
          SFA_wells = rbind(SFA_wells, iSFA_well, stringsAsFactors =F)
        }
        
        # as.POSIXct(strptime(SFA_wells$DateTime[2], "%d-%b-%Y %H:%M:%S"), format = "%d-%m-%Y %H:%M:%S", tz = "GMT")
        # SFA_wells$DateTime = as.POSIXct(SFA_wells$DateTime, format = "%d-%b-%Y %H:%M:%S", tz = "GMT") ## time formate must agree with data-column
        
        write.csv(SFA_wells, file = "/Users/shua784/Dropbox/PNNL/People/Velo/SFA_all_wells.csv", row.names = F)
        } else {
          SFA_wells = read.csv(fname_SFA_wells_all, stringsAsFactors = F)
        }



save(list = c("HEIS_auto_wells", "HEIS_auto_wells_ids", "HEIS_manual_wells", "HEIS_manual_wells_ids",
              "USGS_wells", "USGS_wells_ids", "USGS_wells_selected", "USGS_wells_selected.names","SFA_wells", "SFA_wells_ids"), file = fname_wells.r)

} else {
load(fname_wells.r)

}




##-------------------- plot all USGS wells------------------------
if (is.plot) {
  

USGS_wells_selected.names = USGS_wells_ids$WellNumber[which(USGS_wells_ids$Easting < range.xcoods[2] & USGS_wells_ids$Easting > range.xcoods[1] & 
                                                              USGS_wells_ids$Northing < range.ycoods[2] & USGS_wells_ids$Northing > range.ycoods[1])]
USGS_wells_selected = data.frame(WellName = character(), Easting = numeric(), 
                                 Northing = numeric(), DateTime = as.POSIXct(character()), WL = numeric(), stringsAsFactors = F)
start.time = as.POSIXct("1990-01-01 00:00:00",tz="GMT",format="%Y-%m-%d %H:%M:%S")
end.time = as.POSIXct("2011-01-01 00:00:00",tz="GMT",format="%Y-%m-%d %H:%M:%S")
jpeg(file="figures/USGS.wells.jpg", width=12, height=16, units="in", res=300)
par(mar =c(4,4,1,1))
plot(0,0,xlim=c(start.time, end.time), ylim = c(100, 305),type = "n", xlab = "Date", ylab = "Water Level (m)",
     axes = F, cex=1.5)
box()
colors = rainbow(100)
for (iwell in USGS_wells_selected.names) {
  
  manual_well = USGS_wells[which(USGS_wells$WellNumber == iwell), ]
  USGS_wells_selected = rbind(USGS_wells_selected, data.frame(WellNumber = manual_well$WellNumber, WL = manual_well$procWaterElevation,
                                                      DateTime = manual_well$procDate,
                                                      Easting = rep(USGS_wells_ids$Easting[which(USGS_wells_ids$WellNumber == iwell)], length(manual_well$WellNumber)),
                                                      Northing = rep(USGS_wells_ids$Northing[which(USGS_wells_ids$WellNumber == iwell)], length(manual_well$WellNumber)),
                                                      stringsAsFactors = F
  ))
  
  lines(manual_well$procDate, manual_well$procWaterElevation, col= sample(colors), lwd = 1 )
  points(manual_well$procDate, manual_well$procWaterElevation, pch=1, cex=1)
  axis.POSIXct(1,at=seq(as.Date("1990-01-01 00:00:00",tz="GMT"),
                        to=as.Date("2011-01-01 00:00:00",tz="GMT"),by="quarter"),
               format="%m/%Y",mgp=c(5,1.7,0),cex.axis=1)
  axis(2,at=seq(100, 305, 5),mgp=c(5,0.7,0),cex.axis=1)
  
}
dev.off()



hist(USGS_wells_selected$DateTime, breaks = 1000, freq = T)

##---------------------- plot all east wells----------------------------
east.wells=c()
pattern = c(glob2rx("15N*"),glob2rx("14N*"), glob2rx("13N*"), glob2rx("12N*"),glob2rx("11N*"), glob2rx("10N*"), glob2rx("09N*"))
east.wells = grep(paste(pattern,collapse = "|"), HEIS_manual_wells_ids$WELL_NAME, value = T)

east.wells.data = data.frame(WellName = character(), Easting = numeric(), 
                            Northing = numeric(), DateTime = as.POSIXct(character()), WL = numeric(), stringsAsFactors = F)

start.time = as.POSIXct("1990-01-01 00:00:00",tz="GMT",format="%Y-%m-%d %H:%M:%S")
end.time = as.POSIXct("2008-01-01 00:00:00",tz="GMT",format="%Y-%m-%d %H:%M:%S")
jpeg(file="figures/east.wells.jpg", width=12, height=16, units="in", res=300)
par(mar =c(4,4,1,1))
plot(0,0,xlim=c(start.time, end.time), ylim = c(150, 305),type = "n", xlab = "Date", ylab = "Water Level (m)",
     axes = F, cex=1.5)
box()
colors = rainbow(100)
for (iwell in east.wells) {
  
  manual_well = HEIS_manual_wells[which(HEIS_manual_wells$WellName == iwell), ]
  east.wells.data = rbind(east.wells.data, data.frame(WellName = manual_well$WellName, WL = manual_well$procWaterElevation,
                                                    DateTime = manual_well$procDate,
                                                    Easting = rep(HEIS_manual_wells_ids$EASTING[which(HEIS_manual_wells_ids$WELL_NAME == iwell)], length(manual_well$WellName)),
                                                    Northing = rep(HEIS_manual_wells_ids$NORTHING[which(HEIS_manual_wells_ids$WELL_NAME == iwell)], length(manual_well$WellName)),
                                                    stringsAsFactors = F
  ))
  
  lines(manual_well$procDate, manual_well$procWaterElevation, col= sample(colors), lwd = 1 )
  points(manual_well$procDate, manual_well$procWaterElevation, pch=1, cex=1)
  axis.POSIXct(1,at=seq(as.Date("1990-01-01 00:00:00",tz="GMT"),
                        to=as.Date("2008-01-01 00:00:00",tz="GMT"),by="quarter"),
               format="%m/%Y",mgp=c(5,1.7,0),cex.axis=1)
  axis(2,at=seq(150, 305, 5),mgp=c(5,0.7,0),cex.axis=1)
  
  # date.range = range(manual_well$procDate)
  # 
  # print(paste(iwell, "has", length(manual_well$procWaterElevation), "obs. points"))
}
dev.off()

hist(east.wells.data$DateTime, breaks = 1000, freq = T)

##---------------------- plot all HEIS manual wells--------------------


start.time = as.POSIXct("1990-01-01 00:00:00",tz="GMT",format="%Y-%m-%d %H:%M:%S")
end.time = as.POSIXct("2017-01-01 00:00:00",tz="GMT",format="%Y-%m-%d %H:%M:%S")
jpeg(file="figures/all.manual.wells.jpg", width=12, height=16, units="in", res=300)
par(mar =c(4,4,1,1))
plot(0,0,xlim=c(start.time, end.time), ylim = c(100, 305),type = "n", xlab = "Date", ylab = "Water Level (m)",
     axes = F, cex=1.5)
box()
colors = rainbow(100)
for (iwell in well_names) {
  
  manual_well = HEIS_manual_wells[which(HEIS_manual_wells$WellName == iwell), ]
  
  lines(manual_well$procDate, manual_well$procWaterElevation, col= sample(colors), lwd = 1 )
  points(manual_well$procDate, manual_well$procWaterElevation, pch=1, cex=1)
  axis.POSIXct(1,at=seq(as.Date("1990-01-01 00:00:00",tz="GMT"),
                        to=as.Date("2017-01-01 00:00:00",tz="GMT"),by="quarter"),
               format="%m/%Y",mgp=c(5,1.7,0),cex.axis=1)
  axis(2,at=seq(150, 305, 5),mgp=c(5,0.7,0),cex.axis=1)
  
}
dev.off()

hist.HEIS = hist(HEIS_manual_wells$procDate, breaks = 1000, freq = T)

}

##---------------------------select wells with data at each time stamp-----------------------
if (!file.exists(fname.selected.wells.df)) {
  


## create empty matrix with Colclasses defined

# initial.time = as.POSIXct("2005-03-29 12:00:00",tz="GMT",format="%Y-%m-%d %H:%M:%S")
# initial.time = as.POSIXct("2007-03-28 12:00:00",tz="GMT",format="%Y-%m-%d %H:%M:%S")
initial.time = as.POSIXct("2007-04-01 00:00:00",tz="GMT",format="%Y-%m-%d %H:%M:%S")
# times = initial.time

# min.time = as.POSIXct("2005-01-01 00:00:00",tz="GMT",format="%Y-%m-%d %H:%M:%S")
# max.time = as.POSIXct("2010-01-01 00:00:00",tz="GMT",format="%Y-%m-%d %H:%M:%S")

# times = seq(min.time, max.time, by = 30*86400)
# times = seq(min.time, max.time, by = "month")
# well_names = unique(HEIS_manual_wells_ids$WELL_NAME)

# USGS_wells_selected.names = USGS_wells_ids$WellNumber[which(USGS_wells_ids$Easting < range.xcoods[2] & USGS_wells_ids$Easting > range.xcoods[1] & 
#                                                               USGS_wells_ids$Northing < range.ycoods[2] & USGS_wells_ids$Northing > range.ycoods[1])]


well_names = c(unique(HEIS_manual_wells_ids$WELL_NAME), unique(USGS_wells_selected.names), unique(SFA_wells_ids$WellName))

# well_names = c(unique(SFA_wells_ids$WellName)[27:48])

# time_mar = 1*24*3600 #1 day range
time_mar = 15*86400 #15 day range

# times=times[1]
times = initial.time
# itime = times

for (i in 1:length(times)) {
  itime = times[i]
  
  print(itime)
  selected.wells = data.frame(WellName = character(),  WellNumber = character(),
                              Easting = numeric(), 
                              Northing = numeric(), DateTime = as.POSIXct(character()), WL = numeric(), stringsAsFactors = F)
  

    # well_names = c("699-39-79", "199-D3-2", "199-B2-12", "399-5-1")
      # well_names = c("399-1-1")
      for (iwell in well_names) {
        
        # iwell = c("199-B2-12")
        # iwell = c("199-D3-2")

        if (iwell %in% SFA_wells$WellName) {
          print(paste(iwell, "(SFA)"))
          manual_well = SFA_wells[which(SFA_wells$WellName == iwell), ]
          index = which.min(abs(as.numeric(manual_well$DateTime - itime)))
          DateTime = manual_well$DateTime[index]

          if (DateTime == itime) {
            WL = manual_well$WL[index]
            selected.wells = rbind(selected.wells, data.frame(WellName = manual_well$WellName[index],
                                                              WellNumber = manual_well$WellName[index],
                                                              WL = WL,
                                                              DateTime = DateTime,
                                                              Easting = SFA_wells_ids$Easting[which(SFA_wells_ids$WellName == iwell)],
                                                              Northing = SFA_wells_ids$Northing[which(SFA_wells_ids$WellName == iwell)],
                                                              stringsAsFactors = F
            ))
          } else if (DateTime < itime + time_mar & DateTime > itime - time_mar) {
            print(paste(iwell,"has wl within 1day of itime (SFA well)"))
            WLs = manual_well$WL[which(manual_well$DateTime <itime + time_mar & manual_well$DateTime > itime - time_mar)]
            WL = median(WLs)
            selected.wells = rbind(selected.wells, data.frame(WellName = manual_well$WellName[index],
                                                              WellNumber = manual_well$WellName[index],
                                                              WL = WL,
                                                              DateTime = DateTime,
                                                              Easting = SFA_wells_ids$Easting[which(SFA_wells_ids$WellName == iwell)],
                                                              Northing = SFA_wells_ids$Northing[which(SFA_wells_ids$WellName == iwell)],
                                                              stringsAsFactors = F
            ))
          }
        }
        ## sample wells from mvAwln
        if (iwell %in% HEIS_auto_wells$WellName) {
          # print(paste(iwell, "(mvAwln)"))
          auto_well = HEIS_auto_wells[which(HEIS_auto_wells$WellName == iwell), ]
          index = which.min(abs(as.numeric(auto_well$procDate - itime)))
          DateTime = auto_well$procDate[index]
          
          ## find wells having data within given time range
          if (DateTime == itime) {
            WL = auto_well$procWaterElevation[index]
            selected.wells = rbind(selected.wells, data.frame(WellName = auto_well$WellName[index],
                                                              WellNumber = auto_well$WellNumber[index],
                                                              WL = WL,
                                                              DateTime = DateTime,
                                                              Easting = HEIS_manual_wells_ids$EASTING[which(HEIS_manual_wells_ids$WELL_NAME == iwell)],
                                                              Northing = HEIS_manual_wells_ids$NORTHING[which(HEIS_manual_wells_ids$WELL_NAME == iwell)],
                                                              stringsAsFactors = F
            ))
          } else if (DateTime < itime + time_mar & DateTime > itime - time_mar) {
            print(paste(iwell,"has wl within 1day of itime (mvAwln well)"))
            WLs = auto_well$procWaterElevation[which(auto_well$procDate <itime + time_mar & auto_well$procDate > itime - time_mar)]
            WL = median(WLs)
            selected.wells = rbind(selected.wells, data.frame(WellName = auto_well$WellName[index],
                                                              WellNumber = auto_well$WellNumber[index],
                                                              WL = WL,
                                                              DateTime = DateTime,
                                                              Easting = HEIS_manual_wells_ids$EASTING[which(HEIS_manual_wells_ids$WELL_NAME == iwell)],
                                                              Northing = HEIS_manual_wells_ids$NORTHING[which(HEIS_manual_wells_ids$WELL_NAME == iwell)],
                                                              stringsAsFactors = F
            ))
            }

        } 
        ## sample wells from manual HEIS data
        if (iwell %in% HEIS_manual_wells$WellName) {
          # print(paste(iwell, "(HEIS manual)"))
          manual_well = HEIS_manual_wells[which(HEIS_manual_wells$WellName == iwell), ]
          index = which.min(abs(as.numeric(manual_well$procDate - itime)))
          DateTime = manual_well$procDate[index]
          
          if (DateTime == itime) {
            WL = manual_well$procWaterElevation[index]
            selected.wells = rbind(selected.wells, data.frame(WellName = manual_well$WellName[index],
                                                              WellNumber = manual_well$WellNumber[index],
                                                              WL = WL,
                                                              DateTime = DateTime,
                                                              Easting = HEIS_manual_wells_ids$EASTING[which(HEIS_manual_wells_ids$WELL_NAME == iwell)],
                                                              Northing = HEIS_manual_wells_ids$NORTHING[which(HEIS_manual_wells_ids$WELL_NAME == iwell)],
                                                              stringsAsFactors = F
            ))
          } else if (DateTime < itime + time_mar & DateTime > itime - time_mar) {
            print(paste(iwell,"has wl within 1day of itime (manual well)"))
            WLs = manual_well$procWaterElevation[which(manual_well$procDate <itime + time_mar & manual_well$procDate > itime - time_mar)]
            WL = median(WLs)
            selected.wells = rbind(selected.wells, data.frame(WellName = manual_well$WellName[index],
                                                              WellNumber = manual_well$WellNumber[index],
                                                              WL = WL,
                                                              DateTime = DateTime,
                                                              Easting = HEIS_manual_wells_ids$EASTING[which(HEIS_manual_wells_ids$WELL_NAME == iwell)],
                                                              Northing = HEIS_manual_wells_ids$NORTHING[which(HEIS_manual_wells_ids$WELL_NAME == iwell)],
                                                              stringsAsFactors = F
            ))
          }
        }
        ## sample wells from USGS
        if (iwell %in% USGS_wells_selected$WellNumber) {
          # print(paste(iwell, "(USGS)"))
          manual_well = USGS_wells_selected[which(USGS_wells_selected$WellNumber == iwell), ]
          index = which.min(abs(as.numeric(manual_well$DateTime - itime)))
          DateTime = manual_well$DateTime[index]
          
          if (DateTime == itime) {
            WL = manual_well$WL[index]
            selected.wells = rbind(selected.wells, data.frame(WellName = manual_well$WellNumber[index],
                                                              WellNumber = manual_well$WellNumber[index],
                                                              WL = WL,
                                                              DateTime = DateTime,
                                                              Easting = USGS_wells_selected$Easting[index],
                                                              Northing = USGS_wells_selected$Northing[index],
                                                              stringsAsFactors = F
            ))
          } else if (DateTime < itime + time_mar & DateTime > itime - time_mar) {
            print(paste(iwell,"has wl within 1day of itime (USGS well)"))
            WLs = manual_well$WL[which(manual_well$DateTime <itime + time_mar & manual_well$DateTime > itime - time_mar)]
            WL = median(WLs)
            selected.wells = rbind(selected.wells, data.frame(WellName = manual_well$WellNumber[index],
                                                              WellNumber = manual_well$WellNumber[index],
                                                              WL = WL,
                                                              DateTime = DateTime,
                                                              Easting = USGS_wells_selected$Easting[index],
                                                              Northing = USGS_wells_selected$Northing[index],
                                                              stringsAsFactors = F
            ))
          }
        }
        ## sample wells from SFA data

      }
  


selected.wells.unique = selected.wells[!duplicated(selected.wells$WellName), ] # remove duplicated wellNames
selected.wells.unique = selected.wells[!duplicated(selected.wells$Easting), ] # remove duplicated well coords
selected.wells.unique = selected.wells.unique[complete.cases(selected.wells.unique), ] # remove rows contain NAs

selected.wells.df = data.frame(x=selected.wells.unique$Easting, y=selected.wells.unique$Northing, z = selected.wells.unique$WL)
# colnames(data) = c('x','y','z')
selected.wells.df = selected.wells.df[order(selected.wells.df$x),]


# save(selected.wells.df, file = "results/selected.wells.df_2007-4-1.r") 
# ##----------------------- plot well head----------------------
#         # save(selected.wells.df, file = "results/inital_data_coords.r")
#         
#         
#         plot(selected.wells.df$x, selected.wells.df$y, asp = 1)
#         # par(mfrow = c(2, 1))
        # s= interp(selected.wells.df$x, selected.wells.df$y, selected.wells.df$z, duplicate = "strip", nx=100, ny=100)
        # jpeg(paste("figures/", initial.time, ".jpg", sep = ""), width=8,height=8,units='in',res=300,quality=100)
        # image2D(s, shade=0.2, rasterImage = F, NAcol = "gray",
        #         main = paste(initial.time, "inital head (contour)"), asp = 1, contour = T, add = F
               )
# 
#        
        # points(selected.wells.df$x, selected.wells.df$y, col = "white", pch = 1)
#         # dev.off()

# load("results/inital_data_coords.r")

# selected.wells.df = data

   


}   

save(selected.wells.df, file = fname.selected.wells.df) 

} else {
  
  load(fname.selected.wells.df)
  
  if (dim(selected.wells.df)[1]>2) {
    ## ---------------use inverse distance interpolation------------------
    # geo.data = selected.wells.df
    #  coordinates(geo.data)= ~x+y 
    #  plot(geo.data)
    #  x.range <- range.xcoods  # min/max longitude of the interpolation area
    #  y.range <- range.ycoods  # min/max latitude of the interpolation area
    # grd <- expand.grid(x = seq(from = x.range[1], to = x.range[2], by = idx), y = seq(from = y.range[1], 
    #                                                                                   to = y.range[2], by = idy))  # expand points to grid
    grd = expand.grid(unit_x, unit_y)
    
    # save(grd, file = "results/model_grids.r")
    
    idw.interp = idw(values=selected.wells.df[,"z"],
                     coords = selected.wells.df[,c("x","y")],
                     grid=grd,
                     method="shepard",
                     p=2)
    idw.interp = as.numeric(unlist(idw.interp))
    
    h.initial = array(idw.interp, c(nx, ny))
    
    river.geometry = read.csv(fname_river.geo)
    
    river.geometry = river.geometry[, 2:3]
    
    itime = as.POSIXct("2007-04-01 12:00:00",tz="GMT",format="%Y-%m-%d %H:%M:%S")
    
    jpeg(fname_fig.initialH_idw, width=8,height=8,units='in',res=300,quality=100)
    # plot(selected.wells.df$x, selected.wells.df$y, col = "black", pch = 1, asp=1, xlim = c(x.range[1], x.range[2]))
    head4plot = h.initial
    head4plot[head4plot>200]=200
    image2D(z= head4plot, x= unit_x, y= unit_y, shade=0.2, rasterImage = F, NAcol = "white",
            main = paste("Initial Head", itime), asp = 1, contour = T, zlim = c(100, 200), xlab = "Easting", ylab = "Northing")
    points(selected.wells.df$x, selected.wells.df$y, col = "white", pch = 1, asp=1)
    polygon(river.geometry$x, river.geometry$y, border = "gray", asp=1)
    
    dev.off()
  }
  
  
  
  
}

       
 ## ------------------- Krige------------------------------------------       
        # data = as.geodata(data)
        ##This bins and esimator.type is defined by Xingyuan
        # if (nrow(data$coords)>27) {
        #     # bin1 = variog(data,uvec=c(0,50,100,seq(150,210,30),250,300),trend='cte',bin.cloud=T,estimator.type='modulus')
        #   bin1 = variog(data, uvec=c(0, 500, 1000, 2000, 2500, 3500, 4500, 5500, seq(6000,60000,100)),trend='cte',bin.cloud=T,estimator.type='modulus', option = "cloud")
        # } else {
        #     bin1 = variog(data,uvec=c(0,100,seq(150,210,30),250,300),trend='cte',bin.cloud=T,estimator.type='modulus')
        # }
        # initial.values <- expand.grid(max(bin1$v),seq(300))
        # wls = variofit(bin1,ini = initial.values,fix.nugget=T,nugget = 0.00001,fix.kappa=F,cov.model='exponential')


        #check the varigram
        # if (itime %% 1000 == 1) {
            # jpeg(filename=paste('figures/Semivariance Time = ',start.time,".jpg", sep=''),
            #      width=5,height=5,units="in",quality=100,res=300)
            # plot(bin1,main = paste('Time = ',start.time, sep=''),col='red', pch = 19, cex = 1, lty = "solid", lwd = 2)
            # text(bin1$u,bin1$v,labels=bin1$n, cex= 0.7,pos = 2)
            # lines(wls)
            # dev.off()
          # print(times[itime])
        # }


        # ## Generate boundary and initial condition
        # kc.south = krige.conv(data, loc = pred.grid.south, krige = krige.control(obj.m=wls,type.krige='OK',trend.d='cte',trend.l='cte'))    
        # kc.north = krige.conv(data, loc = pred.grid.north, krige = krige.control(obj.m=wls,type.krige='OK',trend.d='cte',trend.l='cte'))
        # kc.east = krige.conv(data, loc = pred.grid.east, krige = krige.control(obj.m=wls,type.krige='OK',trend.d='cte',trend.l='cte'))
        # kc.west = krige.conv(data, loc = pred.grid.west, krige = krige.control(obj.m=wls,type.krige='OK',trend.d='cte',trend.l='cte'))            
        # 
        # BC.south = rbind(BC.south,kc.south$predict)
        # BC.north = rbind(BC.north,kc.north$predict)
        # BC.east = rbind(BC.east,kc.east$predict)
        # BC.west = rbind(BC.west,kc.west$predict)                        
        
    ## krige initial head
        # if (itime==start.time)
        # {
        #     kc.domain = krige.conv(data, loc = pred.grid.domain, krige = krige.control(obj.m=wls,type.krige='OK',trend.d='cte',trend.l='cte'))
        #     h.initial = as.vector(kc.domain$predict)
        #     dim(h.initial) = c(grid.nx,grid.ny)
        # }

            # jpeg(fname_fig.initialH_krige, width=8,height=8,units='in',res=300,quality=100)
            # image2D(z= h.initial, x= unit_x, y= unit_y, shade=0.2, rasterImage = F, NAcol = "white",
            #         main = paste("Initial Head", start.time), asp = 1)
            # dev.off()


#     }
# }

##----------------import initial head from Hanford_Reach_2007_Initial.h5------------------

# old_ini = h5read("Inputs/test_2007_age/Hanford_Reach_2007_Initial.h5", name = "Initial_Head/Data")
# 
# old_ini = t(old_ini)
# 
# old_ini.list = list(x = seq(from = 538000, to = 614000, by = 250), y = seq(from = 97000, to = 164000, by = 250), z = old_ini)
# 
# new_ini = interp.surface(old_ini.list, cells_proj)
# 
# new_ini[which(is.na(new_ini))] = 110
# 
# new_ini = array(new_ini, c(nx,ny))
# 
# image2D(z= new_ini, x= unit_x, y= unit_y, shade=0.2, rasterImage = F, NAcol = "white", border = NA, resfac = 3,
#         main = c("new_ini"), asp = 1)
# image2D(z= old_ini, x = seq(from = 538000, to = 614000, by = 250), y = seq(from = 97000, to = 164000, by = 250), shade=0.2, rasterImage = F, NAcol = "white",
#         main = c("old_ini"), asp = 1)
# 
# fname_initial.h5 = "Inputs/HFR_model_200m/HFR_H_Initial_new.h5"
# 
# if (file.exists(fname_initial.h5)) {
#   file.remove(fname_initial.h5)
# }
# h5createFile(fname_initial.h5)
# h5createGroup(fname_initial.h5,'Initial_Head')
# 
# h5write(t(new_ini),fname_initial.h5, ## why tranpose? to match HDF5 format
#         'Initial_Head/Data',level=0)
# fid = H5Fopen(fname_initial.h5)
# h5g = H5Gopen(fid,'/Initial_Head')
# h5writeAttribute(attr = 1.0, h5obj = h5g, name = 'Cell Centered')
# h5writeAttribute.character(attr = "XY", h5obj = h5g, name = 'Dimension')
# h5writeAttribute(attr = c(200, 200), h5obj = h5g, name = 'Discretization')
# h5writeAttribute(attr = 500.0, h5obj = h5g, name = 'Max Buffer Size')
# h5writeAttribute(attr = c(0, 0), h5obj = h5g, name = 'Origin') 
# H5Gclose(h5g)
# H5Fclose(fid)
##-----------------------------------------------------------------------------------------------





time.id = avail.time.id


##Generate the initial condition hdf5 file for the domain.
if (file.exists(fname_initial.h5)) {
    file.remove(fname_initial.h5)
}
h5createFile(fname_initial.h5)
h5createGroup(fname_initial.h5,'Initial_Head')

h5write(t(h.initial),fname_initial.h5, ## why tranpose? to match HDF5 format
        'Initial_Head/Data',level=0)
fid = H5Fopen(fname_initial.h5)
h5g = H5Gopen(fid,'/Initial_Head')
h5writeAttribute(attr = 1.0, h5obj = h5g, name = 'Cell Centered')
h5writeAttribute.character(attr = "XY", h5obj = h5g, name = 'Dimension')
h5writeAttribute(attr = c(200, 200), h5obj = h5g, name = 'Discretization')
h5writeAttribute(attr = 500.0, h5obj = h5g, name = 'Max Buffer Size')
h5writeAttribute(attr = c(0, 0), h5obj = h5g, name = 'Origin') 
H5Gclose(h5g)
H5Fclose(fid)


# 
# ##Generate the BC hdf5 file.
# if (file.exists(paste(output_folder,BC.h5,sep=''))) {
#     file.remove(paste(output_folder,BC.h5,sep=''))
# } 
# 
# h5createFile(paste(output_folder,BC.h5,sep=''))
# 
# ### write data
# h5createGroup(paste(output_folder,BC.h5,sep=''),'BC_South')
# h5write(time.id,paste(output_folder,BC.h5,sep=''),'BC_South/Times',level=0)
# h5write(BC.south,paste(output_folder,BC.h5,sep=''),'BC_South/Data',level=0)
# 
# h5createGroup(paste(output_folder,BC.h5,sep=''),'BC_North')
# h5write(time.id,paste(output_folder,BC.h5,sep=''),'BC_North/Times',level=0)
# h5write(BC.north,paste(output_folder,BC.h5,sep=''),'BC_North/Data',level=0)
# 
# h5createGroup(paste(output_folder,BC.h5,sep=''),'BC_East')
# h5write(time.id,paste(output_folder,BC.h5,sep=''),'BC_East/Times',level=0)
# h5write(BC.east,paste(output_folder,BC.h5,sep=''),'BC_East/Data',level=0)
# 
# h5createGroup(paste(output_folder,BC.h5,sep=''),'BC_West')
# h5write(time.id,paste(output_folder,BC.h5,sep=''),'BC_West/Times',level=0)
# h5write(BC.west,paste(output_folder,BC.h5,sep=''),'BC_West/Data',level=0)
# 
# ### write attribute
# fid = H5Fopen(paste(output_folder,BC.h5,sep=''))
# h5g.south = H5Gopen(fid,'/BC_South')
# h5g.north = H5Gopen(fid,'/BC_North')
# h5g.east = H5Gopen(fid,'/BC_East')
# h5g.west = H5Gopen(fid,'/BC_West')
# 
# 
# h5writeAttribute(attr = 1.0, h5obj = h5g.south, name = 'Cell Centered')
# h5writeAttribute(attr = 'X', h5obj = h5g.south, name = 'Dimension')
# h5writeAttribute(attr = grid.x, h5obj = h5g.south, name = 'Discretization')
# h5writeAttribute(attr = 200.0, h5obj = h5g.south, name = 'Max Buffer Size')
# h5writeAttribute(attr = range_x[1], h5obj = h5g.south, name = 'Origin')
# h5writeAttribute(attr = 'h', h5obj = h5g.south, name = 'Time Units')
# h5writeAttribute(attr = 1.0, h5obj = h5g.south, name = 'Transient')
# 
# 
# h5writeAttribute(attr = 1.0, h5obj = h5g.north, name = 'Cell Centered')
# h5writeAttribute(attr = 'X', h5obj = h5g.north, name = 'Dimension')
# h5writeAttribute(attr = grid.x, h5obj = h5g.north, name = 'Discretization')
# h5writeAttribute(attr = 200.0, h5obj = h5g.north, name = 'Max Buffer Size')
# h5writeAttribute(attr = range_x[1], h5obj = h5g.north, name = 'Origin')
# h5writeAttribute(attr = 'h', h5obj = h5g.north, name = 'Time Units')
# h5writeAttribute(attr = 1.0, h5obj = h5g.north, name = 'Transient')
# 
# 
# h5writeAttribute(attr = 1.0, h5obj = h5g.east, name = 'Cell Centered')
# h5writeAttribute(attr = 'Y', h5obj = h5g.east, name = 'Dimension')
# h5writeAttribute(attr = grid.y, h5obj = h5g.east, name = 'Discretization')
# h5writeAttribute(attr = 200.0, h5obj = h5g.east, name = 'Max Buffer Size')
# h5writeAttribute(attr = range_y[1], h5obj = h5g.east, name = 'Origin')
# h5writeAttribute(attr = 'h', h5obj = h5g.east, name = 'Time Units')
# h5writeAttribute(attr = 1.0, h5obj = h5g.east, name = 'Transient')
# 
# 
# h5writeAttribute(attr = 1.0, h5obj = h5g.west, name = 'Cell Centered')
# h5writeAttribute(attr = 'Y', h5obj = h5g.west, name = 'Dimension')
# h5writeAttribute(attr = grid.y, h5obj = h5g.west, name = 'Discretization')
# h5writeAttribute(attr = 200.0, h5obj = h5g.west, name = 'Max Buffer Size')
# h5writeAttribute(attr = range_y[1], h5obj = h5g.west, name = 'Origin')
# h5writeAttribute(attr = 'h', h5obj = h5g.west, name = 'Time Units')
# h5writeAttribute(attr = 1.0, h5obj = h5g.west, name = 'Transient')
# 
# 
# H5Gclose(h5g.south)
# H5Gclose(h5g.north)
# H5Gclose(h5g.east)
# H5Gclose(h5g.west)
# H5Fclose(fid)

# save(list=ls(),file=fname_300A.bc.r)
