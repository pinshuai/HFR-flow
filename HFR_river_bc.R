rm(list=ls())
library("xts")
library("signal")

setwd("/Users/shua784/Dropbox/PNNL/Projects/Reach_scale_model/")
##----------INPUT---------------##
fname_mass1_coord = "data/MASS1/coordinates.csv"
# fname_geoFramework.r = "results/geoframework_100m.r"
fname_mass1_pts = "data/MASS1/transient_1976_2016/"
fname_mass1_xts = "results/mass.data.xts.r"
fname_model_inputs.r = "results/model_inputs_200m.r"
# fname_model_coord = "data/model_coord.dat"

is.smooth = T
##--------------OUTPUT----------------##
fname_DatumH = "Inputs/river_bc/bc_1w_smooth_010107/DatumH_Mass1_"
fname_Gradients = "Inputs/river_bc/bc_1w_smooth_010107/Gradients_Mass1_"
fname_mass1_data.r = "results/mass.data.r"

load(fname_model_inputs.r)
# source("mass.data.R")


# start.time = as.POSIXct("2007-03-28 12:00:00",tz="GMT",format="%Y-%m-%d %H:%M:%S")
start.time = as.POSIXct("2007-01-01 00:00:00",tz="GMT",format="%Y-%m-%d %H:%M:%S")
end.time = as.POSIXct("2016-01-01 00:00:00",tz="GMT",format="%Y-%m-%d %H:%M:%S")

mass.coord = read.csv(fname_mass1_coord)
mass.coord[,"easting"] = mass.coord[,"easting"]-model_origin[1]
mass.coord[,"northing"] = mass.coord[,"northing"]-model_origin[2]


##------------------------- compile all the mass1 data --------------------------
if (!file.exists(fname_mass1_xts)) {
  
  slice.list = as.character(mass.coord[,1])
  
mass.data = list()
for (islice in slice.list) {
  print(islice)
  mass.data[[islice]] = read.csv(paste(fname_mass1_pts,"mass1_",
                                       islice,".csv",sep=""))
}
names(mass.data) = slice.list

for (islice in slice.list) {
  print(islice)
  mass.data[[islice]][["date"]] =
    as.POSIXct(mass.data[[islice]][["date"]],format="%Y-%m-%d %H:%M:%S",tz='GMT')
  mass.data[[islice]][["stage"]] = mass.data[[islice]][["stage"]]+1.039
}


# save(mass.data, file=fname_mass1_data.r)

time.index = seq(from=start.time,to=end.time,by="1 hour")
ntime = length(time.index)
simu.time = c(1:ntime-1)*3600

mass.data.xts = list()
for (islice in slice.list)
{
  print(islice)
  mass.data.xts[[islice]] = xts(mass.data[[islice]],
                                order.by=mass.data[[islice]][["date"]] ,unique=T,tz="GMT")
  
  mass.data.xts[[islice]] = mass.data.xts[[islice]][
    .indexmin(mass.data.xts[[islice]][,"date"]) %in% c(56:59,0:5)]

  index(mass.data.xts[[islice]]) = round(index(mass.data.xts[[islice]]),units="hours")
  mass.data.xts[[islice]] = mass.data.xts[[islice]][
    !duplicated(.index(mass.data.xts[[islice]]))]
  mass.data.xts[[islice]] = merge(mass.data.xts[[islice]],time.index)
  
}

save(mass.data.xts,file=paste(results.dir,"mass.data.xts.r",sep=""))
} else {
  load(fname_mass1_xts)
}


##------------------------- generate river bc-----------------------------------
slice.list = names(mass.data.xts)
# slice.list = c("40", "41")
# islice = slice.list
nslice = length(slice.list)


for (islice in slice.list)
{
    mass.data.xts[[islice]] = mass.data.xts[[islice]][index(mass.data.xts[[islice]])>=start.time,]
    mass.data.xts[[islice]] = mass.data.xts[[islice]][index(mass.data.xts[[islice]])<=end.time,]    
}


time.index = seq(from=start.time,to=end.time,by="1 hour")
ntime = length(time.index)
simu.time = c(1:ntime-1)*3600
mass.gradient = rep(NA,ntime)

# slice.list = as.character(seq(314,330))
# nslice = length(slice.list)

# coord.data = read.table(fname_model_coord)
# rownames(coord.data) = coord.data[,1]
# coord.data =  coord.data[rownames(coord.data) %in% slice.list,]
# nwell = dim(coord.data)[1]
# y = coord.data[slice.list,3]
# names(y)=rownames(coord.data)
# x = coord.data[slice.list,2]
# names(x)=rownames(coord.data)

mass.level = array(NA,c(nslice,ntime))
rownames(mass.level) = slice.list
for (islice in slice.list) {
    mass.level[islice,] = mass.data.xts[[islice]][,"stage"]
}
available.date = which(colSums(mass.level,na.rm=TRUE)>200)

#-----------------------------smooth river stage-------------------------------##
if (is.smooth) {
  

# nwindows = 6 #hour
# nwindows = 24*1 #1d
nwindows = 24*7 #1 week
dt = 3600
filt = Ma(rep(1/nwindows,nwindows))
# new.mass.level = array(NA,c(nslice,(ntime+1)))
new.mass.level = array(NA,c(nslice,ntime+1)) #moving average (ma) add 1 extra time to match the dim(ma_value)
for (islice in 1:nslice)
{
    print(islice)
    ori_time = simu.time
    ori_value = mass.level[islice,]

    ma_value = filter(filt,ori_value)
    ma_time = ori_time-dt*(nwindows-1)/2 # ma_time offset by dt/2
    ma_value = tail(ma_value,-nwindows)
    ma_time = tail(ma_time,-nwindows)
    ma_value = c(ori_value[ori_time<min(ma_time)],ma_value)
    ma_time = c(ori_time[ori_time<min(ma_time)],ma_time)
    ma_value = c(ma_value,ori_value[ori_time>max(ma_time)])
    ma_time = c(ma_time,ori_time[ori_time>max(ma_time)])

    new.mass.level[islice,] = ma_value
}

##generate moving aveage plots with original mass data
# for (islice in 1:nslice) {
  islice = 1
  jpeg(paste("figures/mass_original_vs_mvAve_", slice.list[islice], "_",nwindows,"h.jpg", sep=''),width=8,height=5,units='in', res = 300)

  ori_time = ori_time + start.time
  plot(ori_time, mass.level[islice, 1:length(ori_time)] ,type = "l", col= "black", axes = F, xlab=NA,ylab="Hydaulic head (m)")
  box()

  axis(2,at=seq(118,128,2),mgp=c(5,0.7,0),cex.axis=1)

  axis.POSIXct(1,at=seq(as.Date("2007-01-01",tz="GMT"),
                        to=as.Date("2016-01-01",tz="GMT"),by="quarter"),format="%m/%Y",mgp=c(5,0.7,0))
  ma_time = ma_time + start.time
  lines(ma_time, new.mass.level[islice,], col= "red")


  legend("topright",legend = c("original","mvAve"), col = c("black", "red"), lty = c("solid", "solid"), bty = "n")
  title(paste("mass_original_vs_ma_", slice.list[islice], sep=''))
  dev.off()
# }




mass.level = new.mass.level
simu.time = ma_time
ntime = length(simu.time)

}
##------------------------calculate gradient--------------------------------##
# mass.gradient = array(NA,c(nslice,(ntime+1)))
mass.gradient.x = array(NA,c(nslice,ntime))
mass.gradient.y = array(NA,c(nslice,ntime))
rownames(mass.gradient.y) = slice.list
rownames(mass.gradient.x) = slice.list

for (islice in 1:(nslice-1)) #from top to bottom. 
{
  distance = sqrt((mass.coord[islice+1,
                              "northing"]-mass.coord[islice,"northing"])^2 +
                    (mass.coord[islice+1,"easting"]-mass.coord[islice,"easting"])^2)
  ## calculate grad based on x-direction
  mass.gradient.x[islice,] = (mass.level[islice+1,]-mass.level[islice,]
  )/distance*(mass.coord[islice+1,"easting"]-mass.coord[islice,"easting"])/distance   
  
  ## calculate grad based on y-direction    
  mass.gradient.y[islice,] = (mass.level[islice+1,]-mass.level[islice,]
  )/distance*(mass.coord[islice+1,"northing"]-mass.coord[islice,"northing"])/distance 
}

# gradient_314 is calcuated based on 315 and 314
for (islice in 1:(nslice-1))
{ 
    print(islice)
    Gradients = cbind(simu.time,
                      mass.gradient.x[islice,],
                      mass.gradient.y[islice,],
                      rep(0,(ntime)))
    
    DatumH = cbind(simu.time,
                   rep(mass.coord[islice,"easting"],ntime),
                   rep(mass.coord[islice,"northing"],ntime),                                      
                   mass.level[islice,])

    write.table(DatumH, file=paste(fname_DatumH, slice.list[islice],'.txt',sep=""),col.names=FALSE,row.names=FALSE) 
    write.table(Gradients, file=paste(fname_Gradients, slice.list[islice],".txt",sep=''),col.names=FALSE,row.names=FALSE)
    
    
}



