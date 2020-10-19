library(RSQLite)
library(momentuHMM)

drv = RSQLite::SQLite()
con <- dbConnect(drv, "/home/amdroy/MEGA/DATA/seabirdbank.db")

request = "SELECT DISTINCT
            trip.id as trip, gps.datetime, gps.lon, gps.lat
            FROM gps 
            INNER JOIN trip ON gps.trip = trip.id 
            INNER JOIN bird ON trip.bird = bird.id 
            WHERE trip.file_gps IS NOT NULL 
            AND substr(bird.fieldwork,1,1) = 'P' 
            AND bird.species = 'SV'"
  
gps <- dbGetQuery(con, request)

rawData <- gps[,c("trip","lon","lat")]
colnames(rawData) <- c("ID", "lon", "lat")
birdData <- prepData(data=rawData, type = "LL", coordNames = c("lon", "lat"))

## cluster K-Means for initialization
### STEP
clusterBird_step <- kmeans(na.omit(data.frame(birdData$step)), 3)
muS_1 <- max(clusterBird_step$centers)
muS_2 <- median(clusterBird_step$centers) 
muS_3 <- min(clusterBird_step$centers) 
sdS_1 <- sd(na.omit(birdData$step)[clusterBird_step[[1]] == which(clusterBird_step$centers == max(clusterBird_step$centers))])
sdS_2 <- sd(na.omit(birdData$step)[clusterBird_step[[1]] == which(clusterBird_step$centers == median(clusterBird_step$centers))])
sdS_3 <- sd(na.omit(birdData$step)[clusterBird_step[[1]] == which(clusterBird_step$centers == min(clusterBird_step$centers))])

### ANGLE
## for von mises
kappaA_1 <- 4
kappaA_2 <- 2
kappaA_3 <- 0.1

### ZERO MASS
zeroMass <- length(which(birdData$step == 0))/nrow(birdData) #we need to include zeroMass parameters

### FIT MODEL
stateNames <- c("fly","observe", "dive")
dist = list(step = "gamma", angle = "vm")
anglePar0 <- c(kappaA_1, kappaA_2, kappaA_3) 
m <- fitHMM(data = birdData, nbStates = 3, dist = dist, Par0 = list(step = stepPar0, angle = anglePar0), stateNames = stateNames, formula = ~1)

for (trip in unique(rawData$ID)){
  path = paste0('./figures/', trip, '.png')
  png(path)
  plot(m, animals = trip, ask = FALSE)
  dev.off()
}

save(m, file = 'hmm.RData')