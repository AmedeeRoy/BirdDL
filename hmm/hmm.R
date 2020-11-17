library(momentuHMM)

data_train = read.table("./data_train.csv", sep = ',', header = TRUE)
data_test = read.table("./data_test.csv", sep = ',', header = TRUE)

rawData <- data_train[,c("trip","lon","lat")]
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
    stepPar0 <- c(muS_1, muS_2, muS_3, sdS_1, sdS_2, sdS_3, zeroMass, zeroMass, zeroMass) 
    m <- fitHMM(data = birdData, nbStates = 3, dist = dist, Par0 = list(step = stepPar0, angle = anglePar0), stateNames = stateNames, formula = ~1)

for (trip in unique(rawData$ID)){
  path = paste0('./figures/', trip, '.png')
  png(path)
  plot(m, animals = trip, ask = FALSE)
  dev.off()
}

save(m, file = 'hmm.RData')


predict = viterbi(m)
data_train$hmm = predict

data_train$dive <- NA
for (k in unique(data_train$trip)){
  p = data_train$pressure[data_train$trip == k]
  dive = 1*(median(p)+1 < p)
  data_train$dive[data_train$trip == k] <- dive
}

real = data_train$dive
estim = 1*(data_train$hmm==3)


(mean(estim[real == 1]) + mean(1-estim[real != 1]))/2
# [1] 0.7433632
mean(estim[real == 1])
# [1] 0.5988665
mean(1-estim[real != 1])
# [1] 0.8878599
mean(real == estim)
# [1] 0.8844036

mean(real == 0)
#   [1] 0.9880405