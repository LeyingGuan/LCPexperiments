library(Rcpp)
library(LCP)
library(LCPexperiments)

set.seed(2021)

data_path ="datasets/"
data  = read.csv(paste0(data_path,"facebook/Training/Features_Variant_1.csv"), header = F)
data = scale(data)
tmp = apply(is.na(data),2,sum)
data = data[,tmp == 0]
n = 5000

iterations = 20
comparison_rets = list()
for(i in 1:iterations){
  print(paste0("###############", i, "##############"))
  j0 = ncol(data)
  lltrain = sample(1:nrow(data), n)
  llcal = sample(setdiff(1:nrow(data), lltrain),n)
  llte = sample(setdiff(1:nrow(data), c(llcal, lltrain)),5000)
  data_tr = data[lltrain,]
  data_cal = data[llcal,]
  data_te = data[llte,]
  
  xtrain = as.matrix(data_tr[,-j0])
  ytrain = matrix(data_tr[,j0], ncol = 1)
  xtest = as.matrix(data_te[,-j0])
  ytest = matrix(data_te[,j0], ncol = 1)
  xcalibration = as.matrix(data_cal[,-j0])
  ycalibration = matrix(data_cal[,j0], ncol = 1)
  
  comparison_rets[[i]] = LCPcompare(xtrain = xtrain, ytrain = ytrain, xcalibration = xcalibration,
                                    ycalibration = ycalibration, xtest = xtest, ytest = ytest, 
                                    alpha = 0.05,  quantiles = c(0.025,  0.1, 0.9 , 0.975), 
                                    nfolds = 3, random_state = 1, epochs = 50,
                                    save_path = paste0('UCI_results/facebook/'), print_out = 10)
  print( comparison_rets[[i]]$lens)
  saveRDS(comparison_rets, file = paste0("UCI_results/facebook_LCPcomparison_results.rds"))
}


comparison_rets=readRDS("UCI_results/facebook_LCPcomparison_results.rds")
coverages =matrix(0, ncol = 8, nrow = length(comparison_rets ))
lens =matrix(0, ncol = 8, nrow = length(comparison_rets ))
Infpercent =matrix(0, ncol = 8, nrow = length(comparison_rets ))
for(i in 1:length(comparison_rets)){
  coverages[i,] = comparison_rets[[i]]$coverages
  lens[i,] = comparison_rets[[i]]$lens
  Infpercent[i,] = comparison_rets[[i]]$Infpercent
}
colnames(coverages) <- colnames(lens) <- colnames(Infpercent) <- names(comparison_rets[[i]]$coverages)

apply(coverages,2,mean)
apply(lens,2,mean)
apply(lens,2,median)
apply(Infpercent,2,mean)

apply(coverages,2,sd)

apply(lens,2,sd)

