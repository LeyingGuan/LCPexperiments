library(Rcpp)
library(LCP)
library(LCPexperiments)

set.seed(2021)
n <- 1000; n0 <- 1000
m <- 1000
iterations = 20
sim_name =commandArgs(trailingOnly = TRUE)



comparison_rets = list()
for(i in 1:iterations){
  data = sim_data_generator_1D_example1(sim_name = sim_name, n = n, n0 = n0, m = m)
  xtrain = data$xtrain
  ytrain = data$ytrain
  xcalibration = data$xcalibration
  ycalibration = data$ycalibration
  xtest = data$xtest
  ytest = data$ytest
  PItruth = data$truePI
  
  comparison_rets[[i]] = LCPcompare0(xtrain = xtrain, ytrain = ytrain, xcalibration = xcalibration,
                                    ycalibration = ycalibration, xtest = xtest, ytest = ytest, 
                                    alpha = 0.05,  quantiles = c(0.025,  0.1, 0.9 , 0.975), 
                                    nfolds = 3, random_state = 1, epochs = 35,
                                    save_path = paste0('synthetic_results/example1/',sim_name), print_out = 10)
  comparison_rets[[i]]$PItruth = PItruth
  comparison_rets[[i]]$xtest = xtest
  comparison_rets[[i]]$ytest = ytest
  tmp = comparison_rets[[i]]$PIbands0[,2,] - comparison_rets[[i]]$PIbands0[,1,]
  print(apply(tmp,2,function(z) mean(z==Inf)))
  print(apply(tmp,2,function(z) mean(z[abs(xtest) <=1.96])))
}


saveRDS(comparison_rets, file = paste0('synthetic_results/example1_', sim_name,".rds"))
