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
  data = sim_data_generator_1D_example2(sim_name = sim_name, n = n, n0 = n0, m = m)
  xtrain = data$xtrain
  ytrain = data$ytrain
  xcalibration = data$xcalibration
  ycalibration = data$ycalibration
  xtest = data$xtest
  ytest = data$ytest
  PItruth = data$truePI
  
  comparison_rets[[i]] = LCPcompare(xtrain = xtrain, ytrain = ytrain, xcalibration = xcalibration,
                                    ycalibration = ycalibration, xtest = xtest, ytest = ytest, 
                                    alpha = 0.05,  quantiles = c(0.025,  0.1, 0.9 , 0.975), 
                                    nfolds = 3, random_state = 1, epochs = 35,
                                    save_path = paste0('synthetic_results/example2/',sim_name), print_out = 10)
  comparison_rets[[i]]$PItruth = PItruth
}


saveRDS(comparison_rets, file = paste0('synthetic_results/example2_', sim_name,".rds"))
