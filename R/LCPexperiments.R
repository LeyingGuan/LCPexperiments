library(LCP)
#' LCPcompare - A wrapper for comparing different four constructions
#' using CP/LCP for the simulated and UCI data using default distance construction
#' and auto-tuned h.
#' imports:
#'@import R6
#'@import Rcpp
#'@import torch
#'@import torchvision
#'@import XRPython
#'@import LCP
#'@importFrom torch optim_adam
#'@export
LCPcompare <- function(xtrain, ytrain, xcalibration, ycalibration,
                       xtest, ytest, alpha = 0.05, 
                       quantiles = c(0.025, 0.05, 0.1, 0.9 ,0.95, 0.975,), 
                       nfolds = 3, random_state = 1,
                       save_path = NULL, print_out = 10, epochs = 50){
  hidden_size = 32; lr = 0.005; wd = 1e-6; test_ratio = 0.2;  drop_out = 0.2;
  qlow = 0.025; qhigh = 0.975; use_arrangement = FALSE;  batch_size = 64
  batch_size = 32; optimizer_class = optim_adam;
  p = ncol(xtrain)
  loss_func = nn_mse_loss()
  coverages = rep(NA, 8)
  lens = rep(NA, 8)
  Inflens = rep(NA, 8)
  names(coverages) <- names(lens)<- names(Inflens) <- c("CR", "LCR", "CLR", "LCLR", "CQR",   "LCQR", "CQLR", "LCQLR")
  PIbands = array(0, dim = c(nrow(xtest),2,8))
  synthetic_mse = learnerOptimizer$new(network_new = my_neuralnet, optimizer_class = optimizer_class,
                                       loss_type = "mse",
                                       loss_func = loss_func$forward,all_quantiles = loss_func$quantiles, device = 'cpu',
                                       test_ratio = 0.2,  CV = TRUE, nfolds = nfolds, random_state = random_state,
                                       save_path = save_path,
                                       hidden_size = hidden_size, in_shape = p, drop_out = drop_out,
                                       lr = lr, wd = wd, qlow = qlow, qhigh = qhigh,
                                       use_arrangement = use_arrangement, 
                                       my_test_split_func = my_test_split_func,
                                       my_cv_split_func = my_cv_split_func,
                                       my_epoch_internal_train =my_epoch_internal_train)
  synthetic_mse$fit(x = xtrain, y =ytrain, epochs = epochs, batch_size = batch_size , print_out = print_out)
  requires_grad  =  F
  cv_ret_mse = synthetic_mse$cv_evaluate(my_predict_func = my_predict_func, my_cv_evaluate = my_cv_evaluate_func,
                                         num_classes = 1, requires_grad = requires_grad)
  yhat_cal = as.array(synthetic_mse$predict(xcalibration, my_predict_func = my_predict_func, requires_grad = requires_grad)$yhat)
  
  yhat_te = as.array(synthetic_mse$predict(xtest, my_predict_func = my_predict_func, requires_grad = requires_grad)$yhat)
  
  r = ytrain[,1]-cv_ret_mse$yhat[,1]
  rcal = yhat_cal[,1] - ycalibration[,1]
  rte = yhat_te[,1] - ytest[,1]
  synthetic_var = learnerOptimizer$new(network_new = my_neuralnet, optimizer_class = optimizer_class,
                                       loss_type = "mse",
                                       loss_func = loss_func$forward,all_quantiles = NULL, device = 'cpu',
                                       test_ratio = 0.2,  CV = TRUE, nfolds = nfolds, random_state = random_state,
                                       save_path = save_path,
                                       hidden_size = hidden_size, in_shape = p, drop_out = drop_out,
                                       lr = lr, wd = wd, qlow = qlow, qhigh = qhigh,
                                       use_arrangement = use_arrangement, 
                                       my_test_split_func = my_test_split_func,
                                       my_cv_split_func = my_cv_split_func,
                                       my_epoch_internal_train =my_epoch_internal_train)
  ysd = matrix(log(r^2+mean(r^2)),ncol = 1)
  synthetic_var$foldid = synthetic_mse$foldid
  synthetic_var$fit(x = xtrain, y = ysd, epochs = epochs, batch_size = batch_size , print_out = print_out)
  requires_grad  =  T
  cv_ret_var = synthetic_var$cv_evaluate(my_predict_func = my_predict_func, my_cv_evaluate = my_cv_evaluate_func,
                                         num_classes = 1, requires_grad = requires_grad)  
  calibration_ret_var =  synthetic_var$predict(xcalibration, my_predict_func = my_predict_func, requires_grad = requires_grad)
  test_ret_var =  synthetic_var$predict(xtest, my_predict_func = my_predict_func, requires_grad = requires_grad)
  
  
  estimated_sds = list()
  observed_sds = list()
  observed_sds[[1]] = abs(r);   observed_sds[[2]] = abs(rcal);   observed_sds[[3]] = abs(rte);
  estimated_sds[[1]] =  sqrt(exp(cv_ret_var$yhat[,1])); 
  estimated_sds[[2]] = sqrt(exp(as.array(calibration_ret_var$yhat[,1])))
  estimated_sds[[3]] = sqrt(exp(as.array(test_ret_var$yhat[,1])))

  #CR, CLR
  deltaCP = quantile(observed_sds[[2]] , 1-alpha)
  lens[1] =deltaCP*2
  coverages[1] = mean( observed_sds[[3]]<=deltaCP)
  PIbands[,1,1] = yhat_te[,1]-deltaCP
  PIbands[,2,1] = yhat_te[,1]+deltaCP
  
  deltaCP =  quantile(observed_sds[[2]]/estimated_sds[[2]] , 1-alpha)
  lens[3] =mean(deltaCP*estimated_sds[[3]]*2)
  coverages[3] = mean( observed_sds[[3]]<=(deltaCP * estimated_sds[[3]]))
  PIbands[,1,3] = yhat_te[,1]-deltaCP* estimated_sds[[3]]
  PIbands[,2,3] = yhat_te[,1]+deltaCP* estimated_sds[[3]]
  # LCR, LCLR
  # extract two dimensions
  tangent_cv =  cv_ret_var$jacobians[,1,]
  tangent_cal =  calibration_ret_var$jacobians[,1,]
  tangent_test = test_ret_var$jacobians[,1,]
  if(is.null(dim(tangent_cv))){
    utangent = 1
    uorthogonal = NULL
  }else if(sum(tangent_cv^2) == 0){
    utangent = NULL
    uorthogonal = diag(rep(1, p))
  }else{
    tmp = svd(tangent_cv)
    utangent = tmp$v[,1]
    uorthogonal = tmp$v[,-1]
  }
  Hlists = LCPdefault_distance(utangent, uorthogonal, estimated_sds, xtrain, xcalibration, xtest)
  s0 = mean(Hlists$Hcv); max0 = max(Hlists$Hcv)*2; min0 =quantile(Hlists$Hcv, 0.01)
  hs = exp(seq(log(min0), log(max0), length.out = 20))
  
  Vcal = observed_sds[[2]]
  Vcv = observed_sds[[1]]
  order1 = order(Vcal)
  
  myLCR = LCPmodule$new(H = Hlists$H[order1, order1], V = Vcal[order1], h = 2, alpha = alpha, type = "distance")
  
  t1 = Sys.time()
  auto_ret = myLCR$LCP_auto_tune(V0 =Vcv, H0 =Hlists$Hcv, hs = hs, B = 2, delta =alpha/2, lambda = 1, trace = TRUE)
  t2 = Sys.time()
  print("finish LCR auto-tuning")
  print(t2 - t1)
  
  myLCR$h = auto_ret$h
  
  myLCR$lower_idx()
  myLCR$cumsum_unnormalized()
  myLCR$LCP_construction(Hnew = Hlists$Hnew[,order1], HnewT = Hlists$HnewT[order1,])
  
  deltaLCP = myLCR$band_V
  qL = yhat_te-deltaLCP
  qU = yhat_te+deltaLCP
  coverages[2] =mean(observed_sds[[3]]<=deltaLCP)
  lens[2] = mean(deltaLCP[deltaLCP< Inf])*2
  Inflens[2] = mean(deltaLCP== Inf)
  PIbands[,1,2] = yhat_te[,1]-deltaLCP
  PIbands[,2,2] = yhat_te[,1]+deltaLCP
  
  
  Vcal = observed_sds[[2]]/estimated_sds[[2]]
  Vcv = observed_sds[[1]]/estimated_sds[[1]]
  order1 = order(Vcal)
  
  myLCR = LCPmodule$new(H = Hlists$H[order1, order1], V = Vcal[order1], h = 2, alpha = alpha, type = "distance")
  
  t1 = Sys.time()
  auto_ret = myLCR$LCP_auto_tune(V0 =Vcv, H0 =Hlists$Hcv, hs = hs, B = 2, delta=alpha/2, lambda = 1, trace = TRUE)
  t2 = Sys.time()
  print("finish LCR auto-tuning")
  print(t2 - t1)
  
  myLCR$h = auto_ret$h
  
  myLCR$lower_idx()
  myLCR$cumsum_unnormalized()
  myLCR$LCP_construction(Hnew = Hlists$Hnew[,order1], HnewT = Hlists$HnewT[order1,])
  
  deltaLCP = myLCR$band_V
  qL = yhat_te-deltaLCP * estimated_sds[[3]]
  qU = yhat_te+deltaLCP * estimated_sds[[3]]
  coverages[4] =mean(observed_sds[[3]]<=(deltaLCP * estimated_sds[[3]]))
  lens[4] = mean((deltaLCP * estimated_sds[[3]])[deltaLCP< Inf])*2
  Inflens[4] = mean(deltaLCP== Inf)
  
  PIbands[,1,4] = yhat_te[,1]-deltaLCP* estimated_sds[[3]]
  PIbands[,2,4] = yhat_te[,1]+deltaLCP* estimated_sds[[3]]
  
  ##quantile score construction
  
  loss_func = all_quantile_loss$new(quantiles)
  
  synthetic_qc = learnerOptimizer$new(network_new = my_neuralnet, optimizer_class = optimizer_class,
                                      loss_type = "qc",
                                      loss_func = loss_func$forward,all_quantiles = loss_func$quantiles, device = 'cpu',
                                      test_ratio = 0.2,  CV = TRUE, nfolds = nfolds, random_state = random_state,
                                      save_path = save_path,
                                      hidden_size = hidden_size, in_shape = p, drop_out = drop_out,
                                      lr = lr, wd = wd, qlow = qlow, qhigh = qhigh,
                                      use_arrangement = use_arrangement, 
                                      my_test_split_func = my_test_split_func,
                                      my_cv_split_func = my_cv_split_func,
                                      my_epoch_internal_train =my_epoch_internal_train)
  synthetic_qc$fit(x = xtrain, y =ytrain, epochs = epochs, batch_size = batch_size , print_out = print_out)
  requires_grad = F
  cv_ret_qc = synthetic_qc$cv_evaluate(my_predict_func = my_predict_func, my_cv_evaluate = my_cv_evaluate_func,
                                       num_classes = length(quantiles), requires_grad = requires_grad)
  
  test_ret_qc = synthetic_qc$predict(xtest, my_predict_func = my_predict_func, requires_grad = requires_grad)
  
  calibration_ret_qc = synthetic_qc$predict(xcalibration , my_predict_func = my_predict_func, requires_grad = requires_grad)
  
  
  #quntile conformal
  idx1 = which(quantiles == (alpha/2))
  idx2 = which(quantiles == (1-alpha/2))
  V1cv = as.array(cv_ret_qc$yhat[,idx1]) - ytrain[,1]
  V2cv = ytrain[,1] - as.array(cv_ret_qc$yhat[,idx2])
  Vcv = ifelse(V1cv >=V2cv, V1cv, V2cv)
  loss_func = nn_mse_loss()
  synthetic_var_qc = learnerOptimizer$new(network_new = my_neuralnet, optimizer_class = optimizer_class,
                                          loss_type = "mse",
                                          loss_func = loss_func$forward,all_quantiles = NULL, device = 'cpu',
                                          test_ratio = 0.2,  CV = TRUE, nfolds = nfolds, random_state = random_state,
                                          save_path = save_path,
                                          hidden_size = hidden_size, in_shape = p, drop_out = drop_out,
                                          lr = lr, wd = wd, qlow = qlow, qhigh = qhigh,
                                          use_arrangement = use_arrangement, 
                                          my_test_split_func = my_test_split_func,
                                          my_cv_split_func = my_cv_split_func,
                                          my_epoch_internal_train =my_epoch_internal_train)
  
  
  
  ysd_qc = matrix(log(Vcv^2+mean(Vcv^2)),ncol = 1)
  synthetic_var_qc$foldid = synthetic_qc$foldid
  synthetic_var_qc$fit(x = xtrain, y = ysd_qc, epochs = epochs, batch_size = batch_size , print_out = print_out)

  #quntile conformal
  V1cal = as.array(calibration_ret_qc$yhat[,idx1]) - ycalibration[,1]
  V2cal = ycalibration[,1] - as.array(calibration_ret_qc$yhat[,idx2])
  Vcal = ifelse(V1cal >=V2cal, V1cal, V2cal)
  deltaCP = quantile(Vcal, .95)
  qL = as.array(test_ret_qc$yhat[,idx1]) -deltaCP
  qU = as.array(test_ret_qc$yhat[,idx2])+deltaCP
  coverages[5] =mean(ytest[,1]<= qU & ytest[,1] >=qL)
  lens[5] = mean(qU - qL)
  PIbands[,1,5] = qL
  PIbands[,2,5] = qU
  #quantile conformal + local
  requires_grad = T
  cv_ret_var_qc = synthetic_var_qc$cv_evaluate(my_predict_func = my_predict_func, my_cv_evaluate = my_cv_evaluate_func,
                                               num_classes = 1, requires_grad = requires_grad)
  
  test_ret_var_qc = synthetic_qc$predict(xtest, my_predict_func = my_predict_func, requires_grad = requires_grad)
  
  calibration_ret_var_qc = synthetic_qc$predict(xcalibration , my_predict_func = my_predict_func, requires_grad = requires_grad)
  
  V1cv = as.array(cv_ret_qc$yhat[,idx1]) - ytrain[,1]
  V2cv = ytrain[,1] - as.array(cv_ret_qc$yhat[,idx2])
  Vcv = ifelse(V1cv >=V2cv, V1cv, V2cv)
  
  observed_sds[[1]] = Vcv;   observed_sds[[2]] = Vcal;  
  estimated_sds[[1]] =  sqrt(exp(cv_ret_var_qc$yhat[,1])); 
  estimated_sds[[2]] = sqrt(exp(as.array(calibration_ret_var_qc$yhat[,1])))
  estimated_sds[[3]] = sqrt(exp(as.array(test_ret_var_qc$yhat[,1])))
  
  deltaCP =  quantile(observed_sds[[2]]/estimated_sds[[2]] , 1-alpha)
  qL = as.array(test_ret_qc$yhat[,idx1]) - deltaCP * estimated_sds[[3]]
  qU =  as.array(test_ret_qc$yhat[,idx2]) + deltaCP * estimated_sds[[3]]
  coverages[7] =mean(ytest[,1]<= qU & ytest[,1] >=qL)
  lens[7] = mean(qU - qL)
  PIbands[,1,7] = qL
  PIbands[,2,7] = qU
  tangent_cv =  cv_ret_var_qc$jacobians[,1,]
  tangent_cal =  calibration_ret_var_qc$jacobians[,1,]
  tangent_test = test_ret_var_qc$jacobians[,1,]
  if(is.null(dim(tangent_cv))){
    utangent = 1
    uorthogonal = NULL
  }else if(sum(tangent_cv^2) == 0){
    utangent = NULL
    uorthogonal = diag(rep(1, p))
  }else{
    tmp = svd(tangent_cv)
    utangent = tmp$v[,1]
    uorthogonal = tmp$v[,-1]
  }
  Hlists = LCPdefault_distance(utangent, uorthogonal, estimated_sds, xtrain, xcalibration, xtest)
  s0 = mean(Hlists$Hcv); max0 = max(Hlists$Hcv)*2; min0 =quantile(Hlists$Hcv, 0.01)
  hs = exp(seq(log(min0), log(max0), length.out = 20))
  
  Vcal = observed_sds[[2]]#/estimated_sds[[2]]
  Vcv = observed_sds[[1]]#/estimated_sds[[1]]
  order1 = order(Vcal)
  
  myLCR = LCPmodule$new(H = Hlists$H[order1, order1], V = Vcal[order1], h = 2, alpha = alpha, type = "distance")
  
  t1 = Sys.time()
  auto_ret = myLCR$LCP_auto_tune(V0 =Vcv, H0 =Hlists$Hcv, hs = hs, B = 2, delta =alpha/2, lambda = 1, trace = TRUE)
  t2 = Sys.time()
  print("finish LCR auto-tuning")
  print(t2 - t1)
  
  myLCR$h = auto_ret$h
  
  myLCR$lower_idx()
  myLCR$cumsum_unnormalized()
  myLCR$LCP_construction(Hnew = Hlists$Hnew[,order1], HnewT = Hlists$HnewT[order1,])
  
  deltaLCP = myLCR$band_V
  qL = as.array(test_ret_qc$yhat[,idx1])-deltaLCP
  qU = as.array(test_ret_qc$yhat[,idx2])+deltaLCP
  coverages[6] =mean(ytest[,1]<= qU & ytest[,1] >=qL)
  lens[6] = mean((qU - qL)[deltaLCP <Inf])
  Inflens[6] = mean(deltaLCP == Inf)
  PIbands[,1,6] = qL
  PIbands[,2,6] = qU
  
  Vcal = observed_sds[[2]]/estimated_sds[[2]]
  Vcv = observed_sds[[1]]/estimated_sds[[1]]
  order1 = order(Vcal)
  
  myLCR = LCPmodule$new(H = Hlists$H[order1, order1], V = Vcal[order1], h = 2, alpha = alpha, type = "distance")
  
  t1 = Sys.time()
  auto_ret = myLCR$LCP_auto_tune(V0 =Vcv, H0 =Hlists$Hcv, hs = hs, B = 2, delta =alpha/2, lambda = 1, trace = TRUE)
  t2 = Sys.time()
  print("finish LCR auto-tuning")
  print(t2 - t1)
  
  myLCR$h = auto_ret$h
  
  myLCR$lower_idx()
  myLCR$cumsum_unnormalized()
  myLCR$LCP_construction(Hnew = Hlists$Hnew[,order1], HnewT = Hlists$HnewT[order1,])
  
  deltaLCP = myLCR$band_V
  qL = as.array(test_ret_qc$yhat[,idx1])-deltaLCP * estimated_sds[[3]]
  qU = as.array(test_ret_qc$yhat[,idx2])+deltaLCP * estimated_sds[[3]]
  coverages[8] =mean(ytest[,1]<= qU & ytest[,1] >=qL)
  lens[8] = mean((qU - qL)[deltaLCP <Inf])
  Inflens[8] = mean(deltaLCP == Inf)
  PIbands[,1,8] = qL
  PIbands[,2,8] = qU
  return(list(coverages = coverages, lens = lens, Infpercent = Inflens,
              PIbands = PIbands))
}



#' LCPcompare0 - A wrapper for comparing CP/LCP using regression
#' imports:
#'@import R6
#'@import Rcpp
#'@import torch
#'@import torchvision
#'@import XRPython
#'@import LCP
#'@export
LCPcompare0 <- function(xtrain, ytrain, xcalibration, ycalibration,
                       xtest, ytest, alpha = 0.05, 
                       quantiles = c(0.025, 0.05, 0.1, 0.9 ,0.95, 0.975,), 
                       nfolds = 3, random_state = 1,
                       save_path = NULL, print_out = 10, epochs = 50){
  hidden_size = 32; lr = 0.005; wd = 1e-6; test_ratio = 0.2;  drop_out = 0.2;
  qlow = 0.025; qhigh = 0.975; use_arrangement = FALSE;  batch_size = 64
  batch_size = 32; optimizer_class = optim_adam;
  p = ncol(xtrain)
  loss_func = nn_mse_loss()
  coverages = rep(NA, 3)
  lens = rep(NA, 3)
  Inflens = rep(NA, 3)
  names(coverages) <- names(lens)<- names(Inflens) <- c("CR", "LCR", "CLR")
  PIbands = array(0, dim = c(nrow(xtest),2,3))
  synthetic_mse = learnerOptimizer$new(network_new = my_neuralnet, optimizer_class = optimizer_class,
                                       loss_type = "mse",
                                       loss_func = loss_func$forward,all_quantiles = loss_func$quantiles, device = 'cpu',
                                       test_ratio = 0.2,  CV = TRUE, nfolds = nfolds, random_state = random_state,
                                       save_path = save_path,
                                       hidden_size = hidden_size, in_shape = p, drop_out = drop_out,
                                       lr = lr, wd = wd, qlow = qlow, qhigh = qhigh,
                                       use_arrangement = use_arrangement, 
                                       my_test_split_func = my_test_split_func,
                                       my_cv_split_func = my_cv_split_func,
                                       my_epoch_internal_train =my_epoch_internal_train)
  synthetic_mse$fit(x = xtrain, y =ytrain, epochs = epochs, batch_size = batch_size , print_out = print_out)
  requires_grad  =  F
  cv_ret_mse = synthetic_mse$cv_evaluate(my_predict_func = my_predict_func, my_cv_evaluate = my_cv_evaluate_func,
                                         num_classes = 1, requires_grad = requires_grad)
  yhat_cal = as.array(synthetic_mse$predict(xcalibration, my_predict_func = my_predict_func, requires_grad = requires_grad)$yhat)
  
  yhat_te = as.array(synthetic_mse$predict(xtest, my_predict_func = my_predict_func, requires_grad = requires_grad)$yhat)
  
  r = ytrain[,1]-cv_ret_mse$yhat[,1]
  rcal = yhat_cal[,1] - ycalibration[,1]
  rte = yhat_te[,1] - ytest[,1]
  synthetic_var = learnerOptimizer$new(network_new = my_neuralnet, optimizer_class = optimizer_class,
                                       loss_type = "mse",
                                       loss_func = loss_func$forward,all_quantiles = NULL, device = 'cpu',
                                       test_ratio = 0.2,  CV = TRUE, nfolds = nfolds, random_state = random_state,
                                       save_path = save_path,
                                       hidden_size = hidden_size, in_shape = p, drop_out = drop_out,
                                       lr = lr, wd = wd, qlow = qlow, qhigh = qhigh,
                                       use_arrangement = use_arrangement, 
                                       my_test_split_func = my_test_split_func,
                                       my_cv_split_func = my_cv_split_func,
                                       my_epoch_internal_train =my_epoch_internal_train)
  ysd = matrix(log(r^2+mean(r^2)),ncol = 1)
  synthetic_var$foldid = synthetic_mse$foldid
  synthetic_var$fit(x = xtrain, y = ysd, epochs = epochs, batch_size = batch_size , print_out = print_out)
  requires_grad  =  T
  cv_ret_var = synthetic_var$cv_evaluate(my_predict_func = my_predict_func, my_cv_evaluate = my_cv_evaluate_func,
                                         num_classes = 1, requires_grad = requires_grad)  
  calibration_ret_var =  synthetic_var$predict(xcalibration, my_predict_func = my_predict_func, requires_grad = requires_grad)
  test_ret_var =  synthetic_var$predict(xtest, my_predict_func = my_predict_func, requires_grad = requires_grad)
  
  
  estimated_sds = list()
  observed_sds = list()
  observed_sds[[1]] = abs(r);   observed_sds[[2]] = abs(rcal);   observed_sds[[3]] = abs(rte);
  estimated_sds[[1]] =  sqrt(exp(cv_ret_var$yhat[,1])); 
  estimated_sds[[2]] = sqrt(exp(as.array(calibration_ret_var$yhat[,1])))
  estimated_sds[[3]] = sqrt(exp(as.array(test_ret_var$yhat[,1])))
  
  #CR, CLR
  deltaCP = quantile(observed_sds[[2]] , 1-alpha)
  lens[1] =deltaCP*2
  coverages[1] = mean( observed_sds[[3]]<=deltaCP)
  PIbands[,1,1] = yhat_te[,1]-deltaCP
  PIbands[,2,1] = yhat_te[,1]+deltaCP
  
  deltaCP =  quantile(observed_sds[[2]]/estimated_sds[[2]] , 1-alpha)
  lens[3] =mean(deltaCP*estimated_sds[[3]]*2)
  coverages[3] = mean( observed_sds[[3]]<=(deltaCP * estimated_sds[[3]]))
  PIbands[,1,3] = yhat_te[,1]-deltaCP* estimated_sds[[3]]
  PIbands[,2,3] = yhat_te[,1]+deltaCP* estimated_sds[[3]]
  # LCR, LCLR
  # extract two dimensions
  tangent_cv =  cv_ret_var$jacobians[,1,]
  tangent_cal =  calibration_ret_var$jacobians[,1,]
  tangent_test = test_ret_var$jacobians[,1,]
  if(is.null(dim(tangent_cv))){
    utangent = 1
    uorthogonal = NULL
  }else if(sum(tangent_cv^2) == 0){
    utangent = NULL
    uorthogonal = diag(rep(1, p))
  }else{
    tmp = svd(tangent_cv)
    utangent = tmp$v[,1]
    uorthogonal = tmp$v[,-1]
  }
  Hlists = LCPdefault_distance(utangent, uorthogonal, estimated_sds, xtrain, xcalibration, xtest)
  s0 = mean(Hlists$Hcv); max0 = max(Hlists$Hcv)*2; min0 =quantile(Hlists$Hcv, 0.01)
  hs = exp(seq(log(min0), log(max0), length.out = 20))
  
  coverages0 = rep(NA, length(hs))
  lens0 = rep(NA, length(hs))
  Inflens0 = rep(NA, length(hs))
  PIbands0 = array(0, dim = c(nrow(xtest),2,length(hs)))
  
  Vcal = observed_sds[[2]]
  Vcv = observed_sds[[1]]
  order1 = order(Vcal)
  
  myLCR = LCPmodule$new(H = Hlists$H[order1, order1], V = Vcal[order1], h = 2, alpha = alpha, type = "distance")
  
  t1 = Sys.time()
  auto_ret = myLCR$LCP_auto_tune(V0 =Vcv, H0 =Hlists$Hcv, hs = hs, B = 2, delta =alpha/2, lambda = 1, trace = TRUE)
  t2 = Sys.time()
  print("finish LCR auto-tuning")
  print(t2 - t1)
  myLCR$h = auto_ret$h
  
  myLCR$lower_idx()
  myLCR$cumsum_unnormalized()
  myLCR$LCP_construction(Hnew = Hlists$Hnew[,order1], HnewT = Hlists$HnewT[order1,])
  
  deltaLCP = myLCR$band_V
  qL = yhat_te-deltaLCP
  qU = yhat_te+deltaLCP
  coverages[2] =mean(observed_sds[[3]]<=deltaLCP)
  lens[2] = mean(deltaLCP[deltaLCP< Inf])*2
  Inflens[2] = mean(deltaLCP== Inf)
  PIbands[,1,2] = yhat_te[,1]-deltaLCP
  PIbands[,2,2] = yhat_te[,1]+deltaLCP
  
  ##different hs
  for(k in 1:length(hs)){
    myLCR$h = hs[k]
    myLCR$lower_idx()
    myLCR$cumsum_unnormalized()
    myLCR$LCP_construction(Hnew = Hlists$Hnew[,order1], HnewT = Hlists$HnewT[order1,])
    
    deltaLCP = myLCR$band_V
    qL = yhat_te-deltaLCP
    qU = yhat_te+deltaLCP
    
    coverages0[k] =mean(observed_sds[[3]]<=deltaLCP)
    lens0[k] = mean(deltaLCP[deltaLCP< Inf])*2
    Inflens0[k] = mean(deltaLCP== Inf)
    PIbands0[,1,k] = yhat_te[,1]-deltaLCP
    PIbands0[,2,k] = yhat_te[,1]+deltaLCP
  }
  
  

  return(list(coverages = coverages, lens = lens, Infpercent = Inflens,
              PIbands = PIbands, coverages0 = coverages0, lens0 = lens0,
              PIbands0 = PIbands0, hs = hs))
}


#' sim_data_generator_1D_example2 - generate 1D simulated data for example2
#'@export
sim_data_generator_1D_example2 <- function(sim_name, n = 1000, n0 = 1000, m = 1000, alpha = 0.05){
  if(sim_name == "1D_setA"){
    #sin
    noise_generating = function(x){
      abs(sin(x))
    }
  }else if(sim_name == "1D_setB"){
    #cos
    noise_generating = function(x){
      abs(cos(x))
    }
    
  }else if(sim_name == "1D_setC"){
    #linear
    noise_generating = function(x){
      abs(x)
    }

  }else if(sim_name == "1D_setD"){
    #constant
    noise_generating = function(x){
      rep(1,length(x))
    }
  }
  x <- runif(n,-2,2); 
  s <- noise_generating(x); 
  y <- s*rnorm(n);
  x1 <- runif(m,-2,2); x0 <-runif(n0,-2,2)
  x1 <- x1[order(x1)];
  s1 <-noise_generating(x1); s0 = noise_generating(x0)
  y1 = s1*rnorm(m); y0 = s0 *rnorm(n0)
  xtrain = matrix(x, ncol = 1)
  xcalibration = matrix(x0, ncol = 1)
  xtest = matrix(x1, ncol = 1)
  ytrain = matrix(y, ncol = 1)
  ycalibration = matrix(y0, ncol = 1)
  ytest = matrix(y1, ncol = 1)
  truePI = matrix(0, ncol = 2, nrow = m)
  truePI[,2] = qnorm(1-alpha/2)*s1
  truePI[,1] = qnorm(alpha/2)*s1
  return(list(xtrain = xtrain, ytrain = ytrain,
              xcalibration = xcalibration, ycalibration = ycalibration,
              xtest = xtest, ytest = ytest, truePI = truePI))
}

#' sim_data_generator_1D_example1 - generate 1D simulated data for example1
#'@export
sim_data_generator_1D_example1 <- function(sim_name, n = 1000, n0 = 1000, m = 1000, alpha = 0.05){
  if(sim_name == "1D_setA"){
    #sin
    noise_generating = function(x){
      abs(sin(x))
    }
  }else if(sim_name == "1D_setB"){
    #cos
    noise_generating = function(x){
      abs(cos(x))
    }
    
  }else if(sim_name == "1D_setC"){
    #linear
    noise_generating = function(x){
      abs(x)
    }
    
  }else if(sim_name == "1D_setD"){
    #constant
    noise_generating = function(x){
      rep(1,length(x))
    }
  }
  x <- rnorm(n = n, mean = 0, sd = 1) 
  s <- noise_generating(x); 
  y <- s*rnorm(n);
  x1 <- rnorm(n = m, mean = 0, sd = 1) ; x0 <- rnorm(n = n0, mean = 0, sd = 1) 
  x1 <- x1[order(x1)];
  s1 <-noise_generating(x1); s0 = noise_generating(x0)
  y1 = s1*rnorm(m); y0 = s0 *rnorm(n0)
  xtrain = matrix(x, ncol = 1)
  xcalibration = matrix(x0, ncol = 1)
  xtest = matrix(x1, ncol = 1)
  ytrain = matrix(y, ncol = 1)
  ycalibration = matrix(y0, ncol = 1)
  ytest = matrix(y1, ncol = 1)
  truePI = matrix(0, ncol = 2, nrow = m)
  truePI[,2] = qnorm(1-alpha/2)*s1
  truePI[,1] = qnorm(alpha/2)*s1
  return(list(xtrain = xtrain, ytrain = ytrain,
              xcalibration = xcalibration, ycalibration = ycalibration,
              xtest = xtest, ytest = ytest, truePI = truePI))
}


