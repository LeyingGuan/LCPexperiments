#neural network pytorch
library(torch)
library(torchvision)
library(XRPython)
library(R6)
#devtools::install_github("mlverse/torch")
#install.packages("torchvision")
all_quantile_loss = nn_module(
  classname = "aqc_loss",
  initialize=function(quantiles){
    self$quantiles = quantiles
  },
  forward = function(preds, target){
    losses = torch_tensor(array(0, dim=dim(preds)))
    for(i in 1:length(self$quantiles)){
      errors = target[,1] - preds[,i]
      q = self$quantiles[i]
      losses[,i] =torch_max((q-1)*errors, other = q * errors)
    }
    loss = torch_mean(torch_sum(losses, dim = 2))
    return(loss)
  }
)


my_epoch_internal_train <- function(model, loss_func, xtrain, ytrain, batch_size, 
                                 optimizer_class, lr, wd, device,
                                 cnt = 0, best_cnt = Inf){
  model = model$to(device = device)
  optimizer =optimizer_class(model$parameters, lr = lr, weight_decay = wd)
  model$train()
  n = nrow(xtrain)
  shuffle_idx = sample(1:n, n)
  xtrain = xtrain[shuffle_idx,,drop = FALSE]
  ytrain = ytrain[shuffle_idx,,drop = FALSE]
  xtrain =  torch::torch_tensor(xtrain, dtype = torch::torch_float(), requires_grad = FALSE,
                                device = device)
  ytrain =  torch::torch_tensor(ytrain, dtype = torch::torch_float(), requires_grad = FALSE,
                                device = device)
  
  epoch_losses = c()
  n = dim(xtrain)[1]
  n0 = ceiling(n/batch_size)
  epoch_ranges1 = rep(0, n0)
  epoch_ranges2 = rep(0, n0)
  for(i in 1:n0){
    epoch_ranges1[i] = (i-1) * batch_size+1
    epoch_ranges2[i] = min(epoch_ranges1[i]+batch_size-1, n)
  }
  for(idx in 1:n0){
    cnt = cnt+1
    optimizer$zero_grad()
    batch_x = xtrain[epoch_ranges1[idx]:epoch_ranges2[idx],]
    batch_y = ytrain[epoch_ranges1[idx]:epoch_ranges2[idx],]
    preds = model(batch_x)
    loss = loss_func(preds, batch_y)
    loss$backward()
    optimizer$step()
    epoch_losses = c(epoch_losses, loss$item())
    if(cnt >= best_cnt){
      break;
    }
  }
  epoch_loss = mean(epoch_losses)
  return(list(model = model, epoch_loss = epoch_loss, cnt = cnt))
}


my_test_split_func <- function(x, y, test_ratio, random_state = NULL){
  if (!is.null(random_state)) {
    # reinstate system seed after simulation
    sysSeed <- .GlobalEnv$.Random.seed
    on.exit({
      if (!is.null(sysSeed)) {
        .GlobalEnv$.Random.seed <- sysSeed 
      } else {
        rm(".Random.seed", envir = .GlobalEnv)
      }
    })
    set.seed(random_state, kind = "Mersenne-Twister", normal.kind = "Inversion")
  }
  n = length(y)
  p = ncol(x)
  lltest = ceiling(n * test_ratio)
  lltest = sample(1:n, lltest)
  lltrain = setdiff(1:n,   lltest)
  xtrain = x[lltrain,,drop = FALSE]
  ytrain = y[lltrain,,drop = FALSE]
  xtest = x[lltest,,drop = FALSE]
  ytest = y[lltest,,drop = FALSE]
  return(list(xtrain = xtrain, ytrain= ytrain,
              xtest = xtest, ytest = ytest))
}

my_cv_split_func <- function(x, y, nfolds = 10, random_state = NULL){
  if (!is.null(random_state)) {
    # reinstate system seed after simulation
    sysSeed <- .GlobalEnv$.Random.seed
    on.exit({
      if (!is.null(sysSeed)) {
        .GlobalEnv$.Random.seed <- sysSeed 
      } else {
        rm(".Random.seed", envir = .GlobalEnv)
      }
    })
    set.seed(random_state, kind = "Mersenne-Twister", normal.kind = "Inversion")
  }
  n = length(y)
  p = ncol(x)
  shuffled_idx = sample(1:n, n)
  n0 = ceiling(n/nfolds)
  k01 = nfolds - (n0 * nfolds - n)
  k02 = nfolds - k01
  idx1 = rep(0, nfolds)
  idx2 = rep(0, nfolds)
  foldid = list()
  for(k in 1:nfolds){
    if(k == 1){
      idx1[k] = 1
      idx2[k] = idx1[k]+n0-1
    }else{
      idx1[k] = idx2[k-1] + 1
      if(k <= k01){
        idx2[k] = idx1[k]+n0-1
      }else{
        idx2[k] = idx1[k]+n0-2
      }
    }
  }
  xtrain = list()
  ytrain = list()
  xtest = list()
  ytest = list()
  for(k in 1:nfolds){
    ll = shuffled_idx[idx1[k]:idx2[k]]
    foldid[[k]] = ll
    xtrain[[k]] = x[-ll,,drop = FALSE]
    ytrain[[k]] = y[-ll,,drop = FALSE]
    xtest[[k]] = x[ll,,drop = FALSE]
    ytest[[k]] = y[ll,,drop = FALSE]
  }
  return(list(xtrain = xtrain, ytrain= ytrain,
              xtest = xtest, ytest = ytest, foldid = foldid))
}


my_predict_func = function(x, model, device){
  model$eval()
  ret_val = model(x)
  return(ret_val)
}



#returns the cv evaluation for prediction and gradients on the training data
my_cv_evaluate_func <- function(x, y, num_classes, requires_grad, predict_net, save_path,
                                nfolds, foldid, best_epoch, device){
  yhat = array(NA, dim = c(nrow(x), num_classes))
  jacobians = NULL
  if(requires_grad){
    jacobians = array(NA, dim = c(dim(x)[1],num_classes,
                                  dim(x)[-1]))
  }
  for(k in 1:nfolds){
    print(k)
    model0 <-torch_load(path = paste0(save_path,"_trained_fold", k,"_epoch", best_epoch,".pt"))
    x1 = x[foldid[[k]],,drop = FALSE]
    x1_tensor = torch::torch_tensor(x1, dtype = torch::torch_float(), requires_grad = requires_grad ,
                                    device = device)
    model0$eval()
    y1hat = predict_net(x = x1_tensor, model = model0)
    yhat[foldid[[k]], ] = as.array(y1hat)
    if(requires_grad){
      x_jacobian = compute_jacobian_autograd(inputs = x1_tensor, outputs = y1hat)
      jacobians[foldid[[k]],,] = as.array(x_jacobian, drop = FALSE)
    }
  }
  ret = list(yhat = yhat, jacobians = jacobians)
  return(ret)
}



compute_jacobian_autograd = function(inputs, outputs){
  if(!inputs$requires_grad){
    stop("no grad calculated.")
  }
  num_classes = dim(outputs)[2]
  jacobian = torch::torch_zeros(num_classes,  dim(inputs)[1], dim(inputs)[2])
  grad_output = torch::torch_zeros(dim(outputs)[1], dim(outputs)[2])
  if(inputs$is_cuda){
    grad_output = grad_output$cuda()
    jacobian = jacobian$cuda()
  }
  for(i in 1:num_classes){
    if(i!=1){
      inputs$grad$zero_()
    }
    grad_output$zero_()
    grad_output[,i] = 1
    outputs$backward(grad_output, keep_graph = TRUE)
    jacobian[i,,] = inputs$grad$data()
  }
  return(torch_transpose(jacobian, dim0 = 1, dim1 = 2))
}



my_neuralnet <- nn_module(
  classname = "my_nnet",
  initialize = function(in_shape = 1, out_shape = 1, hidden_size = 64,drop_out = 0.5){
    # in_channels, out_channels, kernel_size, stride = 1, padding = 0
    self$in_shape = in_shape
    self$out_shape = out_shape
    self$hidden_size = hidden_size
    self$drop_out = drop_out
    self$build_model()
    self$init_weights()
  },
  build_model = function(){
    self$base_model = nn_sequential(
      nn_linear(self$in_shape, self$hidden_size),
      nn_relu(),
      nn_dropout(self$drop_out),
      nn_linear(self$hidden_size, self$hidden_size),
      nn_relu(),
      nn_dropout(self$drop_out),
      nn_linear(self$hidden_size, self$out_shape)
    )
  },
  
  init_weights = function(){
    #Initialize the network parameters
    for(i in 1:length(self$base_model)){
      m = self$base_model[[i]]
      if(m$.classes[1] == "nn_linear"){
        nn_init_orthogonal_(m$weight)
        nn_init_constant_(m$bias, 0)
      }
    }
  },
  forward = function(x){
    return(self$base_model(x))
   # torch_squeeze(self$base_model(x))
  }
)

all_quantile_loss = nn_module(
  classname = "aqc_loss",
  initialize=function(quantiles){
    self$quantiles = quantiles
  },
  forward = function(preds, target){
    losses = torch_tensor(array(0, dim=dim(preds)))
    for(i in 1:length(self$quantiles)){
      errors = target[,1] - preds[,i]
      q = self$quantiles[i]
      losses[,i] =torch_max((q-1)*errors, other = q * errors)
    }
    loss = torch_mean(torch_sum(losses, dim = 2))
    return(loss)
  }
)



learnerOptimizer <- R6Class(classname = 'LearnerOptmizer', list(
  model = NULL,
  model1 = NULL,
  optimizer_class = NULL,
  loss_func = NULL,
  test_ratio = NULL,
  random_state = NULL,
  loss_history = NULL,
  test_loss_history = NULL,
  full_loss_history = NULL,
  loss_type = "mse",
  foldid = NULL,
  x = NULL,
  y = NULL,
  CV = FALSE,
  device = NULL,
  save_path = NULL,
  nfolds = NULL,
  best_epoch = NULL,
  cv_split_func = NULL,
  test_split_func = NULL,
  hidden_size = NULL, 
  in_shape = NULL, 
  out_shape = NULL,
  drop_out = NULL,
  lr= NULL,
  wd = NULL, 
  qlow = NULL, 
  qhigh = NULL,
  use_arrangement = FALSE, 
  epoch_internal_train = NULL,
  compute_coverage  = TRUE,
  target_coverage = NULL,
  all_quantiles = NULL,
  initialize = function(network_new, optimizer_class, loss_func, device = 'cpu',
                        loss_type = "mse", all_quantiles = NULL,
                        test_ratio = 0.2,  CV = TRUE, nfolds = 5, random_state = 0,
                        hidden_size = 64, in_shape  = 1, drop_out = 0.2,
                        lr = 0.01 , wd = 1e-6 , qlow = NULL , qhigh = NULL,
                        use_arrangement = FALSE,
                        save_path = 'mymodel', my_cv_split_func = NULL, 
                        my_test_split_func = NULL, my_epoch_internal_train = NULL){
    if(is.null(my_epoch_internal_train)){
      stop("no epoch updating function provided.")
    }
    self$loss_type = loss_type
    self$optimizer_class = optimizer_class
    self$loss_func = loss_func
    self$test_ratio = test_ratio
    self$CV = CV
    self$all_quantiles = all_quantiles
    self$device = device
    self$nfolds = nfolds
    self$best_epoch = NULL
    self$save_path = save_path
    self$random_state = random_state
    self$in_shape = in_shape
    self$hidden_size  = hidden_size
    self$drop_out = drop_out
    self$lr = lr
    self$wd = wd
    self$qlow = qlow
    self$qhigh = qhigh
    self$use_arrangement = use_arrangement
    self$full_loss_history = c()
    self$model1 = list()
    self$loss_history = list()
    self$test_loss_history = list()
    self$epoch_internal_train = my_epoch_internal_train
    if(!self$CV){
      if(is.null(my_test_split_func)){
        stop("no cv splitting function is available.")
      }
      self$test_split_func = my_test_split_func

    }else{
      if(is.null(my_cv_split_func)){
        stop("no cv splitting function is available.")
      }
      self$cv_split_func = my_cv_split_func
    }
    if(self$loss_type == "mse"){
      self$out_shape = 1
    }else if(self$loss_type == "qc"){
      self$target_coverage = (self$qhigh - self$qlow)* 100.0
      if(is.null(all_quantiles)){
        self$all_quantiles =c(self$qlow, self$qhigh)
      }else{
        self$all_quantiles =all_quantiles
      }
      self$out_shape = length(self$all_quantiles)
    }else{
      stop("undefined module structure.")
    }
    torch::torch_manual_seed(self$random_state)
    self$model = network_new(in_shape = self$in_shape, 
                             out_shape = self$out_shape,
                             hidden_size = self$hidden_size,
                             drop_out = self$drop_out)
    if(!self$CV){
      nfolds = 1
      self$nfolds = 1
    }
    for(i in 1:nfolds){
      torch::torch_manual_seed(self$random_state)
      self$model1[[i]] =network_new(in_shape = self$in_shape, 
                                    out_shape = self$out_shape,
                                    hidden_size = self$hidden_size,
                                    drop_out = self$drop_out)
      self$loss_history[[i]] = c(NA)
      self$test_loss_history[[i]] = c(NA)
    }
  },
  
  fit = function(x, y, epochs, batch_size, print_out = 1,verbose = TRUE){
    self$x = x
    self$y = y
    best_epoch = epochs
    if(!self$CV){
      if(is.null(self$foldid)){
        data = self$test_split_func(x = x, y = y, 
                                    test_ratio = self$test_ratio, random_state = self$random_state)
      }else{
        data = list()
        data$xtrain = list()
        data$ytrain = list()
        for(k in 1:length(self$foldid)){
          data$xtrain[[k]] = self$x[self$foldid[[k]],,drop = FALSE]
          data$ytrain[[k]] = self$y[self$foldid[[k]],,drop = FALSE]
        }
      }
    }else{
      if(is.null(self$foldid)){
        data = self$cv_split_func(x = x, y = y, nfolds = self$nfolds, random_state = self$random_state)
        self$foldid = data$foldid
      }else{
        data = list()
        data$xtrain = list()
        data$ytrain = list()
        data$xtest = list()
        data$ytest= list()
        for(k in 1:length(self$foldid)){
          data$xtrain[[k]] = self$x[-self$foldid[[k]],,drop = FALSE]
          data$ytrain[[k]] = self$y[-self$foldid[[k]],,drop = FALSE]
          data$xtest[[k]] = self$x[self$foldid[[k]],,drop = FALSE]
          data$ytest[[k]] = self$y[self$foldid[[k]],,drop = FALSE]
        }
      }
    }
    xtrain = data$xtrain
    ytrain = data$ytrain
    xtest = data$xtest
    ytest = data$ytest
    for(k in 1:self$nfolds){
      print(paste0("###CV_fold: ", k))
      if(!self$CV){
        xtrain_tensor =xtrain
        ytrain_tensor = ytrain
        xtest_tensor = torch::torch_tensor(xtest, dtype = torch::torch_float(), requires_grad = FALSE)
        ytest_tensor = torch::torch_tensor(ytest, dtype = torch::torch_long(), requires_grad = FALSE)
      }else{
        xtrain_tensor =xtrain[[k]]
        ytrain_tensor = ytrain[[k]]
        xtest_tensor = torch::torch_tensor(xtest[[k]], dtype = torch::torch_float(), requires_grad = FALSE)
        ytest_tensor = torch::torch_tensor(ytest[[k]], dtype = torch::torch_float(), requires_grad = FALSE)
      }
      best_cnt = 1e10
      best_test_epoch_loss = 1e10
      cnt = 0
      for(e in c(1:epochs)){
        tmp = self$epoch_internal_train(model = self$model1[[k]], loss_func = self$loss_func,
                                        xtrain = xtrain_tensor, ytrain = ytrain_tensor, 
                                        batch_size = batch_size,  optimizer_class = self$optimizer_class, 
                                        lr = self$lr, wd = self$wd, device = self$device, cnt = 0, best_cnt = Inf)
        cnt = tmp$cnt
        self$loss_history[[k]] = c(self$loss_history[[k]], tmp$epoch_loss)
        self$model1[[k]] = tmp$model
        if(!is.null(self$save_path)){
          torch_save(tmp$model, path = paste0(self$save_path,"_trained_fold", k,"_epoch", e,".pt"))
        }
        ##test eval
        self$model1[[k]]$eval()
        preds = self$model1[[k]](xtest_tensor)
        test_epoch_loss = self$loss_func(preds, ytest_tensor)$item()
        self$test_loss_history[[k]] = c(self$test_loss_history[[k]], test_epoch_loss)
        
        if(test_epoch_loss <= best_test_epoch_loss){
          best_test_epoch_loss = test_epoch_loss
          best_epoch = e
          best_cnt = cnt
        }
        if(e %%  print_out == 0 & verbose){
          print(paste0("Split: Epoch ", e, ", Train ", tmp$epoch_loss,
                       ", Test ", test_epoch_loss, ", Best Epoch",
                       best_epoch, ", Best Epoch loss ", best_test_epoch_loss))
        }
      }
      self$test_loss_history[[k]] = self$test_loss_history[[k]][-1]
      self$loss_history[[k]] = self$loss_history[[k]][-1]
    }
    ##best index and refitting with all data
    if(self$CV){
      loss_history = matrix(NA, ncol = self$nfolds, nrow = epochs)
      test_loss_history = matrix(NA, ncol = self$nfolds, nrow = epochs)
      for(k in 1:self$nfolds){
        loss_history[,k] = self$loss_history[[k]]
        test_loss_history[,k] = self$test_loss_history[[k]]
      }
      test_loss_history = apply(test_loss_history,1,mean)
      loss_history = apply(loss_history,1,mean)
      self$loss_history = loss_history
      self$test_loss_history = test_loss_history
    }else{
      self$loss_history = self$loss_history[[1]]
      self$test_loss_history = self$test_loss_history[[1]]
    }
    best_epoch = which.min(self$test_loss_history)
    self$best_epoch  = best_epoch
    for(e in 1:best_epoch){
      tmp =  self$epoch_internal_train(model = self$model, loss_func = self$loss_func,
                                 xtrain = x, ytrain = y, 
                                 batch_size = batch_size,  optimizer_class = self$optimizer_class, 
                                 lr = self$lr, wd = self$wd,
                                 device = self$device, cnt = 0, best_cnt = Inf)
      self$model = tmp$model
      if(!is.null(self$save_path)){
        torch_save(tmp$model, path = paste0(self$save_path,"_trained_full_epoch", e,".pt"))
      }
      self$full_loss_history = c(self$full_loss_history, tmp$epoch_loss)
      if(e%% print_out == 0 & verbose){
        print(paste0("Full: Epoch ", e, ", cnt ", cnt,
                     ", Loss ", tmp$epoch_loss))
        
      }
    }
  },
  cv_evaluate = function(my_cv_evaluate, num_classes, my_predict_func, requires_grad = TRUE){
    ret = my_cv_evaluate(self$x, self$y, num_classes, requires_grad = requires_grad,
                         save_path = self$save_path,
                         predict_net = my_predict_func,
                         nfolds = self$nfolds, foldid = self$foldid, best_epoch = self$best_epoch, 
                         device = self$device)
    return(ret)
  },
  
  predict = function(x, my_predict_func, requires_grad = TRUE){
    x_tensor = torch::torch_tensor(x, dtype = torch::torch_float(), requires_grad = requires_grad ,
                                   device = self$device)
    self$model$eval()
    yhat_te = my_predict_func(x = x_tensor, model = self$model)
    x_jacobian = NULL
    if(requires_grad){
      x_jacobian = compute_jacobian_autograd(inputs = x_tensor, outputs = yhat_te)
    }
    ret = list(yhat = yhat_te, jacobians =  x_jacobian)
    return(ret)
  }
)
)



