#neural network pytorch
library(torch)
library(torchvision)
library(XRPython)
library(R6)

mse_neuralnet <- nn_module(
  classname = "mse_nnet",
  initialize = function(in_shape = 1, hidden_size = 64,drop_out = 0.5){
    # in_channels, out_channels, kernel_size, stride = 1, padding = 0
    self$in_shape = in_shape
    self$out_shape = 1
    self$hidden_size = hidden_size
    self$drop_out = drop_out
    self$build_model()
    self$init_weight()
  },
  build_model = function(){
    self$base_model = nn_sequential(
      nn_linear(self$in_shape, self$hidden_size),
      nn_relu(),
      nn_dropout(self$dropout),
      nn_linear(self$hidden_size, self$hidden_size),
      nn_relu(),
      nn_dropout(self$drop_out),
      nn_dropout(self$hidden_size, 1)
    )
  },
  init_weights = function(){
    #Initialize the network parameters
    for(m in self$base_model){
      if(XRPython::isinstance(m, nn_linear)){
        nn_init_orthogonal_(m$weight)
        nn_init_constant_(m$bias, 0)
      }
    }
  },
  forward = function(x){
    return(torch::torch_squeeze(self$base_model(x)))
  }
)

epoch_internal_train <- function(model, loss_func, x_train, y_train, batch_size, optimizer,
                                 cnt = 0, best_cnt = Inf){
  model$train()
  shuffle_idx = sample(1:nrow(x_train), n)
  x_train = xtrain[shuffle_idx,]
  y_train = ytrain[shuffle_idx]
  epoch_losses = c()
  n = nrow(x_train)
  n0 = ceiling(n/batch_size)
  epoch_ranges1 = rep(0, n0)
  epoch_ranges2 = rep(0, n0)
  for(i in 1:n0){
    epoch_ranges1[i] = (i-1) * batch_size+1
    epoch_ranges2[i] = min(epoch_ranges1[i]+batch_size-1, n)
  }
  for(idx in 1:n0){
    cnt = cnt+1
    optmizer$zero_grad()
    batch_x = x_train[epoch_ranges1[i]:epoch_ranges2[i],]
    batch_y = y_train[epoch_ranges1[i]:epoch_ranges2[i]]
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
  return(list(epoch_loss = epoch_loss, cnt = cnt))
}

learnerOptimizer <- R6Class(classname = 'LearnerOptmizer', list(
  model = NULL,
  optmizer_class = NULL,
  loss_func = NULL,
  test_ratio = NULL,
  random_state = NULL,
  loss_history = NULL,
  test_loss_history = NULL,
  full_loss_history = NULL,
  
  initialize = function(model, optiizer_class, loss_func, device = 'cpu',
                        test_ratio = 0.2, random_state = 0){
    self$model = model
    self$optmizer_class = optmizer_class
    self$optmizer = optimizer_class(model$parameters)
    self$loss_func = loss_func$to(devide)
    self$test_ratio = test_ratio
    self$random_state = random_state
    self$loss_history = c()
    self$test_loss_history = c()
    self$full_loss_history = c()
  },
  
  fit = function(x, y, epochs, batch_size, verbose = TRUE){
    model = self$model
    model = model$to(self$device)
    optimizer = self$optimizer_class(model$parameters)
    best_epoch = epochs
    data = train_test_split(x = x, y = y, 
                            test_ratio = self$test_ratio, random_state = self$random_state)
    xtrain = data$xtrain
    ytrain = data$ytrain
    xtest = data$xtest
    ytest = data$ytest
    #convert data to tensor
    xtrain = torch::torch_tensor(xtrain, dtype = torch::torch_float(), requires_grad = FALSE)
    xtest = torch::torch_tensor(xtest, dtype = torch::torch_float(), requires_grad = FALSE)
    ytrain = torch::torch_tensor(ytrain, dtype = torch::torch_float(), requires_grad = FALSE)
    ytest = torch::torch_tensor(ytest, dtype = torch::torch_float(), requires_grad = FALSE)
    #start learning
    best_cnt = 1e10
    best_test_epoch_loss = 1e10
    cnt = 0
    for(e in c(1:epochs)){
      tmp = epoch_internal_train(model, self$loss_func, x_train, y_train,
                                 batch_size, optmizer, cnt)
      cnt = tmp$cnt
      self$loss_history = c(self$loss_histor, tmp$epoch_loss)
      ##test eval
      model$eval()
      preds = model(xtest$to(self$device))
      test_epoch_loss = self$loss_func(preds, ytest$to(self$device))$cpu()$detech()
      self$test_loss_history = c(self$test_loss_history, test_epoch_loss)
      if(test_epoch_loss <= best_test_epoch_loss){
        best_test_epoch_loss = test_epoch_loss
        best_epoch = e
        best_cnt = cnt
      }
      if(e %% 100 == 0 & verbose){
        print(paste0("Split: Epoch ", e, ", Train ", tmp$epoch_loss,
                     ", Test ", test_epoch_loss, ", Best Epoch",
                     best_epoch, ", Best Epoch loss ", best_test_epoch_loss))
      }
    }
    # use all the data to train the model, for best_cnt steps
    x = torch::torch_tensor(x, dtype = torch::torch_float(), requires_grad = FALSE)
    y = torch::torch_tensor(y, dtype = torch::torch_float(), requires_grad = FALSE)
    cnt = 0
    for(e in 1:best_epoch){
      if(cnt > best_cnt){
        break;
      }
      tmp = epoch_internal_train(self$model, self$loss_func, x, y,
                                 batch_size, optmizer, cnt)
      cnt = tmp$cnt
      self$full_loss_history = c(self$full_loss_history, tmp$epoch_loss)
      if(e%%100 == 0 & verbose){
        print(paste0("Full: Epoch ", e, ", cnt ", cnt,
                     ", Loss ", tmp$epoch_loss))
        
      }
    }
    return(model)
  },
  
  predict = function(x, grad_eval = TRUE){
    self$model$eval()
    ret_val = self$model(torch::torch_tensor(x, dtype = torch::torch_float(), requires_grad =  grad_eval))
    return(ret_val)
  }
)
)


