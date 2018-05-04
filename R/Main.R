# Author: Tyler Cady
# Version: 1.0
# Dependencies:
#  keras 2.1.5
#  tensorflow 1.5
#  DataPrep 1.0
#  ConvolutionalNetworkDesign 1.0
#
# Description:
#   This script is the body of the MNIST-FASHION-CNN project. Make sure to update the 
#   relative paths to the required data and function files to reflect your system. There
#   will be inline comments to describe the process and methods used.  
#

# PRELIMINARY WORK 
# Load keras and set the file paths. 
require(keras)
functions.file.path <- "~/Documents/FinalProject/MNIST-Fashion-CNN-Keras/R"
data.file.path <- "~/Documents/Data"
# Load in the functions
source(paste(functions.file.path, "/DataPrep.R", sep = ""))
source(paste(functions.file.path, "/CnnDesign.R", sep = ""))
# Load in the data
train <- read.csv(paste(data.file.path, "/fashion-mnist_train.csv", sep = ""), header = TRUE)
test <- read.csv(paste(data.file.path, "/fashion-mnist_test.csv", sep = ""), header = TRUE)
# Prepare the data using the DataPrep.R script
data.sets <- DataPrep(train = train, test = test)

# TRAINING THE NETWORK 
# 1. Design seven different CNNs. 
# TODO (Tyler): Build a grid of values and loop through them to design the various networks.
cnn1 <- ConvolutionalNetworkDesign(conv_layer_filters = c(32,64), conv_layer_kernel_size = list(c(3,3), c(2,2)), conv_layer_activation = c("relu", "relu"), pool_layer_pool_size = c(2,2), dropout_rates = c(0.25, 0.5), 
               layer_dense_units = c(128), layer_dense_activation = c("relu", "softmax"), classes = 10)
cnn2 <- ConvolutionalNetworkDesign(conv_layer_filters = c(32,64), conv_layer_kernel_size = list(c(4,4), c(2,2)), conv_layer_activation = c("relu", "relu"), pool_layer_pool_size = c(2,2), dropout_rates = c(0.25, 0.5), 
                layer_dense_units = c(128), layer_dense_activation = c("relu", "softmax"), classes = 10)
cnn3 <- ConvolutionalNetworkDesign(conv_layer_filters = c(32,64), conv_layer_kernel_size = list(c(5,5), c(3,3)), conv_layer_activation = c("relu", "relu"), pool_layer_pool_size = c(2,2), dropout_rates = c(0.25, 0.5), 
                 layer_dense_units = c(128), layer_dense_activation = c("relu", "softmax"), classes = 10)
cnn4 <- ConvolutionalNetworkDesign(conv_layer_filters = c(32,64), conv_layer_kernel_size = list(c(5,5), c(3,3)), conv_layer_activation = c("relu", "relu"), pool_layer_pool_size = c(4,4), dropout_rates = c(0.25, 0.5), 
                 layer_dense_units = c(128), layer_dense_activation = c("relu", "softmax"), classes = 10)
cnn5 <- ConvolutionalNetworkDesign(conv_layer_filters = c(32,64), conv_layer_kernel_size = list(c(5,5), c(3,3)), conv_layer_activation = c("relu", "relu"), pool_layer_pool_size = c(8,8), dropout_rates = c(0.25, 0.5), 
                 layer_dense_units = c(128), layer_dense_activation = c("relu", "softmax"), classes = 10)
cnn6 <- ConvolutionalNetworkDesign(conv_layer_filters = c(32,64), conv_layer_kernel_size = list(c(5,5), c(3,3)), conv_layer_activation = c("relu", "relu"), pool_layer_pool_size = c(2,2), dropout_rates = c(0.1, 0.2), 
                 layer_dense_units = c(128), layer_dense_activation = c("relu", "softmax"), classes = 10)
cnn7 <- ConvolutionalNetworkDesign(conv_layer_filters = c(32,64), conv_layer_kernel_size = list(c(5,5), c(3,3)), conv_layer_activation = c("relu", "relu"), pool_layer_pool_size = c(2,2), dropout_rates = c(0.4, 0.75), 
                 layer_dense_units = c(128), layer_dense_activation = c("relu", "softmax"), classes = 10)
cnn8 <- ConvolutionalNetworkDesign(conv_layer_filters = c(32,64), conv_layer_kernel_size = list(c(5,5), c(3,3)), conv_layer_activation = c("relu", "relu"), pool_layer_pool_size = c(2,2), dropout_rates = c(0.05, 0.1),
                 layer_dense_units = c(128), layer_dense_activation = c("relu", "softmax"), classes = 10)
model.list <- list(cnn1, cnn2, cnn3, cnn4, cnn5, cnn6, cnn7, cnn8)

# 2. Fit the models using the training and validation sets, then store the results. 
# TODO (Tyler): Loop through the values and store the results in a list.
batch <- 128
epochs <- 10

for (cnn in model.list) {
	cnn %>% compile(
  	loss = loss_categorical_crossentropy,
  	optimizer = optimizer_adadelta(),
  	metrics = c("accuracy")
	)
}

history1 <- cnn1 %>% fit(
  set.list$train.X, set.list$train.target,
  batch_size = batch,
  epochs = epochs,
  verbose = 1,
  validation_data = list(set.list$val.X, set.list$val.target)
)

history2 <- cnn2 %>% fit(
  set.list$train.X, set.list$train.target,
  batch_size = batch,
  epochs = epochs,
  verbose = 1,
  validation_data = list(set.list$val.X, set.list$val.target)
)

history3 <- cnn3 %>% fit(
  set.list$train.X, set.list$train.target,
  batch_size = batch,
  epochs = epochs,
  verbose = 1,
  validation_data = list(set.list$val.X, set.list$val.target)
)

history4 <- cnn4 %>% fit(
  set.list$train.X, set.list$train.target,
  batch_size = batch,
  epochs = epochs,
  verbose = 1,
  validation_data = list(set.list$val.X, set.list$val.target)
)

history5 <- cnn5 %>% fit(
  set.list$train.X, set.list$train.target,
  batch_size = batch,
  epochs = epochs,
  verbose = 1,
  validation_data = list(set.list$val.X, set.list$val.target)
)

history6 <- cnn6 %>% fit(
  set.list$train.X, set.list$train.target,
  batch_size = batch,
  epochs = epochs,
  verbose = 1,
  validation_data = list(set.list$val.X, set.list$val.target)
)

history7 <- cnn7 %>% fit(
  set.list$train.X, set.list$train.target,
  batch_size = batch,
  epochs = epochs,
  verbose = 1,
  validation_data = list(set.list$val.X, set.list$val.target)
)

history8 <- cnn8 %>% fit(
  set.list$train.X, set.list$train.target,
  batch_size = batch,
  epochs = epochs,
  verbose = 1,
  validation_data = list(set.list$val.X, set.list$val.target)
)

# 3. Evaluate the networks on a test set.
# TODO (Tyler): Loop through the models and store results in a list.
results1 <- cnn1 %>% evaluate(set.list$test.X, set.list$test.target)
results2 <- cnn2 %>% evaluate(set.list$test.X, set.list$test.target)
results3 <- cnn3 %>% evaluate(set.list$test.X, set.list$test.target)
results4 <- cnn4 %>% evaluate(set.list$test.X, set.list$test.target)
results5 <- cnn5 %>% evaluate(set.list$test.X, set.list$test.target)
results6 <- cnn6 %>% evaluate(set.list$test.X, set.list$test.target)
results7 <- cnn7 %>% evaluate(set.list$test.X, set.list$test.target)
results8 <- cnn8 %>% evaluate(set.list$test.X, set.list$test.target)

# 4. Predict classes using the test set. 
# TODO (Tyler): Loop through the models and store the results in a list.
preds1 <- cnn1 %>% predict_classes(set.list$test.X)
preds2 <- cnn2 %>% predict_classes(set.list$test.X)
preds3 <- cnn3 %>% predict_classes(set.list$test.X)
preds4 <- cnn4 %>% predict_classes(set.list$test.X)
preds5 <- cnn5 %>% predict_classes(set.list$test.X)
preds6 <- cnn6 %>% predict_classes(set.list$test.X)
preds7 <- cnn7 %>% predict_classes(set.list$test.X)
preds8 <- cnn8 %>% predict_classes(set.list$test.X)
















































