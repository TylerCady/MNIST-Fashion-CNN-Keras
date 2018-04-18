# Author: Tyler Cady 
# Version 1.0
# 04/17/2018
DataPrep <- function(train, test, data.name.list = c("train", "val", "test"), seed = 510, validation.size = 0.2) {
	# Description:
	# 	This function processes and prepares the MNIST Fashion database for use in a 2dCNN
	# 	The output of this function are three properly formatted datasets with their corresponding
	# 	one-hot-encoded target matrices. 1. The training data and target labels. 2. The validation
	# 	data and target labels. 3. The test data and target labels. 
	#
	# Args:
	# 	train: the training dataset
	# 	test: the test dataset
	# 	data.name.list: a vector of data splits, default is c("train","val","test")
  # 	seed: the random seed for reproducibility, default is 510
  # 	validation.size: the proportion of training observations to include in the validation set
  #
  #	Returns:
  #		The function returns a list of six datasets that constitute a training set with the
  # 	associated target values, a validation set with the corresponding target values, and
  #		a test set with the associated target values. 
  #
  n <- nrow(train)
  set.seed(seed)
  val.rows <- base::sample(n, size = validation.size * n, replace = TRUE)
  image.rows <- 28
	image.columns <- 28

  for (i in data.name.list) {
  	if (i == "train") {
  		assign(paste0(i, ".X"), train[-val.rows, -1])
  		assign(paste0(i, ".target"), train[-val.rows, 1])
  		train.X <- as.matrix(train.X[, 1:dim(train.X)[2]])
			train.target <- as.matrix(train.target)
			dim(train.X) <- c(nrow(train.X), image.rows, image.columns, 1)
			train.target <- to_categorical(train.target, 10)
			train.X <- train.X / 255
  	} else if (i == "val") {
  			assign(paste0(i, ".X"), train[val.rows, -1])
  			assign(paste0(i, ".target"), train[val.rows, 1])
  			val.X <- as.matrix(val.X[, 1:dim(val.X)[2]])
				val.target <- as.matrix(val.target)
				dim(val.X) <- c(nrow(val.X), image.rows, image.columns, 1)
				val.target <- to_categorical(val.target, 10)
				val.X <- val.X / 255
  		} else if (i == "test") {
  				assign(paste0(i, ".X"), test[, -1])
  				assign(paste0(i, ".target"), test[, 1])
  				test.X <- as.matrix(test.X[, 1:dim(test.X)[2]])
					test.target <- as.matrix(test.target)
					dim(test.X) <- c(nrow(test.X), image.rows, image.columns, 1)
					test.target <- to_categorical(test.target, 10)
					test.X <- test.X / 255
  			} else {
  					stop("Error: Misspecified data split names")
  					}

  }

  return(list(train.X, train.target, val.X, val.target, test.X, test.target))
}



















