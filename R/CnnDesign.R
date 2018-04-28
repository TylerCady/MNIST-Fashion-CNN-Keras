# Author: Tyler Cady 
# Version 1.0
# Dependencies: 
#		keras 2.1.5
# 	tensorflow 1.5
#
ConvolutionalNetworkDesign <- function(conv_layer_filters, conv_layer_kernel_size, conv_layer_activation, pool_layer_pool_size, dropout_rates, 
                                       layer_dense_units, layer_dense_activation, classes = 10, image_rows = 28, image_columns = 28) {
  #	
  #	Description:
  #		This function allows for the architectural customization of a simple convolutional network 
  #		within the keras framework. 
  #	
  #	Args:
  #		conv_layer_kernel_size: A list of length 2 vectors that specifies the kernel size for the
  #			respective conv_2d layers. 
  #		conv_layer_activation: A character vector that specifies the activation funcitons for the 
  #			respective conv_2d layers.
  #		pool_layer_pool_size: A list of length 2 vectors that specifies the pooling kernel size
  #			for the respective max_pooling_2d layers.
  # 	dropout_rates: A length 2 vector of floats in [0,1] that specify the respective dropout 
  #			rates.
  #		layer_dense_units: A length 2 vector of integers that specifies the amount of hidden units 
  #			to include in the dense layers. 
  #		layer_dense_activation: A length 2 character vector that specifies the activation functions 
  # 		for the dense layers. 
  #		classes: The number of labeled target classes in the dataset.
  #		image_rows: The amount of rows of pixals that the image has.
  #		image_columns: The amount of columns of pixals the image has.
  #
  # Returns:
  # 	This function returns a keras sequential model; in particular a convolutional neural network 
  #		with two 2d convolutional layers, one 2d max pooling layer, two fully connected layers, two
  #		dropout layers, and the other specified parameters in the function input. 
  #
  input.shape <- c(image_rows, image_columns, 1)
  n.classes <- classes
  
  model <- keras_model_sequential()
  
  model %>%
    layer_conv_2d(filters = conv_layer_filters[1], kernel_size = conv_layer_kernel_size[[1]], activation = conv_layer_activation[1],
                  input_shape = input.shape) %>%
    layer_conv_2d(filters = conv_layer_filters[2], kernel_size = conv_layer_kernel_size[[2]], activation = conv_layer_activation[2]) %>%
    layer_max_pooling_2d(pool_size = pool_layer_pool_size[1]) %>%
    layer_dropout(rate = dropout_rates[1]) %>% 
    layer_flatten() %>%
    layer_dense(units = layer_dense_units[1], activation = layer_dense_activation[1]) %>%
    layer_dropout(rate = dropout_rates[2]) %>%
    layer_dense(units = n.classes, activation = layer_dense_activation[2])
  
  return(model)

}





































