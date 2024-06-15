import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')
tf.config.set_visible_devices([], 'GPU') # To enable GPU acceleration, comment out this line and ensure CUDA and cuDNN libraries are properly installed

## modified from https://github.com/alialaradi/DeepGalerkinMethod

#%% Fully connected (dense) layer - modification of Keras layer class
   
class DenseLayer(tf.keras.layers.Layer):
    
    # constructor/initializer function (automatically called when new instance of class is created)
    def __init__(self, output_dim, input_dim, seed, transformation=None):
        '''
        Args:
            input_dim:       dimensionality of input data
            output_dim:      number of outputs for dense layer
            transformation:  activation function used inside the layer; using
                             None is equivalent to the identity map 
        
        Returns: customized Keras (fully connected) layer object 
        '''        
        
        # create an instance of a Layer object (call initialize function of superclass of DenseLayer)
        super(DenseLayer,self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        
        ### define dense layer parameters
        # w vectors (weighting vectors for output of previous layer)
        self.W = self.add_variable("W", shape=[self.input_dim, self.output_dim],
                                   initializer = tf.keras.initializers.GlorotNormal(seed=seed))
        # bias vectors
        self.b = self.add_variable("b", shape=[1, self.output_dim],initializer = tf.keras.initializers.glorot_uniform(seed=seed))
        
        if transformation:
            if transformation == "tanh":
                self.transformation = tf.tanh
            elif transformation == "relu":
                self.transformation = tf.nn.relu
            elif transformation == "sigmoid":
                self.transformation = tf.nn.sigmoid
        else:
            self.transformation = transformation
    
    
    # main function to be called 
    def call(self,X):
        '''Compute output of a dense layer for a given input X 

        Args:                        
            X: input to layer            
        '''
        
        # compute dense layer output
        S = tf.add(tf.matmul(X, self.W), self.b)
                
        if self.transformation:
            S = self.transformation(S)
        
        return S

class FinalLayer(tf.keras.layers.Layer):
    
    # constructor/initializer function (automatically called when new instance of class is created)
    def __init__(self, output_dim, input_dim, seed, final_trans="sigmoid"):
        '''
        Args:
            input_dim:       dimensionality of input data
            output_dim:      number of outputs for dense layer
            final_trans:  activation function used inside the layer; using
                             None is equivalent to the identity map 
        
        Returns: customized Keras (fully connected) layer object 
        '''        
        
        # create an instance of a Layer object (call initialize function of superclass of DenseLayer)
        super(FinalLayer,self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        
        ### define dense layer parameters
        # w vectors (weighting vectors for output of previous layer)
        self.W = self.add_variable("W", shape=[self.input_dim, self.output_dim],
                                   initializer = tf.keras.initializers.GlorotNormal(seed=seed))
        # bias vectors
        self.b = self.add_variable("b", shape=[1, self.output_dim],initializer = tf.keras.initializers.glorot_uniform(seed=seed))
        if final_trans:
            if final_trans == "tanh":
                self.final_trans = tf.tanh
            elif final_trans == "relu":
                self.final_trans = tf.nn.relu
            elif final_trans == "sigmoid":
                self.final_trans = tf.nn.sigmoid
        else:
            self.final_trans = final_trans
    
    
    # main function to be called 
    def call(self,X):
        '''Compute output of a dense layer for a given input X 

        Args:                        
            X: input to layer            
        '''
        
        # compute dense layer output
        S = tf.add(tf.matmul(X, self.W), self.b)
                
        if self.final_trans:
            S = tf.concat([S[:, :2], self.final_trans(S[:, 2:])], axis=1)
        return S
    
#%% Neural network architecture used in DGM - modification of Keras Model class

class DGMNet(tf.keras.Model):
    
    # constructor/initializer function (automatically called when new instance of class is created)
    def __init__(self, layer_width, n_layers, input_dim, activation, seed, final_trans="tanh"):
        '''
        Args:
            layer_width: number of neurons in each layer
            n_layers:    number of intermediate dense layers
            input_dim:   dimension of input data 
            activation:  activation function used in all layers except final
            final_trans: transformation used in final layer
        
        Returns: customized Keras model object representing DGM neural network
        '''  
        
        # create an instance of a Model object (call initialize function of superclass of DGMNet)
        super(DGMNet,self).__init__()
        
        # define initial layer as fully connected 
        self.initial_layer = DenseLayer(layer_width, input_dim, transformation = activation,  seed = seed)
        
        # define intermediate dense layers
        self.n_layers = n_layers
        self.DenseLayerList = []
        
        for i in range(self.n_layers):
            self.DenseLayerList.append(DenseLayer(layer_width, layer_width, transformation = activation, seed = seed+ i + 1))
        
        # define final layer as fully connected with a single output (function value)
        self.final_layer = FinalLayer(3, layer_width, final_trans=final_trans, seed = seed + self.n_layers + 1)
    
    
    # main function to be called  
    def call(self,X):
        '''            
        Args:
            x: sampled space inputs

        Run the DGM model and obtain fitted function value at the inputs x                
        '''  

        # call initial layer
        S = self.initial_layer.call(X)

        # call intermediate Dense layers
        for i in range(self.n_layers):
            S = self.DenseLayerList[i].call(S)
        
        # call final Dense layers
        result = self.final_layer.call(S)
        
        return result