from utils1 import *


"""
Creating a feed-forward NN and initializing the parameters
"""

def feedforward(stepsize):
    compRelu=elementwise(comprelu)
    net_init, net_apply = stax.serial(
        Dense1(256), compRelu,
        Dense1(256), compRelu,
        Dense1(256), Flatten,
        Dense1(256), compRelu,
        Dense1(1)
    )
    net_params, opt_init, opt_update, get_params=initparams(net_init,stepsize)
    return net_params, opt_init, opt_update, get_params,net_apply


