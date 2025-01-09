import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax import random
from jax import lax
import math
import sys
from random import randint
import itertools
from jax import jacfwd, jacrev
from subprocess import call
import matplotlib.pyplot as plt

# Generate key which is used to generate random numbers
key = random.PRNGKey(0)
key2 = jax.random.PRNGKey(2)
from jax.example_libraries import stax
from jax.example_libraries.stax import (BatchNorm, Conv, Dense, Flatten, 
                                   MaxPool, Relu, LogSoftmax)

from jax.scipy.special import logsumexp
from jax.example_libraries import optimizers
from jax.nn import (relu, log_softmax, softmax, softplus, sigmoid, elu,
                    leaky_relu, selu, gelu)
from jax.nn.initializers import glorot_normal, normal, ones, zeros


#Functions

def Dense1(out_dim, W_init=glorot_normal(), b_init=normal()):
    
    """
    Creating single layer of a feed-forward Neural Network.
    """
    
    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2 = random.split(rng)
        W, b = W_init(k1, (input_shape[-1], out_dim))+W_init(k2, (input_shape[-1], out_dim))*1j, b_init(k2, (out_dim,))+b_init(k1, (out_dim,))*1j
        return output_shape, (W, b)
    def apply_fun(params, inputs, **kwargs):
        W, b = params
        return jnp.dot(inputs, W) + b
    return init_fun, apply_fun


def initparams(net_init,stepsize):
    
    """
    Initializing parameters.
    """
    
    rng = random.PRNGKey(0)
    in_shape = (-1,1)
    out_shape, net_params = net_init(rng, in_shape)
    opt_init, opt_update, get_params = optimizers.adam(step_size=stepsize)
    return net_params, opt_init, opt_update, get_params


def comprelu(x):
    
    """
    Custom ReLU function for complex numbers: Acts on real and imaginary parts separately.
    """
    
    a=jnp.real(x)
    b=jnp.imag(x)
    a=relu(a)
    b=relu(b)
    return a+b*1j


def elementwise(fun, **fun_kwargs):
    init_fun = lambda rng, input_shape: (input_shape, ())
    apply_fun = lambda params, inputs, **kwargs: fun(inputs, **fun_kwargs)
    return init_fun, apply_fun


"""
Generating a complex number randomly in the fundamental domain in complex upper-half plane.

We use a normal distribution to generate the real part, and an exponential distribution to generate the imaginary part.
"""
def ComplexInput(lamb):
    
    """
    Generating a complex number randomly in the fundamental domain in complex upper-half plane.

    We use a normal distribution to generate the real part, and an exponential distribution to generate the imaginary part.
    """
    
    mean1=0
    sd1=0.25
    y=np.random.exponential(lamb,size=(1))
    x=np.random.normal(mean1, sd1, size=(1))
    while x**2 + y**2<1 or y<0 or x<-0.5 or x>0.5:
        x=np.random.normal(mean1, sd1, size=(1))
        y=np.random.exponential(lamb,size=(1))
    return x+y*1j


def percentageloss(x,y):
    
    """
    Percentage difference between the magnitude of two complex numbers 
    """
    
    return 100*(jnp.abs(x)-jnp.abs(y))/jnp.abs(y)



def init_loss(params,batch,net_apply):
    
    """
    Initial loss 
    """
    
    inputs, targs,m = batch
    predictions=net_apply(params, inputs)
    diff12=(jnp.abs(predictions[0]))**2 + (jnp.abs(predictions[1]-1+0*1j))**2
    return diff12[0]



def loss(params, batch,net_apply):
    
    """
    Loss function imposing SL(2,Z) invriance 
    """
    
    inputs, targs,m = batch
    predictions = net_apply(params, inputs)
    targets=net_apply(params, targs)
    diff=predictions - targets
    return jnp.sum((jnp.abs(diff)**2))



def holo_loss(params, batch,net_apply):
    
    """
    Loss function imposing Holomorphy 
    """
    
    inputs, targs,m = batch
    f1 = lambda inputs: net_apply(params,inputs)
    f2=lambda targs: net_apply(params, targs)
    i1=jax.jvp(f1, (inputs,), (np.ones(inputs.shape)+0*1j,))[1]
    i2=jax.jvp(f1, (inputs,), (0-np.ones(inputs.shape)*1j,))[1]
    o1=jax.jvp(f2, (targs,), (np.ones(targs.shape)+0*1j,))[1]
    o2=jax.jvp(f2, (targs,), (0-np.ones(targs.shape)*1j,))[1]
    loss1=i1-i2*1j
    loss2=o1-o2*1j
    return jnp.sum((jnp.abs(loss1))**2)+jnp.sum((jnp.abs(loss2))**2)



def total_loss(params,batch,net_apply):
    
    """
    Total loss 
    """
    
    c=loss(params,batch,net_apply)+holo_loss(params,batch,net_apply)+init_loss(params,batch,net_apply)
    return c


def randomat(x,interval):
    
    """
    Generating a random 2x2 matrix with integer entries between -interval and interval 
    """
    
    low=-interval
    high=interval
    y=[[0 ,0  ],[0  , 0 ]]
    y[0][0]=randint(low,high)
    y[0][1]=randint(low,high)
    y[1][0]=randint(low,high)
    y[1][1]=randint(low,high)
    return (y[0][0]*x + y[0][1]),y



def trans(x,interval):
    
    """
    Generating a random 2x2 SL(2,Z) matrix with integer entries between -interval and interval 
    """
    
    low=-interval
    high=interval
    y=[[0 ,0  ],[0  , 0 ]]
    while y[0][0]*y[1][1]-y[0][1]*y[1][0]!=1:
        y[0][0]=randint(low,high)
        y[0][1]=randint(low,high)
        y[1][0]=randint(low,high)
        y[1][1]=randint(low,high)
    return (y[0][0]*x + y[0][1])/(y[1][0]*x + y[1][1]),y


def step(i, opt_state, opt_update, get_params,net_apply, batch):
    
    """
    Updating NN parameters
    """
    
    params1 = get_params(opt_state)
    losss=total_loss(params1,batch,net_apply)
    g = grad(total_loss)(params1, batch,net_apply)
    return opt_update(i, g, opt_state),losss

    
    
def trans2(y,x):
    
    """
    Some check function (not important)
    """
    
    return (y[0][0]*x + y[0][1])/(y[1][0]*x + y[1][1])


def sl2check(batch_size,mats):
    
    """
    Some check function (not important)
    """
    
    in3=np.zeros((batch_size,1))+0*1j
    sl2=np.zeros((batch_size,1))+0*1j
    for i in range (batch_size):
        in3[i][0]=ComplexInput()
    for i in range (batch_size):
        sl2[i][0]=trans2(mats[i],in3[i][0])
    return in3, sl2


def sl2check2(batch_size,n):
    
    """
    Some check function (not important)
    """
    
    in3=np.zeros((batch_size,1))+0*1j
    sl2=np.zeros((batch_size,1))+0*1j
    for i in range (batch_size):
        in3[i][0]=ComplexInput()
    for i in range (batch_size):
        sl2[i][0]=trans(in3[i][0],n)
    return in3, sl2
    
