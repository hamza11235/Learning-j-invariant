from utils import *



def gen(batch_size,n,lamb):
    
    """
    Generating random points in the fundamental domain, and transforming each of them by a random SL(2,Z)
    """
    
    
    inputs=np.zeros((batch_size,1))+0*1j
    sl2ztr=np.zeros((batch_size,1))+0*1j
    mats=[]
    inputs[0][0]=jnp.exp(1j*2*math.pi/3)
    inputs[1][0]=1j
    for i in range (2,batch_size):
        inputs[i][0]=ComplexInput(lamb)
    for i in range (batch_size):
        sl2ztr[i][0],c=trans(inputs[i][0],n)
        mats.append(c)  
    return (inputs,sl2ztr,mats)