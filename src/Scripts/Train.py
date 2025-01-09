from utils1 import *
from Dataset import*

"""
Training function
"""
def NN(batch_size,steps,num_epochs,n,params, opt_init, opt_update, get_params,net_apply,lo):
    opt_state = opt_init(params)
    losss=10
    dat=[]
    arr=[10,100,1000,10000]
    for j in range(steps):
        data_generator = ( gen(batch_size,n,arr[randint(0,3)] ) for _ in range(num_epochs))
        itercount = itertools.count()
        while losss>lo:
            try:
                c=next(data_generator)
            except:
                break
            opt_state,losss= step(next(itercount), opt_state,opt_update,get_params,net_apply, c)
            inputs,sl2ztr,m=c
            x1=net_apply(get_params(opt_state), sl2ztr)
            y1=net_apply(get_params(opt_state), inputs)
            y=percentageloss(x1,y1)
            x=np.arange(len(y))
            dat.append(c)
    plt.scatter(x,y)
    plt.ylabel("Percentage difference")
    plt.xlabel("Points")
    plt.show()
    net_params = get_params(opt_state)
    return net_params,dat

"""
Slightly varied training functions
"""
def NNmat(batch_size,steps,num_epochs,n,params, opt_init, opt_update, get_params,mats,net_apply,lo):
    opt_state = opt_init(params)
    losss=10
    for j in range(steps):
        inputs=np.zeros((batch_size,1))+0*1j
        sl2ztr=np.zeros((batch_size,1))+0*1j
        inputs[0][0]=jnp.exp(1j*2*math.pi/3)
        inputs[1][0]=1j
        for i in range (2,batch_size):
            inputs[i][0]=ComplexInput()
        for i in range (batch_size):
            sl2ztr[i][0]=trans2(mats[i],inputs[i][0])
        print(inputs)
        print(sl2ztr)
        data_generator = ((inputs, sl2ztr) for _ in range(num_epochs))
        itercount = itertools.count()
        while losss>lo:
            opt_state,losss= step(next(itercount), opt_state,opt_update,get_params,net_apply, next(data_generator))
    net_params = get_params(opt_state)
    x1=net_apply(net_params, sl2ztr)
    y1=net_apply(net_params, inputs)
    y=percentageloss(x1,y1)
    print(y)
    x=np.arange(len(y))
    plt.scatter(x, y)
    plt.show()
    return net_params,inputs,sl2ztr

def NNin(batch_size,steps,num_epochs,n,params, opt_init, opt_update, get_params,net_apply,inputs,lo):
    opt_state = opt_init(params)
    losss=10
    for j in range(steps):
        sl2ztr=np.zeros((batch_size,1))+0*1j
        mats=[]
        for i in range (batch_size):
            sl2ztr[i][0],c=trans(inputs[i][0],n)
            mats.append(c)
        print(inputs)
        print(sl2ztr)
        data_generator = ((inputs, sl2ztr) for _ in range(num_epochs))
        itercount = itertools.count()
        while losss>lo:
            opt_state,losss= step(next(itercount), opt_state,opt_update,get_params,net_apply, next(data_generator))
    net_params = get_params(opt_state)
    x1=net_apply(net_params, sl2ztr)
    y1=net_apply(net_params, inputs)
    y=percentageloss(x1,y1)
    print(y)
    x=np.arange(len(y))
    plt.scatter(x, y)
    plt.show()
    return net_params,sl2ztr,mats