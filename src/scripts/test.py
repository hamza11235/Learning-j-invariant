from utils import *
from Dataset import*



def test(learned_params,net_apply, batch_size, n, lamb):
    
    """
    Testing learned parameters on unknown data
    """
    
    
    a1,b1,c1=gen(batch_size,n,lamb)
    xs=net_apply(learned_params,b1)
    ys=net_apply(learned_params, a1)
    y=percentageloss(ys,xs)
    x=np.arange(len(y))
    plt.scatter(x, y)
    plt.xlabel('Points')
    plt.ylabel('Percentage difference')
    plt.show()