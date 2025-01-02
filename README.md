# Learning the j-invariant

The goal of this repository is to make progress towards learning the j-invariant, $j(\tau)$, using JAX. This is a modular function of $\tau$ of weight zero for special linear group **SL**(2, $\mathbb{Z}$) defined on the upper half-plane of complex numbers, and is the unique such function that is _holomorphic_ away from a simple pole at the cusp such that:

$j(e^{2\pi/3})=0, \hspace{0.2cm} j(i)=1728$

Since the function is meromorphic, it has a Laurent series which is given by:

$j(\tau)=q^{-1}+744+196884q+21493760q^{2} \dots$

where $q=e^{2\pi i \tau}$. Since this function is unique in the upper-half plane, under certain conditions, one can hope to learn it using neural networks. We use a feed-forward neural network, with complex ReLU activation, which acts on each of the real and imaginary parts seperately. We setup three loss functions, corresponding to the normalization, holomorphicity and the **SL**(2, $\mathbb{Z}$) invariance. 

The long-term goal is to use neural networks to learn more about modular functions in general, which will have consequences for both mathematics and string theory.
