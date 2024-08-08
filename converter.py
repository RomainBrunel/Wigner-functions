import numpy as np
from scipy.special import genlaguerre
from functools import lru_cache
from scipy.special import factorial
import matplotlib.pyplot as plt

def mygenlaguerre(n,m):
        # Loop over each pair of n and alpha
    results = np.empty(n.shape, dtype=np.poly1d)
    computed_polynomials = {}
    
    # Loop over each pair of n and alpha
    for index in np.ndindex(n.shape):
        n_val = n[index]
        alpha_val = m[index]
        
        # Use tuple (n_val, alpha_val) as the key to check if we already computed this polynomial
        key = (n_val, alpha_val)
        
        if key in computed_polynomials:
            # If already computed, use the cached polynomial
            results[index] = computed_polynomials[key]
        else:
            # Create the generalized Laguerre polynomial for this pair (n, alpha)
            L = genlaguerre(n_val, alpha_val)
            computed_polynomials[key] = L
            results[index] = L
    
    # Return the array of generalized Laguerre polynomials
    return results

def W_mn(x,p,m,n):
    vectorized_apply = np.vectorize(lambda f, p: f(p))
    a = 1/np.pi*np.exp(-x**2-p**2)
    b = (-1)**n*(x-1j*p)**(m-n)
    c = np.sqrt(2**abs(m-n)*factorial(n,exact=True)/factorial(m,exact=True))
    dbis = mygenlaguerre(n,abs(m-n))
    d = vectorized_apply(dbis,2*x**2+2*p**2)
    return a*b*c*d

def wigner_from_density(rho, x, p):
    M = np.array(np.meshgrid(x,p,np.arange(rho.shape[0]),np.arange(rho.shape[0])))
    W = W_mn(M[0],M[1],M[2].astype(np.int16),M[3].astype(np.int16))
    return np.einsum('mn,ijmn',rho,W)

def density_from_wigner(Wign, n, m):
    M = np.array(np.meshgrid(np.arange(Wign.shape[0]),np.arange(Wign.shape[0]),n,m))
    W = W_mn(M[0],M[1],M[2].astype(np.int16),M[3].astype(np.int16))
    return  np.einsum('ij,ijmn',Wign,W)



def density_rom_wigner(wigner):
    pass 

def mygenlaguerre2(n,m,x,p):
        # Loop over each pair of n and alpha
    results = np.empty(n.shape)
    computed_polynomials = {}
    computed_axis = {}
    
    # Loop over each pair of n and alpha
    for index in np.ndindex(n.shape):
        n_val = n[index]
        alpha_val = m[index]
        X = x[index]
        P = p[index]
        # Use tuple (n_val, alpha_val) as the key to check if we already computed this polynomial
        key = (n_val, alpha_val)
        axis = (n_val, alpha_val,X, P)
        
        if key in computed_polynomials:
            # If already computed, use the cached polynomial
            if axis in computed_axis:
                results[index] = computed_axis[axis]
            else :
                computed_axis[axis] = computed_polynomials[key](X**2+P**2)
                results[index] = computed_axis[axis]
        else:
            # Create the generalized Laguerre polynomial for this pair (n, alpha)
            L = genlaguerre(n_val, alpha_val)
            computed_polynomials[key] = L
            computed_axis[axis] = computed_polynomials[key](X**2+P**2)
            results[index] = computed_axis[axis]
    return results

def W_mn2(x,p,m,n):
    vectorized_apply = np.vectorize(lambda f, p: f(p))
    a = 1/np.pi*np.exp(-x**2-p**2)
    b = (-1)**n*(x-1j*p)**(m-n)
    c = np.sqrt(2**abs(m-n)*factorial(n,exact=True)/factorial(m,exact=True))
    dbis = mygenlaguerre2(n,abs(m-n),x,p)
    # d = vectorized_apply(dbis,2*x**2+2*p**2)
    return a*b*c*dbis

def wigner_from_density2(rho, x, p):
    M = np.array(np.meshgrid(x,p,np.arange(rho.shape[0]),np.arange(rho.shape[0])))
    W = W_mn2(M[0],M[1],M[2].astype(np.int16),M[3].astype(np.int16))
    return np.einsum('mn,ijmn',rho,W)