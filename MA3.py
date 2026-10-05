""" MA3.py

Student: Axel Lindblo,
Mail: axel.lindblom.9615@student.uu.se
Reviewed by:
Date reviewed:

"""
import random
import matplotlib.pyplot as plt
import math as m
import concurrent.futures as future
import numpy as np
import multiprocessing as mp
from numba import njit
from statistics import mean 
from time import perf_counter as pc


# Exc1
def approximate_pi(n):
    circle = 0
    in_x = []
    in_y = []
    
    out_x = []
    out_y = []
    for _ in range(n):
        x = random.uniform(-1,1)
        y = random.uniform(-1,1)
        if (x**2 + y**2) <= 1:
            circle += 1
            in_x.append(x)
            in_y.append(y)
        else:
            out_x.append(x)
            out_y.append(y)
    pi_app = 4*(circle/n)
    
        
    plt.scatter(out_x, out_y, color="blue", s=5, label="Outside the circle")
    
    plt.scatter(in_x, in_y, color="red", s=5, label="Inside the circle")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Monte Carlo-approximation of pi with n = {n}")
    plt.axis("equal")
    plt.legend()
    plt.grid()
    plt.show()
    # n is the number of points

    # Write your code here
    return pi_app

# Exc2, approximation
def sphere_volume(n, d): 
    ins = 0
    for i in range(n):
        pnt = [random.uniform(-1,1) for _ in range(d)]
        if sum(map(lambda x: x**2, pnt)) <= 1:
            ins += 1
    # n is the number of points

    # d is the number of dimensions of the sphere 
    vol_app = 2**d*(ins/n)
    return vol_app

#Exc2, real value
def hypersphere_exact(n, d):
    return (m.pi**(d/2)/(m.gamma((d/2)+1)))


#Exc3: numba version
@njit
def sphere_volume_numba(n:int, d:int)->float:
    ins = 0
    for _ in range(n):
        p = 0.0
        for i in range(d):
            x = np.random.uniform(-1.0,1.0)
            p += x*x
        if p <= 1:
            ins += 1
    return (2**d)*ins/n

    # n is the number of points
    # d is the number of dimensions of the sphere
    #np is the number of processes


#Exc4: parallel code - parallelize actual computations by splitting data
def sphere_volume_parallel2(n, d, np1=10):
    npp = n//np1
    with future.ProcessPoolExecutor() as ex:
        results = ex.map(sphere_volume,[npp]*np1,[d]*np1)
    # n is the number of points
    # d is the number of dimensions of the sphere
    # np1 is the number of processes
    return sum(results)/np1
    
def main():
    # Exc1
    dots = [1000, 10000, 100000]
    for n in dots:
        approximate_pi(n)

    # Exc2
    n = 100000
    d = 2
    sphere_volume(n, d)
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")

    n = 100000
    d = 11
    sphere_volume(n, d)
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")

    # Exc3
    
    n = 1000000
    d = 11
   
    for _ in range(3):
        start = pc()
        sphere_volume(n, d)
        stop = pc()
        print(f"Exc3: Sequential time of {d} and {n}: {stop-start}")
    for _ in range(3):
        start1 = pc()
        sphere_volume_numba(n,d)
        stop1 = pc()
        time = stop1-start1
        print(f"What is numba time? for {d} and {n}: {time}")
        

    # Exc4
    n = 1000000
    d = 11
    start = pc()
    sphere_volume(n, d)
    stop = pc()
    start2 = pc()
    sphere_volume_parallel2(n, d, np1=10)
    stop2 = pc()

    print(f"Exc4: Sequential time of {d} and {n}: {stop-start}")
    print(f"What is parallel time?: {stop2-start2}")

    
    

if __name__ == '__main__':
	main()
