using Parameters, LinearAlgebra, Distributions, Random, Statistics, StatFiles, DataFrames, Optim, CSV

include("PS2B_HLittle_model.jl") #import the functions we've written

prim, mut = Initialize()
afterPrim(prim) #updates eta in the mutable struct
eps_update(prim, mut; rho1=mut.ρ)

#see if the quadrature function works
Likelihood_Array = Quadrature_Likelihood(prim, mut; param=mut.param)
