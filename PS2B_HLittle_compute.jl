using Parameters, LinearAlgebra, Distributions, Random, Statistics, StatFiles, DataFrames, Optim, CSV

include("PS2B_HLittle_model.jl") #import the functions we've written

prim, mut = Initialize()
afterPrim(prim) #updates eta in the mutable struct
eps_update(prim, mut; rho1=mut.ρ)

#see if the quadrature function works
Likelihood_Array = Quadrature_Likelihood(prim, mut; param=mut.param)

#see what the quadrature function looks like in the log likelohood function
Likelihood = Log_Like_Quad()








#=
###########test to see where I'm breaking
#doing what I want the function above to do manually
sum = 0 #initialize
    for i in eachindex(Likelihood_Array)
        sum += log(Likelihood_Array[i])
    end #close the for loop
println(sum)