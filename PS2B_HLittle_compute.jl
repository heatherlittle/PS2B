using Parameters, LinearAlgebra, Distributions, Random, Statistics, StatFiles, DataFrames, Optim, CSV

include("PS2B_HLittle_model.jl") #import the functions we've written

prim, mut = Initialize()
afterPrim(prim) #updates eta in the mutable struct
eps_update(prim, mut; rho1=mut.ρ)

#see if the quadrature function works
Likelihood_Array_Quad = Quadrature_Likelihood(prim, mut; param=mut.param)

#see what the quadrature function looks like in the log likelohood function
Likelihood_Quad = Log_Like_Quad(mut.param)

#see if the accept reject function works
Likelihood_Array_AR = AR_Likelihood(prim, mut; param=mut.param)

#see what the accept reject function looks like in the log likelohood function
Likelihood_AR = Log_Like_AR(mut.param)

#see if the GHK function works
Likelihood_Array_GHK = GHK_Likelihood(prim, mut; param=mut.param)

#see what the GHK function looks like in the log likelihood function
Likelihood_GHK = Log_Like_GHK(mut.param)

####compare the time for each
time_LL_Quad = @elapsed Log_Like_Quad()
time_LL_AR = @elapsed Log_Like_AR()


############################################################################
#optimize using the quadrature method
param_guess = mut.param
param_BFGS = optimize(Likelihood_Quad, param_guess, BFGS())
println("The parameters that minimize the log likelihood function with the BFGS algorithm are ", param_BFGS.minimizer, ".")
############################################################################
#optimize using the accept reject method
param_guess = mut.param
param_BFGS = optimize(Likelihood_AR, param_guess, BFGS())
println("The parameters that minimize the log likelihood function with the BFGS algorithm are ", param_BFGS.minimizer, ".")



#=
###########test to see where I'm breaking
#doing what I want the function above to do manually
sum = 0 #initialize
    for i in eachindex(Likelihood_Array)
        sum += log(Likelihood_Array[i])
    end #close the for loop
println(sum)