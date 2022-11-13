#import the packages we'll need
using Parameters, LinearAlgebra, Distributions, Random, Statistics, StatFiles, DataFrames, Optim, CSV

#note that we will be using the same data as in PS1B! I'll put the relevant file in the repositiory as done with PS1B
###in new folder, called same thing, call in dataframe as in PS1B in primitives struct
@with_kw struct Primitives

    #using http://www.sparse-grids.de/, we downloaded the following gridpoints/nodes and weights (last column as weights)
    KPU_d1_l20 = DataFrame(load("/Users/hlittle/Desktop/PS2B/KPU_d1_l20.dta"))
    mat_d1 = Matrix(KPU_d1_l20) #one column of nodes and one column of weights
    ###mat_d1 is length 31
    KPU_d2_l20 = DataFrame(load("/Users/hlittle/Desktop/PS2B/KPU_d2_l20.dta"))
    mat_d2 = Matrix(KPU_d2_l20) #two columns of nodes one column of weights
    ###mat_d2 is length 705

    df = DataFrame(load("/Users/hlittle/Desktop/PS2B/Mortgage_performance_data.dta")) 
    mat_df = Matrix(df) #turn the data frame into a Matrix
    
    #concatenate the columns associated with noted vectors into your matrix X, we assume that the last bit is a list, not a difference
    X::Array{Float64, 2} = hcat(mat_df[:, 14], mat_df[:, 5], mat_df[:, 24], mat_df[:, 25], mat_df[:, 21], mat_df[:, 3], mat_df[:, 7], mat_df[:, 8], mat_df[:, 9], mat_df[:, 10], mat_df[:, 23], mat_df[:, 27], mat_df[:, 28], mat_df[:, 29], mat_df[:, 30])
    #X::Array{Float64, 2} = hcat(ones(16355), mat_df[:, 14], mat_df[:, 5], mat_df[:, 24], mat_df[:, 25], mat_df[:, 21], mat_df[:, 3], mat_df[:, 7], mat_df[:, 8], mat_df[:, 9], mat_df[:, 10], mat_df[:, 23], mat_df[:, 27], mat_df[:, 28], mat_df[:, 29], mat_df[:, 30])
    #concatenatw ethe columns assocuated with noted vectors into your matrix, Zit
    Zit::Array{Float64, 2} = hcat(mat_df[:, 14], mat_df[:, 16], mat_df[:, 18])
    #call the outcome variable, i_close_first_year (20th column)
    Yit::Array{Float64, 2} = hcat(mat_df[:, 15], mat_df[:, 17], mat_df[:, 19])
    #call the T outcome variable, which I created a new variable for in stata and saved
    Ti:: Array{Float64, 1} = mat_df[:, 35]
    N::Int64 = length(Ti) #the number of individuals
    η::Array{Float64, 2} = zeros(N, 2) #this should maybe be 16355, that's the number of obs in stata


end #close Primitives struct

mutable struct Mutable

    #each param individually
    α0::Float64
    α1::Float64
    α2::Float64
    β::Array{Float64, 1}
    γ::Float64
    ρ::Float64

    #create a vector of the parameters you're estimating
    param::Array{Float64, 1}

    #create a matrix of the epsilons
    ε_mat::Array{Float64, 2}

    #create a matrix for the eta shocks
    η::Array{Float64, 2}

end

function Initialize()
    prim = Primitives()

    #fill the parameters and param vector with the guesses we see in number 4
    α0 = 0
    α1 = -1
    α2 = -1
    β = zeros(15)
    γ = 0.3
    ρ = 0.5
    param = [α0, α1, α2, β[1], β[2], β[3], β[4], β[5], β[6], β[7], β[8], β[9], β[10], β[11], β[12], β[13], β[14], β[15], γ, ρ]

    #fill in the epsilons with zeros, will change later once we have etas
    ε_mat = zeros(prim.N, 3)

    #fill in the eta matrix with zeros
    η = zeros(prim.N, 3)

    mut = Mutable(α0, α1, α2, β, γ, ρ, param, ε_mat, η)
    return prim, mut

end #close the initialize function

function afterPrim(prim::Primitives)
    #this function will fill in the η draws
    Random.seed!(1234)
    dist = Normal()

    for i = 1:prim.N
        for j = 1:2
            mut.η[i,j] = rand(dist)
        end #close for loop over the individual observations
    end #close for loop over periods
    
end #close function filling in the eta shocks

function eps_update(prim::Primitives, mut::Mutable; rho1) #after initializing, update rho as mut.ρ

    #fill in the first column of epsilons using the given distribution, calculated from rho
    Random.seed!(1234)
    dist = Normal(0, (1/(1-rho1))^2)
    for i = 1:prim.N
        mut.ε_mat[i,1] = rand(dist)
    end #close the loop over individuals, filling in εi0 for all individuals i

    #fill in the next two columns of epsilons for each individual
    for i = 1:prim.N
        for j = 2:3
            mut.ε_mat[i,1] = rho1*mut.ε_mat[i, j-1] + mut.η[i,j-1] #note that the eta matrix has columns 1 and 2 corresponding to columns 2 and 3 here
        end #close the loop over the remaining two columns
    end #close loop over inividuals (rows)

    #return nothing, this function serves to update the mutable struct

end #close the function for the epsilon updates

#the following is analogus to the logistic function from PS1B
function Quadrature_Likelihood(prim::Primitives, mut::Mutable; param::Array{Float64, 1})

    #call the elements from the parameter vector
    #=
    alpha0 = mut.α0
    alpha1 = mut.α1
    alpha2 = mut.α2
    beta = mut.β
    gamma = mut.γ
    rho = mut.ρ
    =#
    alpha0 = param[1] 
    alpha1 = param[2]
    alpha2 = param[3]
    beta = param[4:18]
    gamma = param[19]
    rho = param[20]


    #update the epsilons accordingly
    #eps_update(prim, mut, rho1=rho)

    #create a large matrix so we can call the appropriate values in our bounds
    holder = zeros(prim.N, 3, 2)
    for i=1:prim.N
        for t = 1:3
            holder[i, t, 1] = alpha0 + dot(prim.X[i,:], beta) + gamma*prim.Zit[i,t] #really key to see diff between holder panels 1 and 2!
            holder[i, t, 2] = alpha1 + dot(prim.X[i,:], beta) + gamma*prim.Zit[i,t] #diff is in the alpha0 vs alpha1 terms
        end #close loop over time periods, little t
    end #close loop over individuals

    #create a distribution so we can use the cdf() function below
    distrib = Normal(0,1)

    #create transformations of the nodes using "Additional Notes on Quadrature Integration"
    ###note that we go from negative infinity to upper bound b, we transform for EACH node

    #one dimensional integration
    w_1d = prim.mat_d1[:,2] #save the second column as the weights
    u_1d = prim.mat_d1[:,1] #save the first column as the nodes
    ###down below, as we look at each i and each t, we will transform the nodes using log and the upper bound b

    #two dimensional integration
    w_2d = prim.mat_d2[:,3]
    u1_2d = prim.mat_d2[:,1]
    u2_2d = prim.mat_d2[:,2]


    #create a very large array in which to store the elements of the likelihood
    like_array = zeros(prim.N, 3, 4) #individuals, time periods, duration values

    #fill in for T=1,  no quadrature used here
    for i=1:prim.N
        for t = 1:3
            like_array[i, t, 1] = cdf(distrib, (-holder[i,t,1]/(1/(1-rho))))
        end #close loop over time periods, little t
    end #close loop over individuals

    #fill in for T=2, using 1 dimensional quadrature
    for i=1:prim.N
        for t = 1:3
            #first, adjust the nodes as instructed
            u_1d_trans = log.(u_1d) .+ holder[i,t,1]
            #u_1d_trans = log.(u_1d) .+ alpha0 + dot(prim.X[i,:], beta) + gamma*prim.Zit[i,t]
            weighted_sum = 0
            for k = 1:31 #lenth of the transformed node grid
                weighted_sum += w_1d[k] * cdf(distrib, (-holder[i,t,2] - rho*u_1d_trans[k]))*pdf(distrib, (u_1d_trans[k]/(1/(1-rho))))/(1/(1-rho)) * (1/u_1d[k]) #weight*m()*f()*jacobian
                #weighted_sum += w_1d[k] * cdf(distrib, (-holder[i,t,2] - rho*u_1d_trans[k]))*pdf(distrib, (u_1d_trans[k]/(1/(1-rho))))/(1/(1-rho)) * (1/u_1d_trans[k]) #weight*m()*f()*jacobian
                #changed the pdf bit below
                #weighted_sum += w_1d[k] * cdf(distrib, (-holder[i,t,2] - rho*u_1d_trans[k]))*pdf(distrib, (u_1d_trans[k]/(1/(1-rho))))/(1/(1-rho)) * pdf(distrib, u_1d_trans[k]) * (1/u_1d_trans[k]) #weight*m()*f()*jacobian
            end #close loop over each of the nodes
            like_array[i, t, 2] = weighted_sum #use the weighted sum to fill in the likelihood function for person i in time t
        end #close loop over time periods, little t
    end #close loop over individuals 

    #fill in for T=3, using 2 dimensional quadrature
    for i=1:prim.N
        for t = 1:3
            #first, adjust the nodes as instructed
            u1_2d_trans = log.(u1_2d) .+ holder[i,t,1] #using alpha0 with first panel of holder
            u2_2d_trans = log.(u2_2d) .+ holder[i,t,2] #using alpha1 with second panel of holder
            weighted_sum = 0
            for k = 1:705 #lenth of the transformed node grid
                weighted_sum += w_2d[k]*cdf(distrib, (-alpha2 - dot(prim.X[i,:], beta) - gamma*prim.Zit[i,t] - rho*u2_2d_trans[k]))*pdf(distrib, (u2_2d_trans[k]-rho*u1_2d_trans[k]))*(pdf(distrib, (u1_2d_trans[k]/(1/(1-rho))))/(1/(1-rho)))*(1/u1_2d[k])*(1/u2_2d[k]) 
                #weighted_sum += w_2d[k]*cdf(distrib, (-alpha2 - dot(prim.X[i,:], beta) - gamma*prim.Zit[i,t] - rho*u2_2d_trans[k]))*pdf(distrib, (u2_2d_trans[k]-rho*u1_2d_trans[k]))*(pdf(distrib, (u1_2d_trans[k]/(1/(1-rho))))/(1/(1-rho)))*(1/u1_2d_trans[k])*(1/u2_2d_trans[k]) 
            end #close loop over each of the nodes
            like_array[i, t, 3] = weighted_sum #use the weighted sum to fill in the likelihood function for person i in time t
        end #close loop over time periods, little t
    end #close loop over individuals 

    #fill in for T=4, using 2 dimensional quadrature (this is the same as above, but modify the -alpha1 to become +alpha2, then fill in correct panel)
    for i=1:prim.N
        for t = 1:3
            #first, adjust the nodes as instructed
            u1_2d_trans = log.(u1_2d) .+ holder[i,t,1] #using alpha0 with first panel of holder
            u2_2d_trans = log.(u2_2d) .+ holder[i,t,2] #using alpha1 with second panel of holder
            weighted_sum = 0
            for k = 1:705 #lenth of the transformed node grid
                weighted_sum += w_2d[k]*cdf(distrib, (alpha2 + dot(prim.X[i,:], beta) + gamma*prim.Zit[i,t] - rho*u2_2d_trans[k]))*pdf(distrib, (u2_2d_trans[k]-rho*u1_2d_trans[k]))*(pdf(distrib, (u1_2d_trans[k]/(1/(1-rho))))/(1/(1-rho)))*(1/u1_2d[k])*(1/u2_2d[k]) 
                #weighted_sum += w_2d[k]*cdf(distrib, (alpha2 + dot(prim.X[i,:], beta) + gamma*prim.Zit[i,t] - rho*u2_2d_trans[k]))*pdf(distrib, (u2_2d_trans[k]-rho*u1_2d_trans[k]))*(pdf(distrib, (u1_2d_trans[k]/(1/(1-rho))))/(1/(1-rho)))*(1/u1_2d_trans[k])*(1/u2_2d_trans[k]) 
            end #close loop over each of the nodes
            like_array[i, t, 4] = weighted_sum #use the weighted sum to fill in the likelihood function for person i in time t
        end #close loop over time periods, little t
    end #close loop over individuals


    return like_array

end

function Log_Like_Quad()

    #compute the quadrature function
    Likelihood_Array = Quadrature_Likelihood(prim, mut; param=mut.param)

    #for each observation, look at the relevant duration to know whether or not they repaid it (1, 2, 3, or 4)
    ###note that prim.Ti gives the relevant value, but we'll have to convert the float to an integer
    Likelihood_Collapsed = zeros(prim.N, 3)
    for i = 1:prim.N
        Likelihood_Collapsed[i,:] = Likelihood_Array[i,:, Int64(prim.Ti[i])]
    end #close the loop over individuals

    sum = 0 #initialize
    for i = 1:prim.N
        for t = 1:3
            sum += log((Likelihood_Collapsed[i, t])^(prim.Yit[i,t])*(1-Likelihood_Collapsed[i, t])^(1-prim.Yit[i,t]))
        end #close the for loop over periods
    end #close the for loop over individuals

    return sum

end #close function for log likelihood


#everyone is struggling with this question -> we are going to put it on the back burner
function GHK_j(prim::Primitives, mut::Mutable; param::Array{Float64, 1})

    alpha0 = param[1] 
    alpha1 = param[2]
    alpha2 = param[3]
    beta = param[4:18]
    gamma = param[19]
    rho = param[20]

    #to look at the probability of option j, we need to redefine everything relative to that option (that means we'll have to do this whole thing for option j)

    #begin by redefining the variables in difference relative to a choice (see JF's slide 24/32 in the 2nd slide deck):
    #create the Σ matrix as on slide 24 and as directed by Mary/John
    Σ = ((1/(1-rho)^2)*(1/1-rho^2)).*[1 rho rho^2; rho 1 rho; rho^2 rho 1]
    L = cholesky(Σ)
    #we have ν = Lη, where η is a joint standard normal, here of three dimensions... but each individual already has η pulled for themselves
    ###this is further making me think we should be defining j as time periods 0, 1, 2 since we have three η terms already pulled for each indiv i
    
end #close GHK function

function AR_Likelihood(prim::Primitives, mut::Mutable; param::Array{Float64, 1}) #the log likelihood using the accept reject method, see slide 21-22/32 in JF's lecture 2

    alpha0 = param[1] 
    alpha1 = param[2]
    alpha2 = param[3]
    beta = param[4:18]
    gamma = param[19]
    rho = param[20]

    #create a large matrix so we can call the appropriate values in our bounds (just as we'd done in the quadtrature method)
    holder = zeros(prim.N, 3, 2)
    for i=1:prim.N
        for t = 1:3
            holder[i, t, 1] = alpha0 + dot(prim.X[i,:], beta) + gamma*prim.Zit[i,t] #really key to see diff between holder panels 1 and 2!
            holder[i, t, 2] = alpha1 + dot(prim.X[i,:], beta) + gamma*prim.Zit[i,t] #diff is in the alpha0 vs alpha1 terms
        end #close loop over time periods, little t
    end #close loop over individuals    

    #create a very large array in which to store the elements of the likelihood
    like_array = zeros(prim.N, 3, 4) #individuals, time periods, duration values

    #initialize the distribution we will be using from which to pull our error term draws and also to use the cdf and pdf funcs as in the quadrature method
    distrib = Normal(0,1)

    #fill in for T=1,  no Accept/Reject used here
    for i=1:prim.N
        for t = 1:3
            like_array[i, t, 1] = cdf(distrib, (-holder[i,t,1]/(1/(1-rho))))
        end #close loop over time periods, little t
    end #close loop over individuals  
    
    #fill in for T=2
    ###note that we're told to set the simulation draws to 100 in the pset; for each individual i and each period t, you'll draw 100 error terms 
    ###note that we also need to store the error terms for every i,t within T=2 so we can calculate mbar
    for i=1:prim.N
        for t = 1:3
            #first, adjust the nodes as instructed
            
            for k = 1:100 #number of simulations
                weighted_sum += 
            end #close loop over each of the simulations
            like_array[i, t, 2] = weighted_sum #use the weighted sum to fill in the likelihood function for person i in time t
        end #close loop over time periods, little t
    end #close loop over individuals 







end





#=
Graveyard of shitty code and functions

    u_d1 = zeros(32,2,3) #initialize to transform
    u_d1[:,2,1] = prim.mat_d1[:,2]
    u_d1[:,2,2] = prim.mat_d1[:,2]
    u_d1[:,2,3] = prim.mat_d1[:,2] #the weights are the same across time periods
    for i = 1:32
        for t = 1:3
            u_d1[1, i, t] = ln(prim.mat_d1[i])+ holder
        end #close the loop over time
    end #close loop to transform the nodes

    u_d2 = prim.mat_d2 #copy to then transform the nodes as instructed, leaving weights unchanged
    for i = 1:length(prim.mat_d2) #note that we will transform either


#copying over the function from PS1B
function Log_Like_Quad(param_vec::Vector{Float64})

    #run the quadrature function, getting a big matrix over people, time periods, and decisions T
    like_array = Quadrature_Likelihood(prim, mut; param_vec)



    return sum

end #close function for log likelihood

############
#Use the cholesky decomposition function!



=#