
using Distributions
using Random
using LinearAlgebra
using PDMats
using ProgressMeter
E = LinearAlgebra.I
eye(n) = Matrix(1.0E,n,n) #identity matrix

#import Fontconfig, Cairo
using Gadfly

using SpecialFunctions

# fundamaental setting
## calc. RLCT
SIM_ITERS = 20
LOG_FILE_PATH = joinpath("log","result.csv")
## MCMC
MCMC_K = 1000
BURNIN = 20000
THIN = 20
MCMC_ITER = BURNIN + THIN*MCMC_K
## true and model
#Random.seed!(1959)
eps = 1e-5
U_0 = [1; 2; 3; 1-eps]
V_0 = [1 9 9 1+eps]
#const A_0 = ones(10,1).*[0.1; 0.05; 0.02; 0.03; 0.15; 0.25; 0.1; 0.1; 0.15; 0.05] #[0.4 0.3; 0.1 0.3; 0.4 0.2; 0.1 0.2;]#np.array([[0.5,0.3],[0.1,0.3],[0.4,0.4]])
#const B_0 = ones(Float64,1,9) #[0.1 0.1 0.4; 0.9 0.9 0.6;] #np.array([[0.1,0.1,0.4],[0.9,0.9,0.6]])
M = 4 #size(A_0)[1] #vocab. size
N = 4 #size(B_0)[2] #num of docs
H = 2 #num of topics in learner
H_0 = 1 #size(A_0)[2] #true num of topics
## hyper parameter
PHI_U = 0.25
THETA_U = 1.0
PHI_V = 0.75
THETA_V = 1.0
## data
SEED = 1994
#Random.seed!(SEED);
L = 500
SAMPLESIZE = L
TESTSIZE = 100*SAMPLESIZE #15000 #100*SAMPLESIZE

println(U_0)
println(V_0)
println(U_0 * V_0)

function generate_matrix_from_prior(nR, nC; phi=PHI_V, theta=THETA_V, seed=10000)
    W = zeros(Float64, nR, nC);
    for i in 1:nR
        for j in 1:nC
            W[i,j] = rand(Gamma(phi, 1/theta))
        end
    end
    return W
end

function generate_data(U, V; n::Int=SAMPLESIZE, seed=1)
    #===
    for l=1:n
        generate words from Cat(x|AB[:,j]) for j=1,...,N
        as two format: docs array (N, n//N, M) and words array (n, M)
    end

    :param U: stochastic matrix A= (the probability that the word is i when the topic is k)
    :type U: shape is (M,H)
    :param V: stochastic matrix B= (the probability that the topic is k when the doc is j)
    :type V: shape is (H,N)
    :param n: the number of all words in all document. sample size
    :type n: Int
    :return: generated data matrices whose [i,j,l] entry is the (i,j)-element in the l-th matrix.
    :rtype: Array{Int,3}
    ===#
    Random.seed!(seed);
    C = U*V
    #M = size(C)[1]
    #N = size(C)[2]
    Xs_arr = zeros(Int,M,N,n)
    for i in 1:M
        for j in 1:N
            X_ij = rand(Poisson(C[i,j]), n)
            Xs_arr[i,j,:] = X_ij;
        end
    end
    return Xs_arr    
end

# each sampling step for Gibbs sampler
function latent_step(Xs_arr::Array{Int64,3}, U, V; n=SAMPLESIZE)
    #===
    latent variable sampling step.
    
    :param Xs_arr: data
    :param U: param matrix
    :param V: param matrix
    :return: a latent variable s whose shape is (M,H,N,n)
    ===#
    #H = size(A)[2]
    #N = size(B)[2]
    #L = div(n, N)
    W = U*V #W_ij = sum_k U_ik V_kj
    π = zeros(Float64, M, H, N)
    s = zeros(Float64, M, H, N, n)
    for i in 1:M
        U_i = H>1 ? U[i,:] : U
        for j in 1:N
            V_j = H>1 ? V[:,j] : V
            π[i,:,j] = U_i.*V_j ./ W[i,j]
            for l in 1:n
                s[i,:,j,l] = rand(Multinomial(Xs_arr[i,j,l], π[i,:,j]))
            end
        end
    end
    return s
end
    
function U_step(s, V, phi_U, theta_U; n=SAMPLESIZE)
    #===
    matrix U sampling step.
    
    :param s: latent variable tensor; shape is (M, H, N, n)
    :param phi_U: hyperparameter of Gamma prior for U (RLCT depends on it)
    :param theta_U: hyperparameter of Gamma prior for U (RLCT does not depend on it)
    :return: param matrix U
    ===#
    hat_phi_U = sum(s, dims=(3,4)) .+ phi_U
    hat_theta_U = n .* sum(V, dims=2) .+ theta_U
    ## Note: Gamma dist in Julia is x^{phi-1} e^{-x/theta_julia}.
    ## Thus: our theta must be inversed for plugin: Gamma(phi, theta_julia = 1/theta ).
    U = zeros(Float64, M, H)
    for i in 1:M
        for k in 1:H
            U[i,k] = rand(Gamma(hat_phi_U[i,k,1,1], 1 ./ hat_theta_U[k,1]))
        end
    end
    return U
end

function V_step(s, U, phi_V, theta_V; n=SAMPLESIZE)
    #===
    matrix V sampling step.
    
    :param s: latent variable tensor; shape is (M, H, N, n)
    :param phi_V: hyperparameter of Gamma prior for V (RLCT depends on it)
    :param theta_V: hyperparameter of Gamma prior for V (RLCT does not depend on it)
    :return: param matrix U
    ===#
    hat_phi_V = sum(s, dims=(1,4)) .+ phi_V
    hat_theta_V = n .* sum(U, dims=1) .+ theta_V
    ## Note: Gamma dist in Julia is x^{phi-1} e^{-x/theta_julia}.
    ## Thus: our theta must be inversed for plugin: Gamma(phi, theta_julia = 1/theta ).
    V = zeros(Float64, H, N)
    for k in 1:H
        for j in 1:N
            V[k,j] = rand(Gamma(hat_phi_V[1,k,j,1], 1 ./ hat_theta_V[1,k]))
        end
    end
    return V
end

function model_pmf(X::Array{Int64,2}, U, V)
    #===
    probability mass function of model p(X | A,B).
    See also the above markdown cell.
    
    :param X: an (M,N) non-negative integer matrix which means a data.
    :param U: an (M,H) non-negative matrix.
    :param V: an (H,N) non-negative matrix.
    ===#
    #mass = 1.0
    W = U*V
    #Brute Forse
    #===for i in 1:M
        for j in 1:N
            mass *= pdf(Poisson(W[i,j]), X[i,j])
        end
    end===#
    log_mass = sum(X.*log.(W) - W - loggamma.(X.+1))
    mass = exp(log_mass)
    return mass
end

# Gibbs sampler
function run_all_sampling!(
        Xs_arr::Array{Int64,3}, init_S,
        #allSs::Array{Int64,5},
        allUs::Array{Float64,3}, allVs::Array{Float64,3})
    #===
    run Gibbs sampling before doing burn-in and thining.
    
    :param Xs_arr: dataset; shape is (M, N, n)
    :param init_S: initial value of the latent variable; shape is (M, H, N, n)
    :param allSs: (to be mutated!) all sample of latent variables; shape is (M, H, N, n, sampling_iters)
    :param allUs: (to be mutated!) all samle of parameter U; shape is (M, H, sampling_iters)
    :param allVs: (to be mutated!) all samle of parameter V; shape is (H, N, sampling_iters)
    :return: void (this function is mutator of allSs, allUs, and allVs.)
    ===#
    ## get the number of iterations and check array size
    iters = size(allUs)[3];
    @assert size(allVs)[3]==iters
    ## get hyperparameter
    ϕ_U = PHI_U;
    θ_U = THETA_U;
    ϕ_V = PHI_V;
    θ_V = THETA_V;
    ## define Progress object for checking iteration progress
    sampling_progress = Progress(iters);
    ## set S to the initial value
    S = init_S;
    ## set V to the initial value from the prior
    V = generate_matrix_from_prior(H, N)
    ## start all sampling
    @time for k in 1:iters
        ## parameter matrix sampling and saving
        U = U_step(S, V, ϕ_U, θ_U);
        V = V_step(S, U, ϕ_V, θ_V);
        allUs[:,:,k] = U;
        allVs[:,:,k] = V;
        ## latent variable sampling and saving
        S = latent_step(Xs_arr, U, V);
        #allSs[:,:,:,:,k] = S;
        ## Progress update
        next!(sampling_progress);
    end
    #return (allYs, allAs, allBs)
end

function run_process_posterior!(
        Xs_arr::Array{Int64,3}, 
        #allSs::Array{Int64,5},
        allUs::Array{Float64,3}, allVs::Array{Float64,3},
        #gsSs::Array{Int64,5},
        gsUs::Array{Float64,3}, gsVs::Array{Float64,3}, likelihoodMat::Array{Float64,2};
        K::Int=MCMC_K, burn::Int=BURNIN, th::Int=THIN, n::Int=SAMPLESIZE)
    #===
    For sampling result, this function do burn-in, thining, and calculating the likelihood matrix (for WAIC).
    ===#
    # burnin and thining
    ## check MCMC samplesize
    #@assert size(gsUs)[3]==size(gsSs)[5]
    @assert size(gsUs)[3]==size(gsVs)[3]
    @assert size(gsUs)[3]==K
    ## define Progress object.
    thining_progress = Progress(K);
    ## run thining and burn-in
    @time for k in 1:K
        U = allUs[:,:,burn+k*th];
        gsUs[:,:,k] = U;
        V = allVs[:,:,burn+k*th];
        gsVs[:,:,k] = V;
        #gsSs[:,:,:,:,k] = allSs[:,:,:,:,burn+k*th];
        ## generated quantity: likelihood matrix
        #likelihoodMat = calc_likelihoodMat!(k, words_arr, doc_onehot, A, B, likelihoodMat)
        for l in 1:n
            X = Xs_arr[:,:,l];
            likelihoodMat[l,k] = model_pmf(X,U,V);
        end
        next!(thining_progress);
    end
    #return (gsYs, gsAs, gsBs, likelihoodMat)
end

function run_Gibbs_sampler_cored(
    Xs_arr::Array{Int64,3}; K::Int=MCMC_K, burn::Int=BURNIN, th::Int=THIN, n::Int=SAMPLESIZE, seed_MCMC=2)
    #===
    Version: separating core functions
    ===#
    Random.seed!(seed_MCMC);
    ## initial value of A and B is sampled from the prior distributuion
    #### Assume that prior is SYMMETRIC Gamma distribution. Hyparam is scalar.
    init_U = generate_matrix_from_prior(M, H; phi=PHI_U, theta=THETA_U);
    init_V = generate_matrix_from_prior(H, N; phi=PHI_V, theta=THETA_V);
    init_S = latent_step(Xs_arr, init_U, init_V);
    ## sampling iteration
    #### allocate tensors for MCMC sample
    iters = burn + K*th;
    allUs = zeros(Float64, M, H, iters);
    gsUs = zeros(Float64, M, H, K);
    allVs = zeros(Float64, H, N, iters);
    gsVs = zeros(Float64, H, N, K);
    #allSs = zeros(Int64, M, H, N, n, iters);
    #gsSs = zeros(Int64, M, H, N, n, K);
    #### all sampling
    println("Start $iters iteration for GS")
    run_all_sampling!(Xs_arr, init_S, allUs, allVs)
    ## burnin and thining
    likelihoodMat = zeros(Float64, n, K)
    println("Start burn-in and thining from $iters to $K")
    run_process_posterior!(Xs_arr, allUs, allVs, gsUs, gsVs, likelihoodMat)
    return (gsUs, gsVs, likelihoodMat)
end



function run_prior_sampling(Xs_arr; K=MCMC_K, burn=BURNIN, th=THIN, n=SAMPLESIZE, seed_MCMC=2)
    #===
    Run prior sampling K times.
    
    :return: sample of Y(topic indicator variable),A and B from GS and likelihood matrix
    :rtype: tuple(Array{Bool,3};shape(H,n,K), Array{Float64,3};shape(M,H,K), Array{Float64,3};shape(H,N,K), Array{Float64,2};shape(n,K))
    ===#
    Random.seed!(seed_MCMC);
    ## sampling iteration
    #### allocate tensors for MCMC sample
    gsUs = zeros(Float64, M, H, K);
    gsVs = zeros(Float64, H, N, K);
    #gsSs = zeros(Int64, M, H, N, n, K);
    #### all sampling and calc. likelihood Mat
    println("Start $K iteration for prior sampling (no thining)")
    sampling_progress = Progress(K);
    likelihoodMat = zeros(Float64, n, K)
    @time for k in 1:K
        ## sampling and saving
        U = generate_matrix_from_prior(M, H; phi=PHI_U, theta=THETA_U);
        V = generate_matrix_from_prior(H, N; phi=PHI_V, theta=THETA_V);
        S = latent_step(Xs_arr, U, V);
        gsUs[:,:,k] = U;
        gsVs[:,:,k] = V;
        #gsSs[:,:,:,:,k] = S;
        ## calc likelihood matrix
        for l in 1:n
            X = Xs_arr[:,:,l];
            likelihoodMat[l,k] = model_pmf(X,U,V);
        end
        ## Progress update
        next!(sampling_progress);
    end
    return (gsUs, gsVs, likelihoodMat)
end

# define functions for calculating RLCT
function calc_functional_var(loglikeMat::Array{Float64,2})
    #===
    Calculate functional variance from loglike matrix.
    
    :param loglikeMat: a matrix whose (l,k) element is log p(x_l|z_l,A_k,B_k)
    ===#
    n = size(loglikeMat)[1];
    K = size(loglikeMat)[2];
    first_term = reshape(mean(loglikeMat.^2, dims=2),n);
    second_term = reshape(mean(loglikeMat, dims=2).^2, n);
    func_var = sum(first_term - second_term);
    return func_var
end

function calc_predict_dist(X, Us, Vs)
    #===
    Calculate pmf of predictive distribution.
    
    :param X: an (M,N) matrix which means a new data.
    :param Us: an Array whose [:,:,k] element means an MCMC sample of U.
    :param Vs: an Array whose [:,:,k] element means an MCMC sample of V.
    ===#
    @assert size(Us)[3]==size(Vs)[3];
    K = size(Vs)[3];
    mass = 0.0;
    for k in 1:K
        mass += model_pmf(X, Us[:,:,k], Vs[:,:,k]);
    end
    return mass/K
end

function calc_normalized_WAIC(Xs_arr, true_U, true_V, likelihoodMat)
    #===
    Calculating normalized WAIC.
    ===#
    n = SAMPLESIZE;
    emp_loss = -mean(log.(mean(likelihoodMat, dims=2)));
    emp_entropy = -mean([log(model_pmf(Xs_arr[:,:,l], true_U, true_V)) for l in 1:n])
    func_var = calc_functional_var(log.(likelihoodMat))
    #println(typeof(emp_entropy))
    normalized_WAIC = emp_loss + func_var/n - emp_entropy
    return normalized_WAIC
end

function calc_generalization_error(true_U, true_V, Us, Vs; nT=TESTSIZE, seed_T=3)
    #===
    Calculating generalization error using test data generated by true distribution.
    ===#
    test_Xs_arr = generate_data(true_U, true_V, n=nT, seed=seed_T)
    ge = 0.0
    calc_gerr_progress = Progress(nT)
    println("Test sample size nT = $nT")
    @time for t in 1:nT
        q = model_pmf(test_Xs_arr[:,:,t], true_U, true_V)
        pred = calc_predict_dist(test_Xs_arr[:,:,t], Us, Vs)
        ge += log(q) - log(pred) #log(q/pred), but q and pred is small number.
        next!(calc_gerr_progress)
    end
    ge /= nT
    return ge
end


function calc_hamiltonian_unif(likelihoodMat)
    #===
    Calculate hamiltonian if the prior is uniform.
    I.e. this function calclutates negative logarithm likelihood at each MCMC sample.
    ===#
    K = size(likelihoodMat)[2]
    #@assert K==size(As)[3]
    #@assert K==size(Bs)[3]
    loglikeMat = log.(likelihoodMat)
    loglike = sum(loglikeMat, dims=1)[1,:]
    @assert length(loglike) == K
    return -loglike
end

function run_single_inference(true_U, true_V, seed)
    #===
    Run single inference for debug mode.
    ===#
    Random.seed!(seed);
    train_X = generate_data(true_U, true_V, n=SAMPLESIZE, seed=seed+1);
    println("Gibbs Sampling")
    param_Us, param_Vs, likelihoodMat = run_Gibbs_sampler_cored(train_X, seed_MCMC=seed+2);
    println("Calculation Hamiltonian trace")
    hamils = calc_hamiltonian_unif(likelihoodMat)
    println("Calculation Normalized WAIC")
    normalized_WAIC = calc_normalized_WAIC(train_X, true_U, true_V, likelihoodMat);
    println("Calculaton Generalization Error")
    ge = calc_generalization_error(true_U, true_V, param_Us, param_Vs, seed_T=seed+3);
    return (hamils, ge, normalized_WAIC)
end

function run_single_inference_debug(true_U, true_V, seed)
    #===
    Run single inference for debug mode.
    ===#
    Random.seed!(seed);
    train_X = generate_data(true_U, true_V, n=SAMPLESIZE, seed=seed+1);
    println("Gibbs Sampling")
    param_Us, param_Vs, likelihoodMat = run_Gibbs_sampler_cored(train_X, seed_MCMC=seed+2);
    println("Calculation Normalized WAIC")
    normalized_WAIC = calc_normalized_WAIC(train_X, true_U, true_V, likelihoodMat);
    println("Calculaton Generalization Error")
    ge = calc_generalization_error(true_U, true_V, param_Us, param_Vs, seed_T=seed+3);
    return (param_Us, param_Vs, likelihoodMat, ge, normalized_WAIC)
end

function run_single_inference_PS(true_U, true_V, seed)
    #===
    Run single inference using prior sampling (worst benchmark).
    ===#
    Random.seed!(seed);
    train_X = generate_data(true_U, true_V, n=SAMPLESIZE, seed=seed+1);
    println("Prior Sampling")
    latent_Ss, param_Us, param_Vs, likelihoodMat = run_prior_sampling(train_X, seed_MCMC=seed+2)
    println("Calculation Normalized WAIC")
    normalized_WAIC = calc_normalized_WAIC(train_X, true_U, true_V, likelihoodMat);
    if normalized_WAIC>1e10
        normalized_WAIC = 1e10
    elseif isnan(normalized_WAIC)
        normalized_WAIC = 1e12 + 2e11
    elseif normalized_WAIC<0
        println("Warning: normalized_WAIC became negative!")
    end
    println("Calculaton Generalization Error")
    ge = calc_generalization_error(true_U, true_V, param_Us, param_Vs, seed_T=seed+3);
    if ge>1e10
        ge = 1e10
    elseif isnan(ge)
        ge = 1e12 + 2e11
    elseif ge<0
        println("Warning: Generalization Error became negative!")
    end
    return (latent_Ss, param_Us, param_Vs, likelihoodMat, ge, normalized_WAIC)
end

function run_multi_inference(true_U, true_V, sim_iters, log_file_path)
    #===
    for it in 1:sim_iters
        run_single_inference
    end
    and calculate RLCT from above $sim_iters simulation results.
    
    :param true_U: true parameter non-negative matrix U
    :param true_V: true parameter non-negative matrix V
    :param sim_iters: number of simulations
    :param log_file_path: path of the file to write the experimental log
    :return: generalization errors and normalized WAICs in the simulations
    :rtype: tuple(Array{Float64,1}, Array{Float64,1})
    ===#
    
    seeds = SEED .+ SEED * (1:sim_iters)
    gerrors = zeros(Float64, sim_iters)
    normalized_WAICs = zeros(Float64, sim_iters)
    hamiltonian_vecs = zeros(Float64, MCMC_K, sim_iters)
    open(log_file_path, "a") do fp
        println(fp, "## Simulation Setting")
        println(fp, "M,N,H,H_0,PHI_U,THETA_U,PHI_V,THETA_V,SAMPLESIZE,MCMC_K,BURNIN,THIN,TESTSIZE,SIM_ITERS,SEED")
        println(fp, "$M,$N,$H,$H_0,$PHI_U,$THETA_U,$PHI_V,$THETA_V,$SAMPLESIZE,$MCMC_K,$BURNIN,$THIN,$TESTSIZE,$sim_iters,$SEED")
        println(fp, "## Simulation Log")
        println(fp, "iter,gerror,normalized_WAIC,RLCT,RLCT_SEM")
    end
    simulation_progress = Progress(sim_iters)
    for it in 1:sim_iters
        println("# start $it th simulation")
        hamils, ge, norm_W = run_single_inference(true_U, true_V, seeds[it])
        #plot(x=1:MCMC_K, y=hamils, Geom.line, Geom.point)
        hamiltonian_vecs[:,it] = hamils
        gerrors[it] = ge
        normalized_WAICs[it] = norm_W
        now_ges = gerrors[1:it]
        now_nwaics = normalized_WAICs[1:it]
        rlct = (SAMPLESIZE/2)*mean(now_ges + now_nwaics)
        rlct_sem = (SAMPLESIZE/2)*std(now_ges + now_nwaics)/sqrt(it)
        open(log_file_path, "a") do fp
            println(fp,"$it,$ge,$norm_W,$rlct,$rlct_sem")
            println("$it,$ge,$norm_W,$rlct,$rlct_sem")
        end
        next!(simulation_progress)
        #@assert ge>=0
        #@assert norm_W>=0
    end
    return (hamiltonian_vecs, gerrors, normalized_WAICs, seeds)
end

# main process
#===#
hamivecs, gerrors, normalized_WAICs, seeds = @time run_multi_inference(U_0, V_0, SIM_ITERS, LOG_FILE_PATH)
ge_and_nW = gerrors + normalized_WAICs
each_lams = (SAMPLESIZE/2) .* ge_and_nW
lam = (SAMPLESIZE/2)*mean(ge_and_nW)
lam_sd = (SAMPLESIZE/2)*std(ge_and_nW)/sqrt(SIM_ITERS)
println("The RLCT is $lam ± $lam_sd")
#===#

each_lams_mean = mean(each_lams)
each_lams_std = std(each_lams)
upperline = each_lams_mean + each_lams_std
middleline = each_lams_mean
lowerline = each_lams_mean - each_lams_std
plot(y=each_lams, yintercept = [upperline, middleline, lowerline], Geom.violin, Geom.hline(color=["orange","red","orange"], style=[:dash,:solid,:dash]))

lam2 = SAMPLESIZE*mean(gerrors)

lam3 = SAMPLESIZE*mean(normalized_WAICs)

println("$M, $N, $H, $H_0, $PHI_U, $PHI_V")
lam_ub = (H_0*(M+N-1) + (H-H_0)*min(PHI_U*M, PHI_V*N))/2

#Phase Transition Line in VBNMF
phase = M*PHI_U + N*PHI_V < (M+N)/2

lam_reg = (M+N)*H/2

lam_reg_rrr = (M*H + H*N - H*H)/2

#exact lambda_vb
if phase
    lam_vb = (H-H_0)*(M*PHI_U + N*PHI_V) + (M+N)*H_0/2
else
    lam_vb = lam_reg
end
lam_vb

#memo: script for convergence diagnox from visual.

#===
for i in 1:M
    for k in 1:H
        p = plot(x=debug_As[i,k,:], Geom.histogram(bincount=12),
                Guide.title("posthist20191226_A[$i.$k]"))
        img = PNG(joinpath("img_MSI","20191226-d1","posthist_A_$i-$k.png"), 16cm, 16cm)
        draw(img,p)
        p = plot(x=1:MCMC_K, y=debug_As[i,k,:], Geom.point, Geom.line,
                Guide.title("posttrace20191226_A[$i.$k]"))
        img = PNG(joinpath("img_MSI","20191226-d1","posttrace_A_$i-$k.png"), 16cm, 16cm)
        draw(img,p)
    end
end

for k in 1:H
    for j in 1:N
        p = plot(x=debug_Bs[k,j,:], Geom.histogram(bincount=12),
                Guide.title("posthist20191226_B[$k.$j]"))
        img = PNG(joinpath("img_MSI","20191226-d1","posthist_B_$k-$j.png"), 16cm, 16cm)
        draw(img,p)
        p = plot(x=1:MCMC_K, y=debug_Bs[k,j,:], Geom.point, Geom.line,
                Guide.title("posttrace20191226_B[$k.$j]"))
        img = PNG(joinpath("img_MSI","20191226-d1","posttrace_B_$k-$j.png"), 16cm, 16cm)
        draw(img,p)
    end
end
===#
