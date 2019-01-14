# Gaussian
using Distributions
using LinearAlgebra: Diagonal, ⋅, checksquare, cholesky, logdet
using Statistics

"""
    Gaussian(μ, Σ) -> P

Gaussian distribution with mean `μ` and covariance `Σ`. Defines `rand(P)` and `(log-)pdf(P, x)`. The type is designed to work with `StaticArrays`. Internally, a representation of the matrix square root `Σsqrt` of `Σ` is stored. The square root has to satisfy that that `Σ ≈ Σsqrt'*Σsqrt` which means that a Cholesky factor can be used but it would also be possible to store e.g. a `Eigen` factorization. The default is to compute the Choleksy factor when a matrix is passed to the constructor.
"""
struct Gaussian{T,S}
    μ::T
    Σsqrt::S
end

function Gaussian(μ::AbstractVector, Σ::AbstractMatrix)
    Σsqrt = cholesky(Σ).U
    Gaussian{typeof(μ),typeof(Σsqrt)}(μ, Σsqrt)
end
Gaussian(Σ::SMatrix{M,N}) where {M,N} = Gaussian(zero(similar_type(Σ, Size(M))), Σ)
Gaussian(Σ::Diagonal{<:Any,<:SVector}) = Gaussian(zero(similar_type(parent(Σ))), Σ)

Base.length(P::Gaussian) = length(P.μ)

Base.rand(P::Gaussian) = P.μ + P.Σsqrt'*randn(typeof(P.μ))
Base.rand(P::Gaussian, n::Integer) = [rand(P)' for i in 1:n]

function Distributions.logpdf(P::Gaussian, x)
    ε = P.Σsqrt'\(x - P.μ)
    lp  = -logdet(P.Σsqrt) - (ε⋅ε)/2
    lp -= log(2*typeof(lp)(π))*length(P)/2
    return lp
end
Distributions.pdf(P::Gaussian, x) = exp(logpdf(P::Gaussian, x))

Statistics.mean(P::Gaussian) = P.μ
Statistics.cov(P::Gaussian) = P.Σsqrt'*P.Σsqrt
