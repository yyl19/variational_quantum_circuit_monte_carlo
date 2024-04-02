

@adjoint storage(x::Union{StateVector, DensityMatrix}) = storage(x), z -> (z,)
@adjoint StateVector(data::AbstractVector{<:Number}, n::Int) = StateVector(data, n), z -> (z, nothing)
@adjoint StateVector(data::AbstractVector{<:Number}) = StateVector(data), z -> (z,)
@adjoint DensityMatrix(data::AbstractMatrix{<:Number}, n::Int) = DensityMatrix(data, n), z -> (z, nothing)
@adjoint DensityMatrix(data::AbstractMatrix{<:Number}) = DensityMatrix(data), z -> (z,)

# this is stupid, why should I need it
@adjoint dot(x::StateVector, y::StateVector) = Zygote.pullback(dot, storage(x), storage(y))

# #StateVectorBatch
# @adjoint dot(x::StateVectorBatch, y::StateVectorBatch) = Zygote.pullback(dot, storage(x), storage(y))

