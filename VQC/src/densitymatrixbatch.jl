struct DensityMatrixBatch{T <: Number}
	data::Vector{T}
	nqubits::Int
	nitems::Int

function DensityMatrixBatch{T}(data::AbstractVector{<:Number}, nqubits::Int, nitems::Int) where {T <: Number}
	(length(data) == (2^(2*nqubits)*nitems)) || throw(DimensionMismatch())
	new{T}(convert(Vector{T}, data), nqubits, nitems)
end

end
DensityMatrixBatch(data::AbstractVector, nqubits::Int, nitems::Int) = DensityMatrixBatch{eltype(data)}(data, nqubits, nitems)

DensityMatrixBatch(data::AbstractMatrix{T}, nqubits::Int, nitems::Int) where {T <: Number} = DensityMatrixBatch{T}(vec(data), nqubits, nitems)

DensityMatrixBatch(data::AbstractVector{T}, nqubits::Int, nitems::Int) where {T <: Number} = DensityMatrixBatch{T}(data, nqubits, nitems)

#初始化一个stateVectorBatch,所有态都为0态，输入nqubits 和 nitems
function DensityMatrixBatch(nqubits::Int,nitems::Int)
    datas=zeros(ComplexF64,4^nqubits,nitems)
    datas[1,:].=1.
    return DensityMatrixBatch(datas,nqubits,nitems)
end

@adjoint DensityMatrixBatch(nqubits::Int,nitems::Int)=DensityMatrixBatch(nqubits::Int,nitems::Int),z->(nothing,nothing)
# DensityMatrix(data::AbstractVector) = DensityMatrix(data, div(_nqubits(data), 2))
# DensityMatrix(data::AbstractMatrix) = DensityMatrix(reshape(data, length(data)))
# function DensityMatrix{T}(nqubits::Int) where {T<:Number}
# 	 v = zeros(T, 2^(2*nqubits))
# 	 v[1,1] = 1
# 	 return DensityMatrix{T}(v, nqubits)
# end
# DensityMatrix(::Type{T}, nqubits::Int) where {T<:Number} = DensityMatrix{T}(nqubits)
# DensityMatrix(nqubits::Int) = DensityMatrix(ComplexF64, nqubits)
DensityMatrixBatch(x::DensityMatrixBatch) = DensityMatrixBatch(x.data, nqubits(x), nitems(x))
# DensityMatrix(x::StateVector) = (x_data = storage(x); DensityMatrix(kron(conj(x_data), x_data), nqubits(x)))


storage(x::DensityMatrixBatch) = (L = 2^(nqubits(x)); reshape(x.data, L, L*nitems(x)))
QuantumCircuits.nqubits(x::DensityMatrixBatch) = x.nqubits
nitems(x::DensityMatrixBatch)=x.nitems

Base.eltype(::Type{DensityMatrixBatch{T}}) where T = T
Base.eltype(x::DensityMatrixBatch) = eltype(typeof(x))
# Base.getindex(x::DensityMatrix, j::Int...) = getindex(storage(x), j...)
# Base.setindex!(x::StateVector, v, j::Int...) = setindex!(storage(x), v, j...)

Base.convert(::Type{DensityMatrixBatch{T}}, x::DensityMatrixBatch) where {T<:Number} = DensityMatrixBatch(convert(Vector{T}, x.data), nqubits(x),nitems(x))
Base.copy(x::DensityMatrixBatch) = DensityMatrixBatch(copy(x.data), nqubits(x),nitems(x))


# Base.cat(v::DensityMatrix) = v
# function Base.cat(v::DensityMatrix...)
#     a, b = _qcat_util(storage.(v)...)
#     return DensityMatrix(kron(a, b))
# end

# Base.isapprox(x::DensityMatrix, y::DensityMatrix; kwargs...) = isapprox(x.data, y.data; kwargs...)
# Base.:(==)(x::DensityMatrix, y::DensityMatrix) = x.data == y.data

# Base.:+(x::DensityMatrix, y::DensityMatrix) = DensityMatrix(x.data + y.data, nqubits(x))
# Base.:-(x::DensityMatrix, y::DensityMatrix) = DensityMatrix(x.data - y.data, nqubits(x))
# Base.:*(x::DensityMatrix, y::Number) = DensityMatrix(x.data * y, nqubits(x))
# Base.:*(x::Number, y::DensityMatrix) = y * x
# Base.:/(x::DensityMatrix, y::Number) = DensityMatrix(x.data / y, nqubits(x))
Base.:*(x::Vector,y::DensityMatrixBatch) = DensityMatrixBatch(x.*y.data, nqubits(y),nitems(y))


# LinearAlgebra.tr(x::DensityMatrixBatch) = tr(storage(x))
LinearAlgebra.dot(x::DensityMatrixBatch, y::DensityMatrixBatch) = begin
	r1 = Atomic{Float64}(0)
	r2 = Atomic{Float64}(0)
	x1=storage(x)
	y1=storage(y)
	L=2^nqubits(x)
	nt=nitems(y)-1
	Threads.@threads for i in 0:nt
		tmp=dot(x1[:,i*L+1:i*L+L]',y1[:,i*L+1:i*L+L])
		atomic_add!(r1,real(tmp))
		atomic_add!(r2,imag(tmp))
	end
	return r1[]+im*r2[]
end

Base.:one(x::DensityMatrixBatch) =begin
	nq=nqubits(x)
	N=2^nq
	nt=nitems(x)
	x=zeros(4^nq*nt)
	for j in 0:nt*N-1
		x[j*N+mod(j,N)+1]=1
	end
	return DensityMatrixBatch(x,nq,nt)
end
# LinearAlgebra.normalize!(x::DensityMatrix) = (x.data ./= tr(x); x)
# LinearAlgebra.normalize(x::DensityMatrix) = normalize!(copy(x))
# LinearAlgebra.ishermitian(x::DensityMatrix) = ishermitian(storage(x))
# LinearAlgebra.isposdef(x::DensityMatrix) = isposdef(storage(x))

# fidelity(x::DensityMatrix, y::DensityMatrix) = real(tr(sqrt(storage(x)) * sqrt(storage(y))))
# fidelity(x::DensityMatrix, y::StateVector) = real(dot(storage(y), storage(x), storage(y)))
# fidelity(x::StateVector, y::DensityMatrix) = fidelity(y, x)
# distance2(x::DensityMatrix, y::DensityMatrix) = _distance2(x, y)
# distance(x::DensityMatrix, y::DensityMatrix) = _distance(x, y)
# schmidt_numbers(x::DensityMatrix) = eigvals(Hermitian(storage(x)))
# renyi_entropy(x::DensityMatrix; kwargs...) = renyi_entropy(schmidt_numbers(x); kwargs...)

# function rand_densitymatrix(::Type{T}, n::Int) where {T <: Number}
#     (n >= 1) || error("number of qubits must be positive.")
#     L = 2^n
#     v = randn(T, L, L)
#     v = v' * v
#     return normalize!(DensityMatrix(v' * v, n))
# end
# rand_densitymatrix(n::Int) = rand_densitymatrix(ComplexF64, n)

# function QuantumCircuits.permute(x::DensityMatrix, newindex::Vector{Int})
# 	n = nqubits(x)
# 	L = length(x.data)
# 	return DensityMatrix(reshape(permute(reshape(x.data, ntuple(i->2, 2*n)), vcat(newindex, newindex .+ n)), L), n)
# end 



