struct CuDensityMatrixBatch{T <: Number}
	data::CuArray{T}
	nqubits::Int
	nitems::Int

function CuDensityMatrixBatch{T}(data::CuArray{<:Number}, nqubits::Int, nitems::Int) where {T <: Number}
	(length(data) == (2^(2*nqubits)*nitems)) || throw(DimensionMismatch())
	new{T}(convert(CuArray{T}, data), nqubits, nitems)
end

end
CuDensityMatrixBatch(data::CuArray, nqubits::Int, nitems::Int) = CuDensityMatrixBatch{eltype(data)}(data, nqubits, nitems)

CuDensityMatrixBatch(data::CuMatrix{T}, nqubits::Int, nitems::Int) where {T <: Number} = CuDensityMatrixBatch{T}(vec(data), nqubits, nitems)

CuDensityMatrixBatch(data::CuArray{T}, nqubits::Int, nitems::Int) where {T <: Number} = CuDensityMatrixBatch{T}(data, nqubits, nitems)

#初始化一个stateVectorBatch,所有态都为0态，输入nqubits 和 nitems
function CuDensityMatrixBatch{T}(nqubits::Int,nitems::Int) where T
    datas=zeros(T,4^nqubits,nitems)
    datas[1,:].=1.
    return CuDensityMatrixBatch(CuArray(datas),nqubits,nitems)
end

# @adjoint DensityMatrixBatch(nqubits::Int,nitems::Int)=DensityMatrixBatch(nqubits::Int,nitems::Int),z->(nothing,nothing)
# DensityMatrix(data::AbstractVector) = DensityMatrix(data, div(_nqubits(data), 2))
# DensityMatrix(data::AbstractMatrix) = DensityMatrix(reshape(data, length(data)))
# function DensityMatrix{T}(nqubits::Int) where {T<:Number}
# 	 v = zeros(T, 2^(2*nqubits))
# 	 v[1,1] = 1
# 	 return DensityMatrix{T}(v, nqubits)
# end
# DensityMatrix(::Type{T}, nqubits::Int) where {T<:Number} = DensityMatrix{T}(nqubits)
# DensityMatrix(nqubits::Int) = DensityMatrix(ComplexF64, nqubits)
CuDensityMatrixBatch(x::DensityMatrixBatch) = CuDensityMatrixBatch(CuArray(x.data), nqubits(x), nitems(x))
# DensityMatrix(x::StateVector) = (x_data = storage(x); DensityMatrix(kron(conj(x_data), x_data), nqubits(x)))


storage(x::CuDensityMatrixBatch) = (L = 2^(nqubits(x)); reshape(x.data, L, L*nitems(x)))
QuantumCircuits.nqubits(x::CuDensityMatrixBatch) = x.nqubits
nitems(x::CuDensityMatrixBatch)=x.nitems

Base.eltype(::Type{CuDensityMatrixBatch{T}}) where T = T
Base.eltype(x::CuDensityMatrixBatch) = eltype(typeof(x))
# Base.getindex(x::DensityMatrix, j::Int...) = getindex(storage(x), j...)
# Base.setindex!(x::StateVector, v, j::Int...) = setindex!(storage(x), v, j...)

Base.convert(::Type{CuDensityMatrixBatch{T}}, x::CuDensityMatrixBatch) where {T<:Number} = CuDensityMatrixBatch(convert(CuArray{T}, x.data), nqubits(x),nitems(x))
Base.copy(x::CuDensityMatrixBatch) = CuDensityMatrixBatch(copy(x.data), nqubits(x),nitems(x))


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
# Base.:*(x::Vector,y::CuDensityMatrixBatch) = begin
	
# 	DensityMatrixBatch(x.*y.data, nqubits(y),nitems(y))
# end

function apply_coefficient!(z::Vector, y::CuDensityMatrixBatch)
	z1 = CuArray(convert.(ComplexF64,z))
	nq = nqubits(y)
	v=y.data
	@inline function f1(v,z1,n)
		index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
		stride = blockDim().x * gridDim().x
		for j = index:stride:length(v)
			i = j-1
			i>>(n-1)+1
			v[j] *= z1[i>>n+1]
		end
	end
	kernel = @cuda launch=false  f1(v,z1,2*nq)
    config = launch_configuration(kernel.fun)
    threads = min(length(v), config.threads)
    blocks = cld(length(v), threads)
    kernel(v,z1,2*nq; threads, blocks)
	return y
end



# LinearAlgebra.tr(x::DensityMatrixBatch) = tr(storage(x))
# LinearAlgebra.dot(x::CuDensityMatrixBatch, y::CuDensityMatrixBatch) = begin
# 	N=2^nqubits(x)
#     T=Float64
# 	res1 = CuArray{T}([zero(T)])
# 	res2 = CuArray{T}([zero(T)])
# 	@inline function f1(x,y,N,res1,res2,T)
# 		index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
# 		stride = blockDim().x * gridDim().x
# 		s1 =zero(T)
# 		s2 =zero(T)
# 		for j = index:stride:length(x)
# 			i=j-1
# 			bis=i>>N+1
# 			pos= (i&(N-1)) *N
# 			@inbounds tmp=x[bis+pos]*y[j]
# 			s1+=real(tmp)
# 			s2+=imag(tmp)
# 		end
# 		CUDA.@atomic res1[] += s1
# 		CUDA.@atomic res2[] += s2
# 		return nothing
# 	end
# 	kernel = @cuda launch=false  f1(x.data,y.data,N,res1,res2,T)
#     config = launch_configuration(kernel.fun)
#     threads = min(length(x.data), config.threads)
#     blocks = cld(length(x.data), threads)
#     kernel(x.data,y.data,N,res1,res2,T;threads, blocks)
# 	return  CUDA.@allowscalar res1[]+im*res2[]
# end

using CUDA:i32

LinearAlgebra.dot(x::CuDensityMatrixBatch, y::CuDensityMatrixBatch) = begin
    T = Float64
    N=2^nqubits(x)
    res1 = CuArray{T}([zero(T)])
    res2 = CuArray{T}([zero(T)])
    function kernel(x, y, res1,res2, T,N)
        index = threadIdx().x
        thread_stride = blockDim().x
        block_stride = (length(x)-1i32) ÷ gridDim().x + 1i32
        start = (blockIdx().x - 1i32) * block_stride + 1i32
        stop = blockIdx().x * block_stride

        cache1 = CuDynamicSharedArray(T, (thread_stride,))
        cache2 = CuDynamicSharedArray(T, (thread_stride,))
        for j in start-1i32+index:thread_stride:stop
            i=j-1
            bis=i>>N+1
            pos= (i&(N-1)) *N
            @inbounds tmp= x[bis+pos] * y[j]
            @inbounds cache1[index] += real(tmp)
            @inbounds cache2[index] += imag(tmp)
        end
        sync_threads()

        mid = thread_stride
        while true
            mid = (mid - 1i32) ÷ 2i32 + 1i32
            if index <= mid
                @inbounds cache1[index] += cache1[index+mid]
                @inbounds cache2[index] += cache2[index+mid]
            end
            sync_threads()
            mid == 1i32 && break
        end

        if index == 1i32
            CUDA.@atomic res1[] += cache1[1]
            CUDA.@atomic res2[] += cache2[1]
        end
        return nothing
    end
    k = @cuda launch=false kernel(x.data, y.data, res1,res2,T,N)
    config = launch_configuration(k.fun; shmem=(threads) -> threads*sizeof(T))
    threads = min(length(x.data), config.threads)
    blocks = config.blocks
    k(x.data, y.data, res1,res2, T,N; threads=threads, blocks=blocks, shmem=threads*sizeof(T))
    CUDA.@allowscalar res1[]+im*res2[]
end

Base.:one(x::CuDensityMatrixBatch) =begin
	nq=nqubits(x)
	N=2^nq
	nt=nitems(x)
	x1=CUDA.fill(0.,N^2*nt)
	total_itr = N*nt
	@inline function f1(x,N,total_itr)
		index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
		stride = blockDim().x * gridDim().x
		for i = index:stride:total_itr
			j=i-1
			x[j*N+mod(j,N)+1]=1
		end
	end
	kernel = @cuda launch=false  f1(x1,N,total_itr)
    config = launch_configuration(kernel.fun)
    threads = min(total_itr, config.threads)
    blocks = cld(total_itr, threads)
    kernel(x1,N,total_itr; threads, blocks)
	return CuDensityMatrixBatch(CuArray(x1),nq,nt)
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



