struct CuStateVectorBatch{T <: Number} 
    data::CuArray{T}
    nqubits::Int
    nitems::Int

function CuStateVectorBatch{T}(data::CuArray{<:Number}, nqubits::Int,nitems::Int) where {T <: Number}
    @assert length(data) == 2^nqubits*nitems
    new{T}(convert(CuArray{T}, data), nqubits,nitems)
end

end

#StateVectorBatch 构造方法
CuStateVectorBatch(data::CuArray{T}, nqubits::Int,nitems::Int) where {T <: Number} = CuStateVectorBatch{T}(data, nqubits,nitems)

CuStateVectorBatch(data::CuMatrix{T}, nqubits::Int,nitems::Int) where{T <: Number} = CuStateVectorBatch{T}(vec(data), nqubits,nitems)

#StateVectorBatch 只接受同时输入data,nqubits,nitems 三个输入的构造方法。
# StateVectorBatch(nqubits::Int) = StateVectorBatch(ComplexF64, nqubits, nitems)

#初始化一个stateVectorBatch,所有态都为0态，输入nqubits 和 nitems
function CuStateVectorBatch{T}(nqubits::Int,nitems::Int) where T
    datas=zeros(T,2^nqubits,nitems)
    datas[1,:].=1.
    return CuStateVectorBatch(CuVector(vec(datas)),nqubits,nitems)
end

# @adjoint StateVectorBatch(nqubits::Int,nitems::Int)=StateVectorBatch(nqubits::Int,nitems::Int),z->(nothing,nothing)

#将StateVectorBatch 从CPU转移到GPU
CuStateVectorBatch(x::StateVectorBatch) = CuStateVectorBatch(CuArray(x.data), nqubits(x),nitems(x))

storage(x::CuStateVectorBatch) = x.data
QuantumCircuits.nqubits(x::CuStateVectorBatch) = x.nqubits
nitems(x::CuStateVectorBatch) = x.nitems

@adjoint CuStateVectorBatch(nqubits::Int,nitems::Int)=CuStateVectorBatch(nqubits::Int,nitems::Int),z->(nothing,nothing)

Base.eltype(::Type{CuStateVectorBatch{T}}) where T = T
Base.eltype(x::CuStateVectorBatch) = eltype(typeof(x))

# #取对应下标元素 (有啥应用场景)
# Base.getindex(x::StateVectorBatch, j::Int) = getindex(storage(x), j)

# #设置值，感觉意义也不大，暂时先不管
# Base.setindex!(x::StateVectorBatch, v, j::Int) = setindex!(storage(x), v, j)
#GPU里面好像有问题，直接索引 @allowscaler

#
Base.convert(::Type{CuStateVectorBatch{T}}, x::CuStateVectorBatch) where {T<:Number} = CuStateVectorBatch(convert(CuArray{T}, storage(x)), nqubits(x), nitems(x))

Base.copy(x::CuStateVectorBatch) = CuStateVectorBatch(copy(storage(x)), nqubits(x), nitems(x))

# Base.cat(v::StateVectorBatch) = v
# function Base.cat(v::StateVectorBatch...)
#     a, b = _qcat_util(storage.(v)...)
#     return StateVectorBatch(kron(a, b))
# end

# Base.isapprox(x::StateVectorBatch, y::StateVectorBatch; kwargs...) = isapprox(storage(x), storage(y); kwargs...)
# Base.:(==)(x::StateVectorBatch, y::StateVectorBatch) = storage(x) == storage(y)


#Base.:+(x::StateVectorBatch, y::StateVectorBatch) = StateVectorBatch(storage(x) + storage(y), nqubits(x))
#Base.:-(x::StateVectorBatch, y::StateVectorBatch) = StateVectorBatch(storage(x) - storage(y), nqubits(x))
#Base.:*(x::StateVectorBatch, y::Number) = StateVectorBatch(storage(x) * y, nqubits(x))
#Base.:*(x::Number, y::StateVectorBatch) = y * x
#Base.:/(x::StateVectorBatch, y::Number) = StateVectorBatch(storage(x) / y, nqubits(x))
#Base.:*(m::AbstractMatrix, x::StateVectorBatch) = StateVectorBatch( m * storage(x), nqubits(x) )


function apply_coefficient!(z::Vector, y::CuStateVectorBatch)
	z1 = CuArray(convert.(ComplexF64,z))
	nq = nqubits(y)
	v=y.data
	@inline function f1(v,z1,n)
		index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
		stride = blockDim().x * gridDim().x
		for j = index:stride:length(v)
			i = j-1
			v[j] *= z1[i>>n+1]
		end
	end
	kernel = @cuda launch=false  f1(v,z1,nq)
    config = launch_configuration(kernel.fun)
    threads = min(length(v), config.threads)
    blocks = cld(length(v), threads)
    kernel(v,z1,nq; threads, blocks)
	return y
end



#rewrite

# LinearAlgebra.norm(x::StateVectorBatch) = norm(storage(x))

LinearAlgebra.dot(x::CuStateVectorBatch, y::CuStateVectorBatch) = dot(storage(x), storage(y))
# LinearAlgebra.normalize!(x::StateVectorBatch) = (normalize!(storage(x)); x)
# LinearAlgebra.normalize(x::StateVectorBatch) = StateVectorBatch(normalize(storage(x)), nqubits(x))

"""
    fidelity(x, y) 
    tr(√x * √y) if x and y are density matrices
    ⟨x|y⟩^2 if x and y are pure states
"""
#fidelity(x::StateVectorBatch, y::StateVectorBatch) = abs2(dot(x, y))
#distance2(x::StateVectorBatch, y::StateVectorBatch) = _distance2(x, y)
#distance(x::StateVectorBatch, y::StateVectorBatch) = _distance(x, y)


# encoding
#onehot_encoding(::Type{T}, n::Int) where {T <: Number} = StateVectorBatch(onehot(T, 2^n, 1), n)
#onehot_encoding(n::Int) = onehot_encoding(ComplexF64, n)
#onehot_encoding(::Type{T}, i::AbstractVector{Int}) where {T <: Number} = StateVectorBatch(onehot(T, 2^(length(i)), _sub2ind(i)+1), length(i))
#onehot_encoding(i::AbstractVector{Int})= onehot_encoding(ComplexF64, i)

#function onehot(::Type{T}, L::Int, pos::Int) where T
#     r = zeros(T, L)
#     r[pos] = one(T)
#     return r
# end


"""
    kernal_mapping(s::Real) = [cos(s*pi/2), sin(s*pi/2)]
    This maps 0 -> [1, 0] (|0>), and 1 -> [0, 1] (|1>)
"""
# kernal_mapping(s::Real) = [cos(s*pi/2), sin(s*pi/2)]

"""
    qstate(::Type{T}, thetas::AbstractVector{<:Real}) where {T <: Number}
Return a product quantum state of [[cos(pi*theta/2), sin(pi*theta/2)]] for theta in thetas]\n
Example: qstate(Complex{Float64}, [0.5, 0.7])
"""
# function qubit_encoding(::Type{T}, i::AbstractVector{<:Real}) where {T <: Number}
#     isempty(i) && throw("empty input.")
#     v = [convert(Vector{T}, item) for item in kernal_mapping.(i)]
#     (length(v) == 1) && return StateVectorBatch{T}(v[1])
#     a, b = _qcat_util(v...)
#     return StateVectorBatch(kron(a, b))
# end  
# qubit_encoding(mpsstr::AbstractVector{<:Real}) = qubit_encoding(ComplexF64, mpsstr)


# function reset!(x::StateVectorBatch)
#     fill!(storage(x), zero(eltype(x)))
#     x[1] = one(eltype(x))
#     return x
# end
# function reset_onehot!(x::StateVectorBatch, i::AbstractVector{Int})
#     @assert nqubits(x) == length(i)
#     pos = _sub2ind(i) + 1
#     fill!(storage(x), zero(eltype(x)))
#     x[pos] = one(eltype(x))
#     return x
# end

# function reset_qubit!(x::StateVectorBatch, i::AbstractVector{<:Real})
#     @assert nqubits(x) == length(i)
#     if length(i) == 1
#         copyto!(storage(x), kernal_mapping(i[1]))
#         return x
#     end
#     a, b = _qcat_util(kernal_mapping.(i)...)
#     m = length(a)
#     n = length(b)
#     xs = storage(x)
#     for j in 1:m
#         n_start = (j-1) * n + 1
#         n_end = j * n
#         tmp = a[j]
#         @. xs[n_start:n_end] = tmp * b
#     end
#     return x
# end

# function amplitude(s::StateVectorBatch, i::AbstractVector{Int}; scaling::Real=sqrt(2))
#     @assert length(i)==nqubits(s)
#     idx = _sub2ind(i)
#     return scaling==1 ? s[idx] : s[idx] * scaling^(nqubits(s))
# end
# amplitudes(s::StateVectorBatch) = storage(s)

# function rand_state(::Type{T}, n::Int) where {T <: Number}
#     (n >= 1) || error("number of qubits must be positive.")
#     v = randn(T, 2^n)
#     v ./= norm(v)
#     return StateVectorBatch(v, n)
# end
# rand_state(n::Int) = rand_state(ComplexF64, n)

# function QuantumCirCuits.permute(x::StateVectorBatch, newindex::Vector{Int})
#     n = nqubits(x)
#     L = length(storage(x))
#     return StateVectorBatch(reshape(permute(reshape(storage(x), ntuple(i->2,n)), newindex), L), n)
# end


# function _sub2ind(v::AbstractVector{Int})
#     @assert _is_valid_indices(v)
#     isempty(v) && error("input index is empty.")
#     L = length(v)
#     r = v[1]
#     for i in 2:L
#         r |= v[i] << (i-1)
#     end
#     return r
# end

# function _qcat_util(vr::Union{AbstractVector, AbstractMatrix}...)
#     v = reverse(vr)
#     L = length(v)
#     # println("$(typeof(v)), $L")
#     (L >= 2) || error("something wrong.")
#     Lh = div(L, 2)
#     a = v[1]
#     for i in 2:Lh
#         a = kron(a, v[i])
#     end
#     b = v[Lh + 1]
#     for i in Lh+2 : L
#         b = kron(b, v[i])
#     end
#     return a, b
# end

# function _is_valid_indices(i::AbstractVector{Int})
#     for s in i
#         (s == 0 || s == 1) || return false 
#     end   
#     return true
# end

# _nqubits(s::AbstractVector) = begin
#     n = round(Int, log2(length(s)))
#     (2^n == length(s)) || error("state can not be interpretted as a qubit state.")
#     return n
# end
