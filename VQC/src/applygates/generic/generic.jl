include("serial_short_range.jl")
include("threaded_short_range.jl")
include("threaded_long_range.jl")
include("cuda_threaded_short_range.jl")
include("batch_threaded_short_range.jl")

function apply!(x::Gate, state::StateVector) 
	@assert _check_pos_range(x, nqubits(state))
	T = eltype(state)
	if (eltype(x) <: Complex) && (T <: Real)
		state = convert(StateVector{Complex{T}}, state)
	end
	apply_threaded!(x, storage(state))
	return state
end

function apply!(x::Gate, state::StateVectorBatch) 
	@assert _check_pos_range(x, nqubits(state))
	T = eltype(state)
	if (eltype(x) <: Complex) && (T <: Real)
		state = convert(StateVectorBatch{Complex{T}}, state)
	end
	apply_threaded!(x, storage(state), nqubits(state))
	return state
end

function apply!(x::Gate, state::CuStateVectorBatch) 
	@assert _check_pos_range(x, nqubits(state))
	T = eltype(state)
	if (eltype(x) <: Complex) && (T <: Real)
		state = convert(CuStateVectorBatch{Complex{T}}, state)
	end
	apply_threaded!(x, storage(state), nqubits(state))
	return state
end


function apply!(x::Gate, state::DensityMatrix) 
	@assert _check_pos_range(x, nqubits(state))
	T = eltype(state)
	if (eltype(x) <: Complex) && (T <: Real)
		state = convert(DensityMatrix{Complex{T}}, state)
	end
	_dm_apply_threaded!(x, state.data, nqubits(state))
	return state
end

function apply!(x::Gate, state::DensityMatrixBatch)
	@assert _check_pos_range(x, nqubits(state))
	T = eltype(state)
	if (eltype(x) <: Complex) && (T <: Real)
		state = convert(DensityMatrixBatch{Complex{T}}, state)
	end
	_dm_apply_threaded!(x, state.data, nqubits(state))
	return state
end

function apply!(x::Gate, state::CuDensityMatrixBatch)
	@assert _check_pos_range(x, nqubits(state))
	T = eltype(state)
	if (eltype(x) <: Complex) && (T <: Real)
		state = convert(CuDensityMatrixBatch{Complex{T}}, state)
	end
	_dm_apply_threaded!(x, state.data, nqubits(state))
	return state
end

function apply!(circuit::QCircuit, state::Union{StateVector, StateVectorBatch, DensityMatrix, DensityMatrixBatch, CuStateVectorBatch, CuDensityMatrixBatch})
	for gate in circuit
		state = apply!(gate, state)
	end
	return state
end

Base.:*(circuit::QCircuit, state::Union{StateVector, StateVectorBatch, DensityMatrix, DensityMatrixBatch,CuStateVectorBatch, CuDensityMatrixBatch}) = apply!(circuit, copy(state))



function _check_pos_range(x, n::Int)
	pos = ordered_positions(x)
	(length(pos) > 5) && throw("only implement 5-qubit gates and less currently.")
	return (pos[1] >= 1) && (pos[end] <= n)
end



apply_serial!(x::Gate, state::AbstractVector) = _apply_gate_2!(ordered_positions(x), ordered_mat(x), state)

apply_serial!(x::Gate, state::StateVector) = (apply_serial!(x, storage(state)); state)


"""
    currently support mostly 5-qubit gate
"""
apply_threaded!(x::Gate, s::AbstractVector) = (length(s) >= 32) ? _apply_gate_threaded2!(ordered_positions(x), ordered_mat(x), s) : apply_serial!(x, s)

apply_threaded!(x::Gate, s::AbstractVector, n::Int) = (length(s) >= 32) ? _apply_gate_threaded2!(ordered_positions(x), ordered_mat(x), s, n) : apply_serial!(x, s)

apply_threaded!(x::Gate,v::CuArray,n::Int)=_apply_gate_threaded2!(ordered_positions(x),ordered_mat(x), v, n)

# unitary gate operation on density matrix, 这里有问题，我改成了适配densitymatrixBatch的类型了，densitymatrix会出现一些问题。
function _dm_apply_threaded!(x::Gate{N}, s::AbstractVector, n::Int) where N
	pos = ordered_positions(x)
	m = ordered_mat(x)
	if length(s) >= 1024
		_apply_gate_threaded2!(pos, m, s, 2*n)
		_apply_gate_threaded2!(ntuple(i->pos[i]+n, N), conj(m), s, 2*n)
	else
		_apply_gate_2!(pos, m, s)
		_apply_gate_2!(ntuple(i->pos[i]+n, N), conj(m), s)
	end
end

#Apply general quantum gate on DensityMatrixBatch with GPU  
function _dm_apply_threaded!(x::Gate{N}, s::CuArray, n::Int) where N
	pos = ordered_positions(x)
	m = ordered_mat(x)
	_apply_gate_threaded2!(pos, m, s, 2*n)
	_apply_gate_threaded2!(ntuple(i->pos[i]+n, N), conj(m), s, 2*n)
end

function _dm_apply_threaded!(x::Union{XGate,ZGate,CNOTGate,ERyGate}, s::AbstractVector, n::Int)
	pos = ordered_positions(x)
	l=length(pos)
	apply_threaded!(x, s, 2*n)
	pos_tmp=ntuple(i->pos[i]+n, l)
	x_tmp=change_positions(x,Dict(pos[i]=>pos_tmp[i] for i in 1:l))
	apply_threaded!(x_tmp,s,2*n)
end

function _dm_apply_threaded!(x::Union{XGate,ZGate,CNOTGate,ERyGate}, s::CuArray, n::Int)
	pos = ordered_positions(x)
	l=length(pos)
	apply_threaded!(x, s, 2*n)
	pos_tmp=ntuple(i->pos[i]+n, l)
	x_tmp=change_positions(x,Dict(pos[i]=>pos_tmp[i] for i in 1:l))
	apply_threaded!(x_tmp,s,2*n)
end
                  

#为了更方便地执行密度矩阵的自动微分，效果等价于Uρ
function _dm_apply_threaded_left!(x::Gate, s::AbstractVector)
	pos = ordered_positions(x)
	m = ordered_mat(x)
	if length(s) >= 1024
		_apply_gate_threaded2!(pos, m, s)
	else
		_apply_gate_2!(pos, m, s)
	end
end

function _dm_apply_threaded_left!(x::Gate, s::AbstractVector, n::Int)
	pos = ordered_positions(x)
	m = ordered_mat(x)
	if length(s) >= 1024
		_apply_gate_threaded2!(pos, m, s, 2*n)
	else
		_apply_gate_2!(pos, m, s)
	end
end

function _dm_apply_threaded_left!(x::Gate, s::CuVector, n::Int)
	pos = ordered_positions(x)
	m = ordered_mat(x)
	if length(s) >= 1024
		_apply_gate_threaded2!(pos, m, s, 2*n)
	else
		_apply_gate_2!(pos, m, s)
	end
end

_dm_apply_threaded_left!(x::Union{CNOTGate,XGate,ZGate}, s::AbstractVector, n::Int)=apply_threaded!(x, s, 2*n)

_dm_apply_threaded_left!(x::Union{CNOTGate,XGate,ZGate}, s::CuVector, n::Int)=apply_threaded!(x, s, 2*n)
