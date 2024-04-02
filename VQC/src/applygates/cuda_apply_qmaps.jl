"""
	apply!(x::QuantumMap, state::DensityMatrix) 
	apply a generc quantum channel on the quantum state
"""

#DensityMatrixBatch
function apply!(x::QuantumMap, state::CuDensityMatrixBatch) 
	@assert _check_pos_range(x, nqubits(state))
	T = eltype(state)
	if (eltype(x) <: Complex) && (T <: Real)
		state = convert(CuDensityMatrixBatch{Complex{T}}, state)
	end
	_qmap_apply_threaded_bc!(x, state.data, nqubits(state))
	return state
end

function _qmap_apply_threaded_bc!(x::QuantumMap{N}, s::CuArray, n::Int) where N
	pos = ordered_positions(x)
	pos2 = ntuple(i->pos[i]+n, N)
	all_pos = (pos..., pos2...)
	m = ordered_supermat(x)
	_apply_gate_threaded2!(all_pos, m, s, 2*n)
end

##DensityMatrixBatch

function apply_dagger!(x::QuantumMap, state::CuDensityMatrixBatch) 
	@assert _check_pos_range(x, nqubits(state))
	T = eltype(state)
	if (eltype(x) <: Complex) && (T <: Real)
		state = convert(CuDensityMatrix{Complex{T}}, state)
	end
	_qmap_apply_dagger_threaded_bc!(x, state.data, nqubits(state))
	return state
end

function _qmap_apply_dagger_threaded_bc!(x::QuantumMap{N}, s::CuArray, n::Int) where N
	pos = ordered_positions(x)
	pos2 = ntuple(i->pos[i]+n, N)
	all_pos = (pos..., pos2...)
	m = ordered_supermat(x)
	_apply_gate_threaded2!(all_pos, m', s, 2*n)
end


function apply_inverse!(x::QuantumMap, state::CuDensityMatrixBatch) 
	@assert _check_pos_range(x, nqubits(state))
	T = eltype(state)
	if (eltype(x) <: Complex) && (T <: Real)
		state = convert(CuDensityMatrix{Complex{T}}, state)
	end
	_qmap_apply_inverse_threaded_bc!(x, state.data, nqubits(state))
	return state
end

function _qmap_apply_inverse_threaded_bc!(x::QuantumMap{N}, s::CuArray, n::Int) where N
	pos = ordered_positions(x)
	pos2 = ntuple(i->pos[i]+n, N)
	all_pos = (pos..., pos2...)
	m = ordered_supermat(x)
	_apply_gate_threaded2!(all_pos, inv(m), s, 2*n)
end