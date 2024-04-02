include("short_range_serial.jl")
include("short_range_threaded.jl")
include("long_range_threaded.jl")

const LARGEST_SUPPORTED_NTERMS = 5

Base.:*(m::QubitsOperator, v::StateVector) = m(v)
Base.:*(m::QubitsTerm, v::StateVector) = m(v)

#Base.:*(m::QubitsOperator,v::StateVectorBatch) = m(v)
Base.:*(m::QubitsTerm,v::StateVectorBatch) =m(v)

Base.:*(m::QubitsTerm,v::DensityMatrixBatch) = m(v)

Base.:*(m::QubitsTerm,v::CuStateVectorBatch)=m(v)

Base.:*(m::QubitsTerm,v::CuDensityMatrixBatch)=m(v)

function (m::QubitsTerm)(vr::StateVector)
	v = storage(vr)
	vout = similar(v)
	_apply_qterm_util!(m, v, vout)
	return StateVector(vout, nqubits(vr))
end

function (m::QubitsOperator)(vr::StateVector) 
	v = storage(vr)
	vout = zeros(eltype(v), length(v))
	if _largest_nterm(m) <= LARGEST_SUPPORTED_NTERMS
		_apply_util!(m, v, vout)
	else
		workspace = similar(v)
		for (k, v) in m
			for item in v
			   _apply_qterm_util!(QubitsTerm(k, item), v, workspace) 
			   vout .+= workspace
			end
		end
	end
	return StateVector(vout, nqubits(vr))
end

function (m::QubitsTerm)(vr::StateVectorBatch)
	v = storage(vr)
	vout = similar(v)
	_apply_qterm_util!(m, v, vout,nqubits(vr))
	return StateVectorBatch(vout, nqubits(vr),nitems(vr))
end

function (m::QubitsTerm)(vr::DensityMatrixBatch)
	v = vr.data
	vout = similar(v)
	_apply_qterm_util!(m,v,vout,2*nqubits(vr))
	return DensityMatrixBatch(vout, nqubits(vr),nitems(vr))
end

function (m::QubitsTerm)(vr::CuStateVectorBatch)
	v = storage(vr)
	vout = similar(v)
	_apply_qterm_util!(m,v,vout, nqubits(vr))
	return CuStateVectorBatch(vout,nqubits(vr),nitems(vr))
end

function (m::QubitsTerm)(vr::CuDensityMatrixBatch)
	v = vr.data
	vout = similar(v)
	_apply_qterm_util!(m,v,vout,2*nqubits(vr))
	return CuDensityMatrixBatch(vout, nqubits(vr),nitems(vr))
end

##statevectorBatch


function _largest_nterm(x::QubitsOperator)
	n = 0
	for (k, v) in x.data
		n = max(n, length(k))
	end
	return n
end

function _apply_qterm_util!(m::QubitsTerm, v::AbstractVector, vout::AbstractVector, n::Int)
	tmp = coeff(m)
	@. vout = tmp * v
	if length(v) >= 32
		for (pos, mat) in zip(positions(m), oplist(m))
			_apply_gate_threaded2!(pos, mat, vout, n)
		end	
	else    
		for (pos, mat) in zip(positions(m), oplist(m))
			_apply_gate_2!(pos, mat, vout)
		end			
	end
end

function _apply_qterm_util!(m::QubitsTerm, v::CuArray, vout::CuArray, n::Int)
	tmp = coeff(m)
	@. vout = tmp * v
	for (pos, mat) in zip(positions(m), oplist(m))
		_apply_gate_threaded2!(pos, mat, vout, n)
	end	
end

function _apply_qterm_util!(m::QubitsTerm, v::AbstractVector, vout::AbstractVector)
	tmp = coeff(m)
	@. vout = tmp * v
	if length(v) >= 32
		for (pos, mat) in zip(positions(m), oplist(m))
			_apply_gate_threaded2!(pos, mat, vout)
		end	
	else    
		for (pos, mat) in zip(positions(m), oplist(m))
			_apply_gate_2!(pos, mat, vout)
		end			
	end
end


_apply_util!(m::QubitsOperator, v::AbstractVector, vout::AbstractVector) = (length(v) >= 32) ? _apply_threaded_util!(
    m, v, vout) : _apply_serial_util!(m, v, vout)

