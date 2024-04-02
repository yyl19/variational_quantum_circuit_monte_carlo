# gradient of expectation values
"""
	diff for QubitsTerm
"""

@adjoint expectation(m::QubitsTerm, state::Union{StateVector, StateVectorBatch,DensityMatrix, DensityMatrixBatch, CuStateVectorBatch, CuDensityMatrixBatch}) = _qterm_expec_util(m, state)

function _qterm_expec_util(m::QubitsTerm, state::StateVector)
	if length(positions(m)) <= LARGEST_SUPPORTED_NTERMS
	    return expectation(m, state), z -> (nothing, storage( (conj(z) * m + z * m') * state ) )
	else
		v = m * state
		return dot(state, v), z -> begin
		   m1 = conj(z) * m
		   m2 = z * m'
		   _apply_qterm_util!(m1, storage(state), storage(v))
		   v2 = storage( m2 * state )
		   v2 .+= storage(v)
		   return (nothing, v2)
		end
	end
end

function _qterm_expec_util(m::QubitsTerm,state::StateVectorBatch)
	if length(positions(m)) <= LARGEST_SUPPORTED_NTERMS
	    return expectation(m, state), z -> begin
			z1=kron(z,ones(2^nqubits(state)))
			return (nothing, conj(z1) .* storage(m * state) + z1 .* storage(m' * state))
		end
	end
end

function _qterm_expec_util(m::QubitsTerm,state::CuStateVectorBatch)
	if length(positions(m)) <= LARGEST_SUPPORTED_NTERMS
	    return expectation(m, state), z -> begin
			z1=Array(vec(z))
			y1=m*state
			y2=m'*state
			apply_coefficient!(conj(z1),y1)
			apply_coefficient!(z1,y2)
			return (nothing,storage(y1).+storage(y2))
		end
	end
end

_qterm_expec_util(m::QubitsTerm, state::DensityMatrix) = expectation(m, state), z -> (nothing,  (z * matrix(nqubits(state),m)) )

_qterm_expec_util(m::QubitsTerm, state::DensityMatrixBatch) = expectation(m, state), z -> begin
    nq = nqubits(state)
	z1=kron(z,ones(2^(2*nq)))
	return (nothing, storage(z1* (m*one(state))))
end

_qterm_expec_util(m::QubitsTerm, state::CuDensityMatrixBatch) = expectation(m, state), z -> begin
	y=m*one(state)
	apply_coefficient!(Array(vec(z)), y)
	return (nothing, storage(y))
end


#实现z分别乘以每个dm-batch的过程，直接实现这个过程storage(z1 * (m*one(state)))
# function apply_coefficient(z::Vector,nq::Int,nt::Int)


	
# 	return 
# end

"""
	diff for QubitsOperator
"""


@adjoint expectation(m::QubitsOperator, state::Union{StateVector, DensityMatrix}) = _qop_expec_util(m, state)


function _qop_expec_util(m::QubitsOperator, state::StateVector)
	if _largest_nterm(m) <= LARGEST_SUPPORTED_NTERMS
		return expectation(m, state), z -> (nothing, storage( (conj(z) * m + z * m') * state ) )
	else
		state = storage(state)
		workspace = similar(state)
		state_2 = zeros(eltype(state), length(state))
		for (k, v) in m.data
		    for item in v
		    	_apply_qterm_util!(QubitsTerm(k, item[1], item[2]), state, workspace)
		    	state_2 .+= workspace
		    end
		end
		r = dot(state, state_2)
		return r, z -> begin
			if ishermitian(m)
			    state_2 .*= (conj(z) + z)
			else
				state_2 .*= conj(z)
				md = m'
				for (k, v) in md.data
					for item in v
						_apply_qterm_util!(QubitsTerm(k, item[1], item[2]), state, workspace)
						@. state_2 += z * workspace
		    		end
		    	end
			end
		    return (nothing, state_2)    
		end
	end	
end

_qop_expec_util(m::QubitsOperator, state::DensityMatrix) = expectation(m, state), z -> (
nothing, storage( (z * m') * DensityMatrix(one(storage(state)), nqubits(state)) ) )
