
include("expec_dm_serial.jl")
include("cuda_expec_dm_serial.jl")

function expectation(m::QubitsTerm,dm::DensityMatrix)
	isempty(m) && return tr(dm)
	if length(positions(m)) <= LARGEST_SUPPORTED_NTERMS
		r = expectation_value_serial(Tuple(positions(m)),_get_mat(m),storage(dm))
		return r[1][]
	else
		return tr(m*dm)
	end
end

function expectation(m::QubitsTerm,dm::DensityMatrixBatch)
	isempty(m) && return tr(dm)
	if length(positions(m)) <= LARGEST_SUPPORTED_NTERMS
		r = expectation_value_serial(Tuple(positions(m)),_get_mat(m),storage(dm))
		rr= [k[] for k in r]
		return rr
	else
		return tr(m*dm)
	end
end

function expectation(m::QubitsTerm,dm::CuDensityMatrixBatch)
	isempty(m) && return tr(dm)
	if length(positions(m)) <= LARGEST_SUPPORTED_NTERMS
		r = expectation_value_serial(Tuple(positions(m)),_get_mat(m),storage(dm))
		#rr= [k[] for k in r]
		return r
	else
		return tr(m*dm)
	end
end