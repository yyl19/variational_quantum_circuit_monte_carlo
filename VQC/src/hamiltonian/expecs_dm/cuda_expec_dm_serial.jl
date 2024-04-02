function _expectation_value_util(key::Int, ms::AbstractMatrix, state::CuMatrix)
	(size(ms, 1) == size(ms, 2)) || error("observable must be a square matrix.")
	(size(ms, 1) == 2) || error("input must be a 2 by 2 matrix.")
	L1,L2 = size(state)
	nitem = div(L2,L1)
	pos = 2^(key-1)
	m1 = pos-1
	m2 = xor(L1 - 1, 2 * pos - 1)
	r = CUDA.similar(state,L1>>1,nitem)
	m=CuArray(ms)
	@inline function f1(m,state,r,m1,m2,pos,L1)
		index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
		stride = blockDim().x * gridDim().x
		Nx,Ny = size(r)
		cind = CartesianIndices((Nx,Ny))
		for j = index:stride:Nx*Ny
			i = cind[j][1]
			k = cind[j][2]
			l = j - 1
			posa = (2 * l & m2) | (l & m1) + 1 
			posb = posa + pos
			bis = (k-1)*L1
			@inbounds r[i,k]= m[1,1]*state[posa, posa+bis] + m[1,2]*state[posb, posa+bis]+ m[2,1]*state[posa, posb+bis]+m[2,2]*state[posb, posb+bis]
		end
	end
	kernel = @cuda launch=false  f1(m,state,r,m1,m2,pos,L1)
    config = launch_configuration(kernel.fun)
    threads = min(length(r), config.threads)
    blocks = cld(length(r), threads)
    kernel(m,state,r,m1,m2,pos,L1; threads, blocks)
	return Array(real.(sum(r,dims=1)'))
end




#one
expectation_value_serial(pos::Int, m::AbstractMatrix, state::CuMatrix, state_c::CuMatrix) = _expectation_value_util(
	pos, Matrix{eltype(state)}(m), state, state_c)

expectation_value_serial(pos::Int, m::AbstractMatrix, state::CuMatrix) = _expectation_value_util(
	pos, Matrix{eltype(state)}(m), state)



expectation_value_serial(pos::Tuple{Int}, m::AbstractMatrix, state::CuMatrix, state_c::CuMatrix) = expectation_value_serial(
	pos[1], m, state, state_c)

expectation_value_serial(pos::Tuple{Int}, m::CuMatrix, state::CuMatrix) = expectation_value_serial(
	pos[1], m, state)

