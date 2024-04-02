expectation_value_threaded(pos::Int, m::AbstractMatrix, state::CuArray, n::Int) = _expectation_value_threaded_util(
	pos, m, state, n)

expectation_value_threaded(pos::Tuple{Int}, m::AbstractMatrix, state::CuArray, n::Int) = expectation_value_threaded(
	pos[1], m, state, n)


function _expectation_value_threaded_util(key::Int, U::AbstractMatrix, v::CuArray, n::Int)
    Ls = length(v)
    L = 1<<n
    nitem = div(Ls,L)
    u = vec(CuArray(U))
    pos = 2^(key-1)
    m1= pos-1
    m2 = xor(L - 1, 2 * pos - 1)
    total_itr=div(Ls,2)
    r = CUDA.similar(v,L>>1,nitem)
    @inline function f1(u, v, pos, m1, m2, total_itr, n, r)
        index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        stride = gridDim().x * blockDim().x
		Nx,Ny = size(r)
		cind = CartesianIndices((Nx,Ny))
        for j = index:stride:total_itr
            i = cind[j][1]
			k = cind[j][2]
			l = j - 1
            posa = (2 * l & m2) | (l & m1) + 1 + l>>(n-1)<<n
            @inbounds r[i,k] = v[posa]'*v[posa]*u[1] + 2*real(v[posa]'*v[posa+ pos]*u[3])+v[posa+ pos]'*v[posa+ pos]*u[4]
        end
    end
    kernel = @cuda launch=false f1(u, v, pos, m1, m2, total_itr, n, r)
    config = launch_configuration(kernel.fun)
    threads = min(length(r), config.threads)
    blocks = cld(length(r), threads)
    CUDA.@sync begin
        kernel(u, v, pos, m1, m2, total_itr, n, r; threads, blocks)
    end
    return Array(real.(sum(r,dims=1)'))
end
