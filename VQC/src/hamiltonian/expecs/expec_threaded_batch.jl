
"""
    applys when key > 3, U is the transposed op
"""
function _expectation_value_util_H(key::Int, U::AbstractMatrix, v::AbstractVector, n::Int)
    L = 1<<n
    Ls = length(v)
    sizek = 1 << (key - 1)
    mask0 = sizek - 1
    mask1 = xor(L - 1, 2 * sizek - 1)
    f(ist::Int, ifn::Int, pos::Int, m1::Int, m2::Int, mat::AbstractMatrix, p::AbstractVector, n::Int) = begin
    	r = zero(eltype(p))
        @inbounds for i in ist:ifn
            l = (16 * i & m2) | (8 * i & m1) + 1 + i>>(n-4)<<n
            l1 = l + pos
            vi = SMatrix{8, 2}(p[l], p[l+1], p[l+2], p[l+3], p[l+4], p[l+5], p[l+6], p[l+7],
            p[l1], p[l1+1], p[l1+2], p[l1+3], p[l1+4], p[l1+5], p[l1+6], p[l1+7])

            vi_t = transpose(vi)
            @fastmath r += dot(vi_t, mat, vi_t)
        end
        return r
    end

	return parallel_sum(eltype(v), 1<<(n-4) ,div(Ls, 16), Threads.nthreads(), f, sizek, mask0, mask1, U, v, n)
end



"""
    applys when key <= 3, U is the transposed op
"""

function _expectation_value_util_L(key::Int, U::AbstractMatrix, v::AbstractVector, n::Int)
    L = length(v)
    f1(ist::Int, ifn::Int, mat::AbstractMatrix, p::AbstractVector, n::Int) = begin
    	r = zero(eltype(p))
        @inbounds for i in ist:ifn
            l = 16 * i + 1
            vi = SMatrix{2, 8}(p[l], p[l+1], p[l+2], p[l+3], p[l+4], p[l+5], p[l+6], p[l+7],
            p[l+8], p[l+9], p[l+10], p[l+11], p[l+12], p[l+13], p[l+14], p[l+15])

            @fastmath r += dot(vi, mat, vi)
        end
        return r
    end
    f2(ist::Int, ifn::Int, mat::AbstractMatrix, p::AbstractVector, n::Int) = begin
    	r = zero(eltype(p))
        @inbounds for i in ist:ifn
            l = 16 * i + 1
            vi = SMatrix{2, 8}(p[l], p[l+2], p[l+1], p[l+3], p[l+4], p[l+6], p[l+5], p[l+7],
            p[l+8], p[l+10], p[l+9], p[l+11], p[l+12], p[l+14], p[l+13], p[l+15])

            @fastmath r += dot(vi, mat, vi)
        end
        return r
    end
    f3(ist::Int, ifn::Int, mat::AbstractMatrix, p::AbstractVector, n::Int) = begin
    	r = zero(eltype(p))
        @inbounds for i in ist:ifn
            l = 16 * i + 1
            vi = SMatrix{2, 8}(p[l], p[l+4], p[l+1], p[l+5], p[l+2], p[l+6], p[l+3], p[l+7],
            p[l+8], p[l+12], p[l+9], p[l+13], p[l+10], p[l+14], p[l+11], p[l+15])

            @fastmath r += dot(vi, mat, vi)
        end
        return r
    end
    if key == 1
        f = f1
    elseif key == 2
        f = f2
    elseif key == 3
        f = f3
    else
        error("qubit position $key not allowed for L.")
    end
    parallel_sum(eltype(v), 1<<(n-4),div(L, 16), Threads.nthreads(), f, U, v, n)
end

function _expectation_value_threaded_util(q0::Int, U::AbstractMatrix, v::AbstractVector, n::Int)
    if q0 > 3
        return _expectation_value_util_H(q0, SMatrix{2,2, eltype(v)}(U), v, n)
    else
        return _expectation_value_util_L(q0, SMatrix{2,2, eltype(v)}(U), v, n)
    end
end
_expectation_value_threaded_util(q0::Tuple{Int}, U::AbstractMatrix, v::AbstractVector, n::Int) = _expectation_value_threaded_util(
    q0[1], U, v, n)

expectation_value_threaded(pos::Int, m::AbstractMatrix, state::AbstractVector, n::Int) = _expectation_value_threaded_util(
	pos, m, state, n)

expectation_value_threaded(pos::Tuple{Int}, m::AbstractMatrix, state::AbstractVector, n::Int) = expectation_value_threaded(
	pos[1], m, state, n)





