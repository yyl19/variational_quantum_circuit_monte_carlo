# one, two, three-qubit gate operations
"""
    applys when key > 3, U is the transposed op
"""

function _apply_onebody_gate_H!(key::Int, U::AbstractMatrix, v::AbstractVector, n::Int)
    L = 1<<n 
    Ls=length(v)
    sizek = 1 << (key - 1) #1<<n 表示2的n次方。
    mask0 = sizek - 1
    mask1 = xor(L - 1, 2 * sizek - 1)
    f(ist::Int, ifn::Int, pos::Int, m1::Int, m2::Int, mat::AbstractMatrix, p::AbstractVector, n::Int) = begin
        @inbounds for i in ist:ifn
            l = (16 * i & m2) | (8 * i & m1) + 1 + i>>(n-4)<<n
            l1 = l + pos
            vi = SMatrix{8, 2}(p[l], p[l+1], p[l+2], p[l+3], p[l+4], p[l+5], p[l+6], p[l+7],
            p[l1], p[l1+1], p[l1+2], p[l1+3], p[l1+4], p[l1+5], p[l1+6], p[l1+7])

            @fastmath p[l], p[l+1], p[l+2], p[l+3], p[l+4], p[l+5], p[l+6], p[l+7],
            p[l1], p[l1+1], p[l1+2], p[l1+3], p[l1+4], p[l1+5], p[l1+6], p[l1+7] = vi * mat
        end
    end

    total_itr = div(Ls, 16)
    parallel_run(total_itr, Threads.nthreads(), f, sizek, mask0, mask1, U, v, n)
end

"""
    applys when key <= 3, U is the not transposed op
"""

function _apply_onebody_gate_L!(key::Int, U::AbstractMatrix, v::AbstractVector, n::Int)
    Ls = length(v)
    L = 1<<n
    f1(ist::Int, ifn::Int, mat::AbstractMatrix, p::AbstractVector, n::Int) = begin
        @inbounds for i in ist:ifn
            l = 16 * i + 1
            vi = SMatrix{2, 8}(p[l], p[l+1], p[l+2], p[l+3], p[l+4], p[l+5], p[l+6], p[l+7],
            p[l+8], p[l+9], p[l+10], p[l+11], p[l+12], p[l+13], p[l+14], p[l+15])
            @fastmath p[l:(l+15)]= mat * vi
        end
    end
    f2(ist::Int, ifn::Int, mat::AbstractMatrix, p::AbstractVector, n::Int) = begin
        @inbounds for i in ist:ifn
            l = 16 * i + 1
            vi = SMatrix{2, 8}(p[l], p[l+2], p[l+1], p[l+3], p[l+4], p[l+6], p[l+5], p[l+7],
            p[l+8], p[l+10], p[l+9], p[l+11], p[l+12], p[l+14], p[l+13], p[l+15])

            @fastmath p[l], p[l+2], p[l+1], p[l+3], p[l+4], p[l+6], p[l+5], p[l+7],
            p[l+8], p[l+10], p[l+9], p[l+11], p[l+12], p[l+14], p[l+13], p[l+15] = mat * vi
        end
    end
    f3(ist::Int, ifn::Int, mat::AbstractMatrix, p::AbstractVector, n::Int) = begin
        @inbounds for i in ist:ifn
            l = 16 * i + 1
            vi = SMatrix{2, 8}(p[l], p[l+4], p[l+1], p[l+5], p[l+2], p[l+6], p[l+3], p[l+7],
            p[l+8], p[l+12], p[l+9], p[l+13], p[l+10], p[l+14], p[l+11], p[l+15])

            @fastmath p[l], p[l+4], p[l+1], p[l+5], p[l+2], p[l+6], p[l+3], p[l+7],
            p[l+8], p[l+12], p[l+9], p[l+13], p[l+10], p[l+14], p[l+11], p[l+15] = mat * vi
        end
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
    total_itr = div(Ls, 16)
    parallel_run(total_itr, Threads.nthreads(), f, U, v, n)
end


_apply_gate_threaded2!(q0::Tuple{Int}, U::AbstractMatrix, v::AbstractVector, n::Int)= _apply_gate_threaded2!(q0[1], U, v, n)

function _apply_gate_threaded2!(q0::Int, U::AbstractMatrix, v::AbstractVector,n::Int)
    if q0 > 3
        return _apply_onebody_gate_H!(q0, SMatrix{2,2, eltype(v)}(transpose(U)), v, n)
    else
        return _apply_onebody_gate_L!(q0, SMatrix{2,2, eltype(v)}(U), v, n)
    end
end


"""
    applys when both keys > 3, U is the transposed op
"""

function _apply_twobody_gate_HH!(key::Tuple{Int, Int}, U::AbstractMatrix, v::AbstractVector, n::Int)
    L = 1<<n
    Ls = length(v)
    q1, q2 = key
    pos1, pos2 = 1 << (q1-1), 1 << (q2-1)
    # stride2, stride3 = pos1 << 1, pos2 << 1
    mask0 = pos1 - 1
    mask1 = xor(pos2 - 1, 2 * pos1 - 1)
    mask2 = xor(L - 1, 2 * pos2 - 1)
    # println("pos1=$pos1, pos2=$pos2, m0=$mask0, m1=$mask1, m2=$mask2")

    total_itr = div(Ls, 32)
    f(ist::Int, ifn::Int, posa::Int, posb::Int, m1::Int, m2::Int, m3::Int, mat::AbstractMatrix, p::AbstractVector, n::Int) = begin
        @inbounds for i in ist:ifn
            l = (32 * i & m3) | (16 * i & m2) | (8 * i & m1) + 1 +i>>(n-5)<<n
            # l = div(l, 2) + 1
            l1 = l + posa
            l2 = l + posb
            l3 = l2 + posa
            # println("l0=$l, l1=$l1, l2=$l2, l3=$l3")
            vi = SMatrix{8, 4}(p[l], p[l+1], p[l+2], p[l+3], p[l+4], p[l+5], p[l+6], p[l+7],
            p[l1], p[l1+1], p[l1+2], p[l1+3], p[l1+4], p[l1+5], p[l1+6], p[l1+7],
            p[l2], p[l2+1], p[l2+2], p[l2+3], p[l2+4], p[l2+5], p[l2+6], p[l2+7],
            p[l3], p[l3+1], p[l3+2], p[l3+3], p[l3+4], p[l3+5], p[l3+6], p[l3+7])

            @fastmath p[l], p[l+1], p[l+2], p[l+3], p[l+4], p[l+5], p[l+6], p[l+7],
            p[l1], p[l1+1], p[l1+2], p[l1+3], p[l1+4], p[l1+5], p[l1+6], p[l1+7],
            p[l2], p[l2+1], p[l2+2], p[l2+3], p[l2+4], p[l2+5], p[l2+6], p[l2+7],
            p[l3], p[l3+1], p[l3+2], p[l3+3], p[l3+4], p[l3+5], p[l3+6], p[l3+7] = vi * mat
        end
    end
    parallel_run(total_itr, Threads.nthreads(), f, pos1, pos2, mask0, mask1, mask2, U, v, n)
end

function _apply_twobody_gate_LH!(key::Tuple{Int, Int}, U::AbstractMatrix, v::AbstractVector, n::Int)
    L = 1<<n
    Ls = length(v)
    q1, q2 = key
    sizej = 1 << (q2-1)
    mask0 = sizej - 1
    mask1 = xor(L - 1, 2 * sizej - 1)
    total_itr = div(Ls, 32)
    f1H(ist::Int, ifn::Int, posb::Int, m1::Int, m2::Int, mat::AbstractMatrix, p::AbstractVector, n::Int) = begin
        @inbounds for i in ist:ifn
            l = (32 * i & m2) | (16 * i & m1) + i>>(n-5)<<n
            l1 = l + posb
            # println("l=$l, l1=$l1")
            vi = SMatrix{8, 4}(p[l+1], p[l+3], p[l+5], p[l+7], p[l+9], p[l+11], p[l+13], p[l+15],
            p[l+2], p[l+4], p[l+6], p[l+8], p[l+10], p[l+12], p[l+14], p[l+16],
            p[l1+1], p[l1+3], p[l1+5], p[l1+7], p[l1+9], p[l1+11], p[l1+13], p[l1+15],
            p[l1+2], p[l1+4], p[l1+6], p[l1+8], p[l1+10], p[l1+12], p[l1+14], p[l1+16])

            @fastmath p[l+1], p[l+3], p[l+5], p[l+7], p[l+9], p[l+11], p[l+13], p[l+15],
            p[l+2], p[l+4], p[l+6], p[l+8], p[l+10], p[l+12], p[l+14], p[l+16],
            p[l1+1], p[l1+3], p[l1+5], p[l1+7], p[l1+9], p[l1+11], p[l1+13], p[l1+15],
            p[l1+2], p[l1+4], p[l1+6], p[l1+8], p[l1+10], p[l1+12], p[l1+14], p[l1+16] = vi * mat
        end
    end

    f2H(ist::Int, ifn::Int, posb::Int, m1::Int, m2::Int, mat::AbstractMatrix, p::AbstractVector, n::Int) = begin
        @inbounds for i in ist:ifn
            l = (32 * i & m2) | (16 * i & m1)+i>>(n-5)<<n
            l1 = l + posb
            # println("l=$l, l1=$l1")
            vi = SMatrix{8, 4}(p[l+1], p[l+2], p[l+5], p[l+6], p[l+9], p[l+10], p[l+13], p[l+14],
            p[l+3], p[l+4], p[l+7], p[l+8], p[l+11], p[l+12], p[l+15], p[l+16],
            p[l1+1], p[l1+2], p[l1+5], p[l1+6], p[l1+9], p[l1+10], p[l1+13], p[l1+14],
            p[l1+3], p[l1+4], p[l1+7], p[l1+8], p[l1+11], p[l1+12], p[l1+15], p[l1+16])

            @fastmath p[l+1], p[l+2], p[l+5], p[l+6], p[l+9], p[l+10], p[l+13], p[l+14],
            p[l+3], p[l+4], p[l+7], p[l+8], p[l+11], p[l+12], p[l+15], p[l+16],
            p[l1+1], p[l1+2], p[l1+5], p[l1+6], p[l1+9], p[l1+10], p[l1+13], p[l1+14],
            p[l1+3], p[l1+4], p[l1+7], p[l1+8], p[l1+11], p[l1+12], p[l1+15], p[l1+16] = vi * mat
        end
    end

    f3H(ist::Int, ifn::Int, posb::Int, m1::Int, m2::Int, mat::AbstractMatrix, p::AbstractVector, n::Int) = begin
        @inbounds for i in ist:ifn
            l = (32 * i & m2) | (16 * i & m1)+i>>(n-5)<<n
            l1 = l + posb
            # println("l=$l, l1=$l1")
            vi = SMatrix{8, 4}(p[l+1], p[l+2], p[l+3], p[l+4], p[l+9], p[l+10], p[l+11], p[l+12],
            p[l+5], p[l+6], p[l+7], p[l+8], p[l+13], p[l+14], p[l+15], p[l+16],
            p[l1+1], p[l1+2], p[l1+3], p[l1+4], p[l1+9], p[l1+10], p[l1+11], p[l1+12],
            p[l1+5], p[l1+6], p[l1+7], p[l1+8], p[l1+13], p[l1+14], p[l1+15], p[l1+16])

            @fastmath p[l+1], p[l+2], p[l+3], p[l+4], p[l+9], p[l+10], p[l+11], p[l+12],
            p[l+5], p[l+6], p[l+7], p[l+8], p[l+13], p[l+14], p[l+15], p[l+16],
            p[l1+1], p[l1+2], p[l1+3], p[l1+4], p[l1+9], p[l1+10], p[l1+11], p[l1+12],
            p[l1+5], p[l1+6], p[l1+7], p[l1+8], p[l1+13], p[l1+14], p[l1+15], p[l1+16] = vi * mat
        end
    end

    if q1 == 1
        f = f1H
    elseif q1 == 2
        f = f2H
    elseif q1 == 3
        f = f3H
    else
        error("qubit position $q1 not allowed for LH.")
    end
    parallel_run(total_itr, Threads.nthreads(), f, sizej, mask0, mask1, U, v, n)
end





"""
    applys when both keys <= 4
"""

function _apply_twobody_gate_LL!(key::Tuple{Int, Int}, U::AbstractMatrix, v::AbstractVector, n::Int)
    L = length(v)
    q1, q2 = key
    f12(ist::Int, ifn::Int, mat::AbstractMatrix, p::AbstractVector) = begin
        @inbounds for i in ist:ifn
            l = 32 * i + 1
            vi = SMatrix{4, 8}(p[l], p[l+1], p[l+2], p[l+3], p[l+4], p[l+5], p[l+6], p[l+7],
            p[l+8], p[l+9], p[l+10], p[l+11], p[l+12], p[l+13], p[l+14], p[l+15],
            p[l+16], p[l+17], p[l+18], p[l+19], p[l+20], p[l+21], p[l+22], p[l+23],
            p[l+24], p[l+25], p[l+26], p[l+27], p[l+28], p[l+29], p[l+30], p[l+31])
            @fastmath p[l:(l+31)]= mat * vi
        end
    end
    f13(ist::Int, ifn::Int, mat::AbstractMatrix, p::AbstractVector) = begin
        @inbounds for i in ist:ifn
            l = 32 * i + 1
            vi = SMatrix{4, 8}(p[l], p[l+1], p[l+4], p[l+5], p[l+2], p[l+3], p[l+6], p[l+7],
            p[l+8], p[l+9], p[l+12], p[l+13], p[l+10], p[l+11], p[l+14], p[l+15],
            p[l+16], p[l+17], p[l+20], p[l+21], p[l+18], p[l+19], p[l+22], p[l+23],
            p[l+24], p[l+25], p[l+28], p[l+29], p[l+26], p[l+27], p[l+30], p[l+31])

            @fastmath p[l], p[l+1], p[l+4], p[l+5], p[l+2], p[l+3], p[l+6], p[l+7],
            p[l+8], p[l+9], p[l+12], p[l+13], p[l+10], p[l+11], p[l+14], p[l+15],
            p[l+16], p[l+17], p[l+20], p[l+21], p[l+18], p[l+19], p[l+22], p[l+23],
            p[l+24], p[l+25], p[l+28], p[l+29], p[l+26], p[l+27], p[l+30], p[l+31] = mat * vi
        end
    end
    f14(ist::Int, ifn::Int, mat::AbstractMatrix, p::AbstractVector) = begin
        @inbounds for i in ist:ifn
            l = 32 * i
            vi = SMatrix{4, 8}(p[l+1], p[l+2], p[l+9], p[l+10], p[l+3], p[l+4], p[l+11], p[l+12],
            p[l+5], p[l+6], p[l+13], p[l+14], p[l+7], p[l+8], p[l+15], p[l+16],
            p[l+17], p[l+18], p[l+25], p[l+26], p[l+19], p[l+20], p[l+27], p[l+28],
            p[l+21], p[l+22], p[l+29], p[l+30], p[l+23], p[l+24], p[l+31], p[l+32])

            @fastmath p[l+1], p[l+2], p[l+9], p[l+10], p[l+3], p[l+4], p[l+11], p[l+12],
            p[l+5], p[l+6], p[l+13], p[l+14], p[l+7], p[l+8], p[l+15], p[l+16],
            p[l+17], p[l+18], p[l+25], p[l+26], p[l+19], p[l+20], p[l+27], p[l+28],
            p[l+21], p[l+22], p[l+29], p[l+30], p[l+23], p[l+24], p[l+31], p[l+32] = mat * vi
        end
    end
    f23(ist::Int, ifn::Int, mat::AbstractMatrix, p::AbstractVector) = begin
        @inbounds for i in ist:ifn
            l = 32 * i + 1
            vi = SMatrix{4, 8}(p[l], p[l+2], p[l+4], p[l+6], p[l+1], p[l+3], p[l+5], p[l+7],
            p[l+8], p[l+10], p[l+12], p[l+14], p[l+9], p[l+11], p[l+13], p[l+15],
            p[l+16], p[l+18], p[l+20], p[l+22], p[l+17], p[l+19], p[l+21], p[l+23],
            p[l+24], p[l+26], p[l+28], p[l+30], p[l+25], p[l+27], p[l+29], p[l+31])

            @fastmath p[l], p[l+2], p[l+4], p[l+6], p[l+1], p[l+3], p[l+5], p[l+7],
            p[l+8], p[l+10], p[l+12], p[l+14], p[l+9], p[l+11], p[l+13], p[l+15],
            p[l+16], p[l+18], p[l+20], p[l+22], p[l+17], p[l+19], p[l+21], p[l+23],
            p[l+24], p[l+26], p[l+28], p[l+30], p[l+25], p[l+27], p[l+29], p[l+31] = mat * vi
        end
    end
    f24(ist::Int, ifn::Int, mat::AbstractMatrix, p::AbstractVector) = begin
        @inbounds for i in ist:ifn
            l = 32 * i
            vi = SMatrix{4, 8}(p[l+1], p[l+3], p[l+9], p[l+11], p[l+2], p[l+4], p[l+10], p[l+12],
            p[l+5], p[l+7], p[l+13], p[l+15], p[l+6], p[l+8], p[l+14], p[l+16],
            p[l+17], p[l+19], p[l+25], p[l+27], p[l+18], p[l+20], p[l+26], p[l+28],
            p[l+21], p[l+23], p[l+29], p[l+31], p[l+22], p[l+24], p[l+30], p[l+32])

            @fastmath p[l+1], p[l+3], p[l+9], p[l+11], p[l+2], p[l+4], p[l+10], p[l+12],
            p[l+5], p[l+7], p[l+13], p[l+15], p[l+6], p[l+8], p[l+14], p[l+16],
            p[l+17], p[l+19], p[l+25], p[l+27], p[l+18], p[l+20], p[l+26], p[l+28],
            p[l+21], p[l+23], p[l+29], p[l+31], p[l+22], p[l+24], p[l+30], p[l+32] = mat * vi
        end
    end
    f34(ist::Int, ifn::Int, mat::AbstractMatrix, p::AbstractVector) = begin
        @inbounds for i in ist:ifn
            l = 32 * i
            vi = SMatrix{4, 8}(p[l+1], p[l+5], p[l+9], p[l+13], p[l+2], p[l+6], p[l+10], p[l+14],
            p[l+3], p[l+7], p[l+11], p[l+15], p[l+4], p[l+8], p[l+12], p[l+16],
            p[l+17], p[l+21], p[l+25], p[l+29], p[l+18], p[l+22], p[l+26], p[l+30],
            p[l+19], p[l+23], p[l+27], p[l+31], p[l+20], p[l+24], p[l+28], p[l+32])

            @fastmath p[l+1], p[l+5], p[l+9], p[l+13], p[l+2], p[l+6], p[l+10], p[l+14],
            p[l+3], p[l+7], p[l+11], p[l+15], p[l+4], p[l+8], p[l+12], p[l+16],
            p[l+17], p[l+21], p[l+25], p[l+29], p[l+18], p[l+22], p[l+26], p[l+30],
            p[l+19], p[l+23], p[l+27], p[l+31], p[l+20], p[l+24], p[l+28], p[l+32] = mat * vi
        end
    end
    if q1==1 && q2 == 2
        f = f12
    elseif q1==1 && q2 == 3
        f = f13
    elseif q1==1 && q2 == 4
        f = f14
    elseif q1==2 && q2 == 3
        f = f23
    elseif q1==2 && q2 == 4
        f = f24
    elseif q1==3 && q2 == 4
        f = f34
    else
        error("qubit position $q1 and $q2 not allowed for LL.")
    end
    total_itr = div(L, 32)
    parallel_run(total_itr, Threads.nthreads(), f, U, v)
end

function _apply_gate_threaded2!(key::Tuple{Int, Int}, U::AbstractMatrix, v::AbstractVector,n::Int)
    q0, q1 = key
    if q0 > 3
        return _apply_twobody_gate_HH!(key, SMatrix{4,4, eltype(v)}(transpose(U)), v, n)
    elseif q1 > 4
        return _apply_twobody_gate_LH!(key, SMatrix{4,4, eltype(v)}(transpose(U)), v, n)
    else
        return _apply_twobody_gate_LL!(key, SMatrix{4,4, eltype(v)}(U), v, n)
    end
end