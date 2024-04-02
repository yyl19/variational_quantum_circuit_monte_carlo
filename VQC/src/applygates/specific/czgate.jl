

"""
    specialized for CZ gate
"""
function apply_threaded!(gt::CZGate, v::AbstractVector)
    (length(v) < 32) && return apply_serial!(gt, v)
    L = length(v)
    q1, q2 = ordered_positions(gt)
    pos1, pos2 = 1 << (q1-1), 1 << (q2-1)

    mask0 = pos1 - 1
    mask1 = xor(pos2 - 1, 2 * pos1 - 1)
    mask2 = xor(L - 1, 2 * pos2 - 1)

    f(ist::Int, ifn::Int, posab1::Int, m1::Int, m2::Int, m3::Int, p::AbstractVector) = begin
        @inbounds for i in ist:ifn
            l = (4 * i & m3) | (2 * i & m2) | (i & m1) + posab1
            p[l] = -p[l]
        end
    end

    total_itr = div(L, 4)
    parallel_run(total_itr, Threads.nthreads(), f, pos1+pos2+1, mask0, mask1, mask2, v)
end

"""
    specialized for CNOT gate
"""
function apply_threaded!(gt::CNOTGate, v::AbstractVector)
    (length(v) < 32) && return apply_serial!(gt, v)
    L = length(v)
    q1, q2 = ordered_positions(gt)
    pos1, pos2 = 1 << (q1-1), 1 << (q2-1)

    mask0 = pos1 - 1
    mask1 = xor(pos2 - 1, 2 * pos1 - 1)
    mask2 = xor(L - 1, 2 * pos2 - 1)

    f_f(ist::Int, ifn::Int, posa::Int, posb::Int, m1::Int, m2::Int, m3::Int, p::AbstractVector) = begin
        @inbounds for i in ist:ifn
            l00 = (4 * i & m3) | (2 * i & m2) | (i & m1) + 1
            l10 = l00 + posa
            l01 = l00 + posb
            l11 = l01 + posa
            p[l10], p[l11] = p[l11], p[l10]
        end
    end

    f_e(ist::Int, ifn::Int, posa::Int, posb::Int, m1::Int, m2::Int, m3::Int, p::AbstractVector) = begin
        @inbounds for i in ist:ifn
            l00 = (4 * i & m3) | (2 * i & m2) | (i & m1) + 1
            l10 = l00 + posa
            l01 = l00 + posb
            l11 = l01 + posa
            p[l01], p[l11] = p[l11], p[l01]
        end
    end

    if positions(gt)[1] == q1
        f = f_f
    else
        f = f_e
    end

    total_itr = div(L, 4)
    parallel_run(total_itr, Threads.nthreads(), f, pos1, pos2, mask0, mask1, mask2, v)
end

##StateVectorBatch
function apply_threaded!(gt::CNOTGate, v::AbstractVector, n::Int)
    (length(v) < 32) && return apply_serial!(gt, v)
    Ls = length(v)
    L = 1<<n
    q1, q2 = ordered_positions(gt)
    pos1, pos2 = 1 << (q1-1), 1 << (q2-1)

    mask0 = pos1 - 1
    mask1 = xor(pos2 - 1, 2 * pos1 - 1)
    mask2 = xor(L - 1, 2 * pos2 - 1)

    f_f(ist::Int, ifn::Int, posa::Int, posb::Int, m1::Int, m2::Int, m3::Int, p::AbstractVector, n::Int) = begin
        @inbounds for i in ist:ifn
            l00 = (4 * i & m3) | (2 * i & m2) | (i & m1) + 1 + i>>(n-2)<<n
            l10 = l00 + posa
            l01 = l00 + posb
            l11 = l01 + posa
            p[l10], p[l11] = p[l11], p[l10]
        end
    end

    f_e(ist::Int, ifn::Int, posa::Int, posb::Int, m1::Int, m2::Int, m3::Int, p::AbstractVector, n::Int) = begin
        @inbounds for i in ist:ifn
            l00 = (4 * i & m3) | (2 * i & m2) | (i & m1) + 1+ i>>(n-2)<<n
            l10 = l00 + posa
            l01 = l00 + posb
            l11 = l01 + posa
            p[l01], p[l11] = p[l11], p[l01]
        end
    end

    if positions(gt)[1] == q1
        f = f_f
    else
        f = f_e
    end

    total_itr = div(Ls, 4)
    parallel_run(total_itr, Threads.nthreads(), f, pos1, pos2, mask0, mask1, mask2, v, n)
end
##StateVectorBatch


"""
    specialized for CPHASE gate
"""
function apply_threaded!(gt::CPHASEGate, v::AbstractVector)
    (length(v) < 32) && return apply_serial!(gt, v)

    L = length(v)
    q1, q2 = ordered_positions(gt)
    pos1, pos2 = 1 << (q1-1), 1 << (q2-1)

    mask0 = pos1 - 1
    mask1 = xor(pos2 - 1, 2 * pos1 - 1)
    mask2 = xor(L - 1, 2 * pos2 - 1)

    f(ist::Int, ifn::Int, posab1::Int, m1::Int, m2::Int, m3::Int, alpha::Number, p::AbstractVector) = begin
        @inbounds for i in ist:ifn
            l = (4 * i & m3) | (2 * i & m2) | (i & m1) + posab1
            @fastmath p[l] *= alpha
        end
    end

    exp_phi = convert(eltype(v), exp(im * parameters(gt)[1] ))

    total_itr = div(L, 4)
    parallel_run(total_itr, Threads.nthreads(), f, pos1+pos2+1, mask0, mask1, mask2, exp_phi, v)
end

function apply_threaded!(gt::ERyGate, v::AbstractVector, n::Int)
    ba = gt.batch[1]
    @assert ba==length(gt.paras)
    L=1<<n
    key=ordered_positions(gt)[1]
    pos = 1 << (key- 1)
    strid=2*pos
    f(ist::Int, ifn::Int, pos::Int, v::AbstractVector,u::Float64) = begin
        u11=cos(u/2)
        u12=-sin(u/2)
        for i in ist:strid:(ifn-1)
            @inbounds for j in 0:(pos-1)
                l=i+j+1
                vi1 = v[l]
                vi2 = v[l + pos]
                vo1 = u11 * vi1 + u12 * vi2
                vo2 = -u12 * vi1 + u11 * vi2
                v[l] = vo1
                v[l + pos] = vo2
            end
        end
    end
    Threads.@threads for i in 1:ba
        f((i-1)*L, i*L, pos, v, gt.paras[i])
    end
end