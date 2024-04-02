# one, two, three-qubit gate operations


#Cuda (GPU) for one qubit gate
function _cuapply_onebody_gate!(key::Int, U::AbstractMatrix, v::CuArray, n::Int)
    Ls = length(v)
    L = 1<<n
    u = CuArray(U)
    pos = 2^(key-1)
    m1= pos-1
    m2 = xor(L - 1, 2 * pos - 1)
    @inline function f1(u, v, pos, m1, m2, total_itr,n)
        index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        stride = gridDim().x * blockDim().x
        @inbounds for j = index:stride:total_itr
            i=j-1
            l = (2 * i & m2) | (i & m1) + 1 + i>>(n-1)<<n
            vi1 = v[l]
            vi2 = v[l + pos]
            vo1 = u[1,1] * vi1 + u[1,2] * vi2
            vo2 = u[2,1] * vi1 + u[2,2] * vi2
            v[l] = vo1
            v[l + pos] = vo2
        end
        return nothing
    end
    total_itr=div(Ls,2)
    kernel = @cuda launch=false f1(u, v, pos, m1, m2, total_itr, n)
    config = launch_configuration(kernel.fun)
    threads = min(total_itr, config.threads)
    blocks = cld(total_itr, threads)
    kernel(u, v, pos, m1, m2, total_itr, n; threads, blocks)
end

_apply_gate_threaded2!(q0::Int, U::AbstractMatrix, v::CuArray, n::Int)=return _cuapply_onebody_gate!(q0, SMatrix{2,2, eltype(v)}(U), v, n)

_apply_gate_threaded2!(q0::Tuple{Int}, U::AbstractMatrix, v::CuArray, n::Int)= _apply_gate_threaded2!(q0[1], U, v, n)
##cuda




###GPU for two qubit gate
_apply_gate_threaded2!(key::Tuple{Int, Int}, U::AbstractMatrix, v::CuArray, n::Int) = _apply_gate_2_impl!(key::Tuple{Int, Int}, U::AbstractMatrix, v::CuArray, n::Int)

function _apply_gate_2_impl!(key::Tuple{Int, Int}, U::AbstractMatrix, v::CuArray, n::Int)
    Ls = length(v)
    L = 1 << n
    q1, q2 = key
    pos1, pos2 = 2^(q1-1), 2^(q2-1)
    m1 = pos1 - 1
    m2 = xor(pos2 - 1, 2 * pos1 - 1)
    m3 = xor(L - 1, 2 * pos2 - 1)
    u=vec(CuArray(transpose(U)))
    @inline function f1(u,v,m1,m2,m3,pos1,pos2,total_itr, n)
        index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        stride = gridDim().x * blockDim().x
        @inbounds for j in index:stride:total_itr
            i= j-1
            l = (4 * i & m3) | (2 * i & m2) | (i & m1) + 1 + i>>(n-2)<<n
            vl = v[l]
            vl1 = v[l + pos1]
            vl2 = v[l + pos2]
            vl12 = v[l + pos1 + pos2]
            vl0 = vl*u[1] + vl1*u[2] + vl2*u[3]+vl12*u[4]
            vl10 = vl*u[5] + vl1*u[6] + vl2*u[7]+vl12*u[8]
            vl20 = vl*u[9] + vl1*u[10] + vl2*u[11]+vl12*u[12]
            vl120 = vl*u[13] + vl1*u[14] + vl2*u[15]+vl12*u[16]
            v[l], v[l + pos1], v[l + pos2], v[l + pos1 + pos2] = vl0 ,vl10,vl20,vl120
        end
        return nothing
    end
    total_itr=div(Ls,4)
    kernel = @cuda launch=false f1(u,v,m1,m2,m3,pos1,pos2,total_itr, n)
    config = launch_configuration(kernel.fun)
    threads = min(total_itr, config.threads)
    blocks = cld(total_itr, threads)
    kernel(u,v,m1,m2,m3,pos1,pos2,total_itr, n; threads, blocks)
end

#GPU for specific gate

function apply_threaded!(gt::XGate, v::CuArray, n::Int)
    Ls = length(v)
    L = 1<<n
    key=ordered_positions(gt)[1]
    pos = 2^(key-1)
    m1= pos-1
    m2 = xor(L - 1, 2 * pos - 1)
    @inline function f1(v, pos, m1, m2, total_itr, n)
        index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        stride = gridDim().x * blockDim().x
        for j = index:stride:total_itr
            i=j-1
            l = (2 * i & m2) | (i & m1) + 1 + i>>(n-1)<<n
            l1 = l + pos
            @inbounds v[l], v[l1] = v[l1], v[l]
        end
        return nothing
    end
    total_itr=div(Ls,2)
    kernel = @cuda launch=false f1( v, pos, m1, m2, total_itr, n)
    config = launch_configuration(kernel.fun)
    threads = min(total_itr, config.threads)
    blocks = cld(total_itr, threads)
    kernel(v, pos, m1, m2, total_itr, n; threads, blocks)
end

function apply_threaded!(gt::ZGate, v::CuArray, n::Int)
    Ls = length(v)
    L = 1<<n
    key=ordered_positions(gt)[1]
    pos = 2^(key-1)
    m1= pos-1
    m2 = xor(L - 1, 2 * pos - 1)
    @inline function f1(v, pos, m1, m2, total_itr, n)
        index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        stride = gridDim().x * blockDim().x
        @inbounds for j = index:stride:total_itr
            i=j-1
            l = (2 * i & m2) | (i & m1) + 1 + i>>(n-1)<<n
            l1 = l + pos
            @fastmath v[l1] = -v[l1]
        end
        return nothing
    end
    total_itr=div(Ls,2)
    kernel = @cuda launch=false f1(v, pos, m1, m2, total_itr, n)
    config = launch_configuration(kernel.fun)
    threads = min(total_itr, config.threads)
    blocks = cld(total_itr, threads)
    CUDA.@sync begin
        kernel(v, pos, m1, m2, total_itr, n; threads, blocks)
    end
end


function apply_threaded!(gt::RzGate, v::CuArray, n::Int)
    Ls = length(v)
    L=1<<n
    key=ordered_positions(gt)[1]
    pos = 1 << (key- 1)
    m1 = pos - 1
    m2 = xor(L - 1, 2 * pos - 1)
    θ = gt.paras[1]
    t1=exp(-im*θ/2)
    t2=exp(im*θ/2)
    @inline function f1(v, pos::Int, m1::Int, m2::Int, n::Int, total_itr::Int ,t1::ComplexF64,t2::ComplexF64)
        index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        stride = gridDim().x * blockDim().x
        @inbounds for j = index:stride:total_itr
            i = j-1
            l = (2 * i & m2) | (i & m1) + 1 + i>>(n-1)<<n
            l1 = l + pos
            @fastmath v[l], v[l1] = t1*v[l], t2*v[l1]
        end
    end
    total_itr = div(Ls, 2)
    kernel = @cuda launch=false f1(v, pos, m1, m2, n, total_itr, t1, t2)
    config = launch_configuration(kernel.fun)
    threads = min(total_itr, config.threads)
    blocks = cld(total_itr, threads)
    kernel(v, pos, m1, m2, n, total_itr, t1, t2; threads, blocks)
end


function apply_threaded!(gt::CNOTGate, v::CuArray, n::Int)
    (length(v) < 32) && return apply_serial!(gt, v)
    Ls = length(v)
    L = 1<<n
    q1, q2 = ordered_positions(gt)
    pos1, pos2 = 1 << (q1-1), 1 << (q2-1)
    m1 = pos1 - 1
    m2 = xor(pos2 - 1, 2 * pos1 - 1)
    m3 = xor(L - 1, 2 * pos2 - 1)

    @inline function f_f(v, pos1, pos2, m1, m2, m3, total_itr, n)
        index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        stride = gridDim().x * blockDim().x
        @inbounds for j = index:stride:total_itr
            i = j-1
            l00 = (4 * i & m3) | (2 * i & m2) | (i & m1) + 1 + i>>(n-2)<<n
            l10 = l00 + pos1
            l01 = l00 + pos2
            l11 = l01 + pos1
            v[l10], v[l11] = v[l11], v[l10]
        end
    end
    @inline function f_e(v, pos1, pos2, m1, m2, m3, total_itr, n)
        index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        stride = gridDim().x * blockDim().x
        @inbounds for j = index:stride:total_itr
            i = j-1
            l00 = (4 * i & m3) | (2 * i & m2) | (i & m1) + 1 + i>>(n-2)<<n
            l01 = l00 + pos2
            l11 = l01 + pos1
            v[l01], v[l11] = v[l11], v[l01]
        end
    end

    if positions(gt)[1] == q1
        f = f_f
    else
        f = f_e
    end

    total_itr = div(Ls, 4)
    kernel = @cuda launch=false f(v, pos1, pos2, m1, m2, m3, total_itr, n)
    config = launch_configuration(kernel.fun)
    threads = min(total_itr, config.threads)
    blocks = cld(total_itr, threads)
    kernel(v, pos1, pos2, m1, m2, m3, total_itr, n; threads, blocks)
end 

function apply_threaded!(gt::ERyGate, v::CuArray, n::Int)
    ba = gt.batch[1]
    @assert ba==length(gt.paras)
    Ls=length(v)
    L=1<<n
    key=ordered_positions(gt)[1]
    pos = 1 << (key- 1)
    m1 = pos - 1
    m2 = xor(L - 1, 2 * pos - 1)
    θs=CuArray(gt.paras)
    @inline function f(v, pos, m1, m2, total_itr, n, θs)
        index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        stride = gridDim().x * blockDim().x
        @inbounds for j = index:stride:total_itr
                i=j-1
                θ=θs[i>>(n-1)+1]
                u11=cos(θ/2)
                u12=-sin(θ/2)
                l = (2 * i & m2) | (i & m1) + 1 + i>>(n-1)<<n
                vi1 = v[l]
                vi2 = v[l + pos]
                vo1 = u11 * vi1 + u12 * vi2
                vo2 = -u12 * vi1 + u11 * vi2
                v[l] = vo1
                v[l + pos] = vo2
            end
        end
    total_itr = div(Ls, 2)
    kernel = @cuda launch=false f(v, pos, m1, m2, total_itr, n, θs)
    config = launch_configuration(kernel.fun)
    threads = min(total_itr, config.threads)
    blocks = cld(total_itr, threads)
    kernel(v, pos, m1, m2, total_itr, n, θs; threads, blocks)
end
