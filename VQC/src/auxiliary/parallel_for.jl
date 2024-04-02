using Base.Threads

const MIN_SIZE = 1024


get_size_1(total_itr::Int, n_threads::Int, thread_id::Int) = div(total_itr * thread_id, n_threads)
get_size_2(total_itr::Int, n_threads::Int, thread_id::Int) = div(total_itr * (thread_id + 1), n_threads)

"""
    The inner function f is assumed to be Zero Based
"""
function parallel_run(total_itr::Int, n_threads::Int, f::Function, args...)
    if (total_itr >= MIN_SIZE) && (n_threads > 1)
        # nave, resi, jobs = partition_jobs(total_itr, n_threads)
        Threads.@threads for thread_id in 0:(n_threads-1)
            ist = get_size_1(total_itr, n_threads, thread_id)
            ifn = get_size_2(total_itr, n_threads, thread_id) - 1
            f(ist, ifn, args...)
        end
    else
        f(0, total_itr-1, args...)
    end
end

"""
    The inner function f is assumed to be Zero Based
"""
function parallel_sum(::Type{T}, total_itr::Int, n_threads::Int, f::Function, args...) where {T<:Real}
    if (total_itr >= MIN_SIZE) && (n_threads > 1)
        # nave, resi, jobs = partition_jobs(total_itr, n_threads)
        r = Atomic{T}(0)
        Threads.@threads for thread_id in 0:(n_threads-1)
            ist = get_size_1(total_itr, n_threads, thread_id)
            ifn = get_size_2(total_itr, n_threads, thread_id) - 1
            atomic_add!(r, f(ist, ifn, args...))
        end
        return r[]
    else
        return f(0, total_itr-1, args...)
    end    
end


function parallel_sum(::Type{T}, total_itr::Int, n_threads::Int, f::Function, args...) where {T<:Complex}
    if (total_itr >= MIN_SIZE) && (n_threads > 1)
        # nave, resi, jobs = partition_jobs(total_itr, n_threads)
        RT = real(T)
        rr = Atomic{RT}(0)
        ri = Atomic{RT}(0)
        Threads.@threads for thread_id in 0:(n_threads-1)
            ist = get_size_1(total_itr, n_threads, thread_id)
            ifn = get_size_2(total_itr, n_threads, thread_id) - 1
            tmp = f(ist, ifn, args...)
            atomic_add!(rr, real(tmp))
            atomic_add!(ri, imag(tmp))
        end
        return Complex(rr[], ri[])
    else
        return f(0, total_itr-1, args...)
    end    
end

#有点特殊,由于要得到每一个态各自的测量结果，因此每个线程处理地是同一个态，所以只能按照态的size来分到不同的线程。
function parallel_sum(::Type{T}, res::Int, total_itr::Int, n_threads::Int, f::Function, args...) where {T<:Complex}
    RT = real(T)
    ri = zeros(div(total_itr,res))
    rr = zeros(div(total_itr,res))
    Threads.@threads for j in 0:res:(total_itr-1)
        tmp = f(j, j+res-1, args...)
        rr[div(j,res)+1]=real(tmp)
        ri[div(j,res)+1]=imag(tmp)
    end
    return Complex.(rr,ri)  
end