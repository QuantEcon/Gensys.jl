# This code is based on a routine originally copyright Chris Sims.
# See http://sims.princeton.edu/yftp/gensys/

"""
```
gensysct(Γ0, Γ1, c, Ψ, Π)
gensysct(Γ0, Γ1, c, Ψ, Π, div)
gensysct(F::Base.LinAlg.GeneralizedSchur, c, Ψ, Π)
gensysct(F::Base.LinAlg.GeneralizedSchur, c, Ψ, Π, div)
```

Generate state-space solution to canonical-form DSGE model.

System given as
```
Γ0*Dy(t) = Γ1*y(t) + c + Ψ*z(t) + Π*η(t),
```
with z an exogenous variable process and η being endogenously white noise expectational errors.

Returned system is
```
Dy(t) = G1*y(t) + C + impact * z(t)
```


Returned values are
```
G1, C, impact, qt', a, b, z, eu
```

Also returned is the qz decomposition, qt'az' = Γ0, qt'bz' = Γ1, with a and b
upper triangular and the system ordered so that all zeros on the diagonal of b are in
the lower right corner, all cases where the real part of bii/aii is greater than or 
equal to div appear in the next block above the zeros, and the remaining bii/aii's 
all have bii/aii<div .  These elements can be used to construct the full backward and 
forward solution.  See the paper \"Solving Linear Rational Expectations Models\", 
http://eco-072399b.princeton.edu/yftp/gensys .  Note that if one simply wants the backwar
and forward projection of y on eΨlon, ignoring existence and uniqueness questions, the
projection can be computed by Fourier methods.

If `div` is omitted from argument list, a `div`>1 is calculated.

### Return codes

* `eu[1]==1` for existence
* `eu[2]==1` for uniqueness
* `eu[1]==-1` for existence for white noise η
* `eu==[-2,-2]` for coincident zeros.
"""



const ϵ = sqrt(eps()) * 10

function gensysct(Γ0, Γ1, c, Ψ, Π, args...)
    F = schurfact!(complex(Γ0), complex(Γ1))
    gensysct(F, c, Ψ, Π, args...)
end

function gensysct(F::Base.LinAlg.GeneralizedSchur, c, Ψ, Π)
    gensysct(F, c, Ψ, Π, new_divct(F))
end

function gensysct(F::Base.LinAlg.GeneralizedSchur, c, Ψ, Π, div)
    eu = [0, 0]
    a, b = F[:S], F[:T]
    n = size(a, 1)
    
    for i in 1:n
        if (abs(a[i, i]) < ϵ) && (abs(b[i, i]) < ϵ)
            info("Coincident zeros.  Indeterminacy and/or nonexistence.")
            eu = [-2, -2]
            G1 = Array{Float64, 2}() ;  C = Array{Float64, 1}() ; impact = Array{Float64, 2}()
            a, b, qt, z = FS[:S], FS[:T], FS[:Q], FS[:Z]
            return G1, C, impact, qt', a, b, z, eu
        end
    end
    movelast = Bool[(real(b[i, i] / a[i, i]) > div) || (abs(a[i, i]) < ϵ) for i in 1:n]
    nunstab = sum(movelast)
    FS = ordschur!(F, !movelast)
    a, b, qt, z = FS[:S], FS[:T], FS[:Q], FS[:Z]

    qt1 = qt[:, 1:(n - nunstab)]
    qt2 = qt[:, (n - nunstab + 1):n]
    a2 = a[(n - nunstab + 1):n, (n - nunstab + 1):n]
    b2 = b[(n - nunstab + 1):n, (n - nunstab + 1):n]
    etawt = Ac_mul_B(qt2, Π)
    bigev, ueta, deta, veta = decomposition_svdct!(etawt)
    zwt = Ac_mul_B(qt2, Ψ)
    bigev, uz, dz, vz = decomposition_svdct!(zwt)
    if isempty(bigev)
        exist = true
    else
        exist = vecnorm(uz- A_mul_Bc(ueta, ueta) * uz, 2) < ϵ * n
    end
    if isempty(bigev)
        existx = true
    else
        zwtx0 = b2 \ zwt
        zwtx = zwtx0
        M = b2 \ a2
        M = scale!(M, 1 / norm(M))
        for i in 2:nunstab
            zwtx = hcat(M * zwtx, zwtx0)
        end
        zwtx = b2 * zwtx
        bigev, ux, dx, vx = decomposition_svdct!(zwtx)
        existx = vecnorm(ux - A_mul_Bc(ueta, ueta) * ux, 2) < ϵ * n
    end
    etawt1 = Ac_mul_B(qt1, Π)
    bigev, ueta1, deta1, veta1 = decomposition_svdct!(etawt1)
    if existx | (nunstab == 0)
       eu[1] = 1
    else
        if exist
            eu[1] = -1
        end
    end
    if isempty(veta1)
        unique = true
    else
        unique = vecnorm(veta1- A_mul_Bc(veta, veta) * veta1, 2) < ϵ * n
    end
    if unique
       eu[2] = 1
    end

    tmat = hcat(eye(n - nunstab), -ueta1 * deta1 * Ac_mul_B(veta1, veta) * (deta \ ueta'))
    G0 =  vcat(tmat * a, hcat(zeros(nunstab, n - nunstab), eye(nunstab)))
    G1 =  vcat(tmat * b, zeros(nunstab, n))
    G1 = G0 \ G1
    usix = (n - nunstab + 1):n
    C = G0 \ vcat(tmat * Ac_mul_B(qt, c), (a[usix, usix] .- b[usix, usix]) \ Ac_mul_B(qt2, c))
    impact = G0 \ vcat(tmat * Ac_mul_B(qt, Ψ), zeros(nunstab, size(Ψ, 2)))
    G1 = z * A_mul_Bc(G1, z)
    G1 = real(G1)
    C = real(z * C)
    impact = real(z * impact)
    return G1, C, impact, qt', a, b, z, eu
end



function new_divct(F::Base.LinAlg.GeneralizedSchur)
    a, b = F[:S], F[:T]
    n = size(a, 1)
    div = 0.001
    for i in 1:n
        if abs(a[i, i]) > ϵ
            divhat = real(b[i, i] / a[i, i])
            if (ϵ < divhat) && (divhat < div)
                div = 0.5 * divhat
            end
        end
    end
    return div
end


function decomposition_svdct!(A)
    Asvd = svdfact!(A)
    bigev = find(Asvd[:S] .> ϵ)
    Au = Asvd[:U][:, bigev]
    Ad = diagm(Asvd[:S][bigev])
    Av = Asvd[:V][:, bigev]
    return bigev, Au, Ad, Av
end