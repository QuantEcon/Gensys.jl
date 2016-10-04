"""
```
GensysOutput
```

The returned system from `gensysdt` or `gensysct` takes the following form:
```
y(t) = G1*y(t-1) + C + impact*z(t) + ywt*inv(I-fmat*inv(L))*fwt*z(t+1)
```

The `GensysOutput` type contains the following fields:

- `G1`: transition matrix (coefficient matrix on lagged `y`)
- `C`:  Contstant vector
- `impact`: Shock coefficient matrix
- `fmat`: 
- `fwt`:
- `ywt`:
- `gev`:
- `eu`: Existence and uniqueness return codes
   - `eu[1]==1` for existence
   - `eu[2]==1` for uniqueness
   - `eu[1]==-1` for existence only with not-s.c. z;
   - `eu==[-2,-2]` for coincident zeros.
```
"""

type GensysOutput{T}
    G1::Matrix{T}
    C::Array{T}
    impact::Matrix{T}
    fmat::Array{Complex{T}}
    fwt::Array{Complex{T}}
    ywt::Array{Complex{T}}
    gev::Array{Complex{T}}
    eu::Vector{Int}
end
