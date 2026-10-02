# Bases

Every basis is orthogonal and represented by a `WTOrthogonalBasis`, a pair
of equal-length `SVector`s: a scaling filter φ and a wavelet filter ψ.

| Kind | Symbols |
|---|---|
| Haar | `WT_HAAR` |
| Daubechies, `n` vanishing moments | `WT_D1` ... `WT_D20` |
| Coiflets | `WT_COIF2` ... `WT_COIF10` |
| Symlets | `WT_SYM2` ... `WT_SYM10` |
| Battle-Lemarie | `WT_BATTLE2`, `WT_BATTLE4`, `WT_BATTLE6` |
| Beyl | `WT_BEYL` |
| Vaidyanathan | `WT_VAID` |
| Morris minimum-bandwidth | `WT_MB42`, `WT_MB82`, `WT_MB83`, `WT_MB84`, `WT_MB103`, `WT_MB123`, `WT_MB143`, `WT_MB163`, `WT_MB183`, `WT_MB243`, `WT_MB323` |
| Fejer-Korovkin | `WT_FK4`, `WT_FK6`, `WT_FK8`, `WT_FK14`, `WT_FK18`, `WT_FK22` |
| Best-localised Daubechies | `WT_BL7`, `WT_BL9`, `WT_BL10` |
| Han | `WT_HAN23`, `WT_HAN33`, `WT_HAN45`, `WT_HAN55` |

## Your own taps

Build a basis from a scaling filter, a wavelet filter, or either alone. The
constructor normalises the taps and derives the missing filter as the
orthogonal complement of the one you give.

```julia
using StaticArrays

my_haar = WTOrthogonalBasis(φ = SA[1.0, 1.0])
my_d4 = WTOrthogonalBasis(φ = SA[0.2303778, 0.7148466, 0.6308808, -0.0279838,
                              -0.1870348, 0.0308414, 0.0328830, -0.0105974])
```

The filters are `SVector`s, so their length is part of the type. A basis
with different-length φ and ψ cannot be constructed.

## Choosing a basis

Shorter filters localise better in space, longer filters in frequency. Haar
is the extreme: exact on piecewise-constant signals, and visibly blocky on
anything smooth. D4 and D8 are the usual working choices when the signal is
smooth but not polynomial. `complement` returns the filter that pairs with
another.
