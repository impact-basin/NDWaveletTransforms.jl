using Waiver
using Reactant
Reactant.set_default_backend("cpu")

r = rand(32) |> Reactant.to_rarray
w = WT_HAAR  |> Reactant.to_rarray

haar1d = @compile nsdwt(r, WT_HAAR, 1)


