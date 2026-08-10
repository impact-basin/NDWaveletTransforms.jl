# @setup_workload begin
#
#     compile_dwt(x, w, l) = begin
#         y = nsdwt(x, w, l)
#         x = nsidwt(y, w, l)
#         y = nswpt(x, w, l)
#         x = nsiwpt(y, w, l)
#         return (x, y)
#     end
#
#     compile_dwt(w, l) = x -> compile_dwt(x, w, l)
#
#     input(T, d) = rand(T, Tuple(32 for _ in 1:d))
#
#     wavelets = [
#         WT_D1, WT_D2, WT_D3, WT_D4,
#         WT_D5, WT_D6, WT_D7, WT_D8,
#         WT_COIF2, WT_COIF4,
#         WT_COIF6, WT_COIF8,
#         WT_HAAR,
#     ]
#
#     types = [Float16, Float32, Float64]
#
#     @compile_workload begin
#         for w in wavelets, T in types
#             input(T, 1) |> compile_dwt(w, 3)
#             input(T, 2) |> compile_dwt(w, 3)
#             input(T, 3) |> compile_dwt(w, 3)
#             input(T, 2) |> compile_dwt(w, (3, 4))
#             input(T, 3) |> compile_dwt(w, (3, 4, 4))
#         end
#     end
# end
