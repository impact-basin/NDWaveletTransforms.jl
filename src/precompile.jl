@setup_workload begin
    signals = [rand(T, k...) for T in [Float32, Float64] for k in [(8), (8, 8), (8, 8, 8)]]
    for signal in signals
        @compile_workload begin
            dwt!(signal, WT_HAAR, 2)
            idwt!(signal, WT_HAAR, 2)
            dwt!(signal, WT_HAAR, 2, convention = :wavelets)
            idwt!(signal, WT_HAAR, 2, convention = :wavelets)
            x = rtree_views(signal)[1]
            dwt!(x, WT_HAAR, 1)
            idwt!(x, WT_HAAR, 1)
            dwt!(x, WT_HAAR, 1, convention = :wavelets)
            idwt!(x, WT_HAAR, 1, convention = :wavelets)
        end
    end
end
