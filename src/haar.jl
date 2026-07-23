# hardcoded 2-D forward Haar transform for comparison
function haar!(img :: Matrix{Number},
              L=minimum(trailing_zeros.([size(img)...]));
	      tmp = similar(img))

    # set up bounds and working area.
    h,v = size(img)

    # we will proceed by iterated views.
    w = @view img[:,:]
    o = @view tmp[:,:]

    # check the bounds on the DWT level.
    m = minimum(trailing_zeros.([h, v])) 
    @assert L <= m "haar!(): Transform level specification too high."

    for n=1:L

    	# update horizontal and vertical bounds.
        h >>= 1
        v >>= 1

	# generate matrix subviews
	lls = @view o[1:h,1:v]
	lhs = @view o[1:h,v+1:end]
	hls = @view o[h+1:end,1:v]
	hhs = @view o[h+1:end,v+1:end]

        # iterate over the image, decimated by 2 in each dimension.
        for j=1:2:2h, k=1:2:2v
            
            # compute filter outputs.
            ll = ((w[j,k] + w[j+1,k]) + (w[j,k+1] + w[j+1,k+1])) / 4
            lh = ((w[j,k] - w[j+1,k]) + (w[j,k+1] - w[j+1,k+1])) / 4
            hl = ((w[j,k] + w[j+1,k]) - (w[j,k+1] + w[j+1,k+1])) / 4
            hh = ((w[j,k] - w[j+1,k]) - (w[j,k+1] - w[j+1,k+1])) / 4

	    # cache filter outputs
	    ji = j >> 1 + 1
	    ki = k >> 1 + 1
	    lls[ji,ki] = ll 
	    lhs[ji,ki] = lh 
	    hls[ji,ki] = hl 
	    hhs[ji,ki] = hh 
        end

        # write out, and zoom views to next level.
	w .= o
        w = @view w[1:h,1:v]
	o = @view o[1:h,1:v]
    end
    img
end

# hardcoded inverse 2-D Haar transform for comparison
function dehaar!(img :: Matrix{Number},
               L=minimum(trailing_zeros.([size(img)...]));
	       tmp = similar(img))

    # we will be accumulating into tmp, which may not be zeroed.
    tmp[:] .= 0

    # bound the 
    h, v = size(img) .>> (L-1)

    basisll = [1.0  1.0;  1.0  1.0]
    basislh = [1.0  1.0; -1.0 -1.0]
    basishl = [1.0 -1.0;  1.0 -1.0]
    basishh = [1.0 -1.0; -1.0  1.0]
    
    for n=1:L

        # generate views of in/out regions
        w = @view img[1:h, 1:v]
        o = @view tmp[1:h, 1:v]

        # rescale
        h >>= 1
        v >>= 1

        # calculate the coefficient views
	lls = @view w[1:h,1:v]
	lhs = @view w[1:h,v+1:end]
	hls = @view w[h+1:end,1:v]
	hhs = @view w[h+1:end,v+1:end]

        # accumulate output by summing over bases
        for j=1:h, k=1:v
            o[2j-1:2j,2k-1:2k] .+= (lls[j,k] * basisll
                               .+   lhs[j,k] * basislh
                               .+   hls[j,k] * basishl 
                               .+   hhs[j,k] * basishh)
        end

        # rescale, including earlier scaling for slicing.
        h <<= 2 
        v <<= 2

        # resulting image is scaling function for next round.
        # we will have to reset the accumulation.
        w[:] .= o[:]
        o[:] .= 0
    end
    img
end
