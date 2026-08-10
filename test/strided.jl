using Waiver

x = testimage("cameraman")
@time y = nsdwt(x, WT_HAAR, (2, 3); wpt=true);

x = testimage("cameraman") .|> Float32
@time y = nsdwt(x, WT_D4, (2, 3); wpt=true);

x = testimage("cameraman") .|> Float32
@time y = dwt(x, WT_D4, 2);
idwt!(y, WT_D4, 2);
x ≈ y


x = testimage("cameraman")
nsdwt(x, WT_HAAR, 1)
