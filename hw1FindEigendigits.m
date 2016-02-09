function [ m, PC ] = hw1FindEigendigits( A )
%HW1FINDEIGENDIGITS Wrapper for pcaeigreduce.

[m, PC] = pcaeigreduce(A);

end

