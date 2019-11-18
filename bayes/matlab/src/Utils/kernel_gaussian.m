function K = kernel_gaussian(X,Y)
if nargin==1
    Y = X;
end
R = L2_distance(X,Y);
K = exp(-R.^2);
end