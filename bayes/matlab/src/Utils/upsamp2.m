function y = upsamp2(x,factor)

  if (nargin==1)
    factor = 1 ; 
  end
  for cnt=1:factor
    y = zeros(2*size(x)) ;
    y(1:2:end,1:2:end) = x ;
    y(2:2:end,1:2:end) = x ;
    y(1:2:end,2:2:end) = x ;
    y(2:2:end,2:2:end) = x ;
    x = y ;
  end
end
