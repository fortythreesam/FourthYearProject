function T = MyConvMatrix(H, m, n)
s = size(H);
T = zeros((m-s(1)+1) * (n-s(2)+1), m*n);
k = 1;
for i=1:(m-s(1)+1)
 for j=1:(n-s(2)+1)
  
  for p=1:s(1)
   T(k,(i-1+p-1)*n+(j-1)+1:(i-1+p-1)*n+(j-1)+1+s(2)-1) = H(p,:);
  end
  
  k = k + 1;
 end
end
