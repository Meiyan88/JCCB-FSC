function Penalty = get_connectivity(data,alpha)

% Compute the correlation information between channels
for i=1:12
    for j=1:20
        data1(:,j)=data(:,i+(j-1)*12);
    end

X1 = corr(data1).^alpha; 
diag_X1=diag(diag(X1));
X1=X1-diag_X1;     
L1=diag(sum(X1,2));
A(:,:,i)=L1-X1;
   
end
Penalty = zeros(240,240);
for i = 1:12
    for n=1:20
        for j= 1:20
            Penalty((j-1)*12+i,(n-1)*12+i)=A(j,n,i);
            Penalty((n-1)*12+i,(j-1)*12+i)=A(j,n,i);
        end
    end
end

end

