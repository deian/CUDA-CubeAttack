function z = trivium(key,iv,nr_output_bytes)

S=zeros(1,288);

S(1:80)=key;
S(94:173)=iv;
S(286:288)=ones(1,3);

for i=1:675%4*288
    S=update_S(S);
end

z=zeros(1,nr_output_bytes*8);
for i=1:length(z)
    [S,tmp]=update_S(S);
    z(i)=tmp;
end
