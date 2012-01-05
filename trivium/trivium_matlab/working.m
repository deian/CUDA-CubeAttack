clc
clear


randint=@(a,b) round(a+(b-a).*rand());
terms=[11,18,20,33,45,47,53,60,61,63,69,78]+1;


N=50;
x=zeros(N,80); fx=zeros(N,1);
for i=1:N
   key=[dec2binvec(randint(0,2^20-1),20) dec2binvec(randint(0,2^20-1),20) dec2binvec(randint(0,2^20-1),20) dec2binvec(randint(0,2^20-1),20)];
   iv=zeros(1,80);
   x(i,:)=key;
   fx(i)=pI(terms,key,iv);
end

for i=1:N/2
   key=xor(x(i,:),x(i+N/2,:));
   iv=zeros(1,80);
   fx2(i)=pI(terms,key,iv);
end

key=zeros(1,80); iv=zeros(1,80);
f0=pI(terms,key,iv);
