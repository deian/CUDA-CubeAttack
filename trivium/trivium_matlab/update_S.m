function [S,z] = update_S(S)
t1=xor( S(66) , S(93)  );
t2=xor( S(162), S(177) );
t3=xor( S(243), S(288) );


z=xor( xor( t1 , t2), t3 );

t1=xor(xor( t1, and( S(91) , S(92)  ) ), S(171));
t2=xor(xor( t2, and( S(175), S(176) ) ), S(264));
t3=xor(xor( t3, and( S(286), S(287) ) ), S(69) );

S(1:93)   =[t3, S(1:92)];
S(94:177) =[t1, S(94:176)];
S(178:288)=[t2, S(178:287)];
