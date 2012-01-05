#ifndef __D_TRIVIUM_KERNEL__
#define __D_TRIVIUM_KERNEL__

#define NR_INIT_ROUNDS 676

#define d_trivium_state(S) \
   S##0,S##1,S##2,S##3,S##4,S##5,S##6,S##7,S##8

#define state(S,a)      S##a

#define s65  (state(S,2)>>1)
#define s68  (state(S,2)>>4)
#define s90  (state(S,2)>>26)
#define s91  (state(S,2)>>27)
#define s92  (state(S,2)>>28)
#define s161 (state(S,5)>>1)
#define s170 (state(S,5)>>10)
#define s174 (state(S,5)>>14)
#define s175 (state(S,5)>>15)
#define s176 (state(S,5)>>16)
#define s242 (state(S,7)>>18)
#define s263 (state(S,8)>>7)
#define s285 (state(S,8)>>29)
#define s286 (state(S,8)>>30)
#define s287 (state(S,8)>>31)

#define update(t1,t2,t3,z)      \
   t1 = s65  ^ s92;             \
   t2 = s161 ^ s176;            \
   t3 = s242 ^ s287;            \
                                \
   z  = t1   ^ t2  ^ t3;        \
                                \
   t1^= (s90  & s91 ) ^ s170;   \
   t2^= (s174 & s175) ^ s263;   \
   t3^= (s285 & s286) ^ s68

#define rotate(t1,t2,t3)                                \
   state(S,8)=(state(S,8)<<1)|(state(S,7)>>31);         \
   state(S,7)=(state(S,7)<<1)|(state(S,6)>>31);         \
   state(S,6)=(state(S,6)<<1)|(state(S,5)>>31);         \
   state(S,5)=(state(S,5)<<1)|(state(S,4)>>31);         \
   state(S,4)=(state(S,4)<<1)|(state(S,3)>>31);         \
   state(S,3)=(state(S,3)<<1)|(state(S,2)>>31);         \
   state(S,2)=(state(S,2)<<1)|(state(S,1)>>31);         \
   state(S,1)=(state(S,1)<<1)|(state(S,0)>>31);         \
   state(S,0)=(state(S,0)<<1)|(t3&0x1);                 \
   state(S,2)=(state(S,2)&0xdfffffff)|((t1&0x1)<<29);   \
   state(S,5)=(state(S,5)&0xfffdffff)|((t2&0x1)<<17)   

#define transform(t1,t2,t3,z)   \
         update(t1,t2,t3,z);    \
         rotate(t1,t2,t3)
   

__device__ __host__ u32 d_trivium(u32 key[3],u32 iv[3],u32 nr_output_bits) {
   u32 i;
   u32 d_trivium_state(S);
   u32 iv0,iv1,iv2,z;
   u32 output=0;
   iv0=iv[0]; iv1=iv[1]; iv2=iv[2];

   /* copy the key to the state */
   state(S,0)=key[0]; state(S,1)=key[1]; state(S,2)=key[2]&0xffff;
   state(S,2)|=(iv0&0x7)<<29; /* S[92,93,94]=IV[0,1,2] */
   iv0=(iv0>>3)|((iv1&0x7)<<29);
   iv1=(iv1>>3)|((iv2&0x7)<<29);
   iv2=(iv2>>3);


   /* copy the rest of the IV */
   state(S,3)=iv0; state(S,4)=iv1; state(S,5)=iv2&0x1fff;
   state(S,6)=state(S,7)=0;
   state(S,8)=0x7<<29;


#if ((NR_INIT_ROUNDS%32)==0)
   #pragma unroll 32 
#elif ((NR_INIT_ROUNDS%16)==0)
   #pragma unroll 16
#elif ((NR_INIT_ROUNDS%8)==0)
   #pragma unroll 8
#elif ((NR_INIT_ROUNDS%4)==0)
   #pragma unroll 4
#elif ((NR_INIT_ROUNDS%2)==0)
   #pragma unroll 2
#endif
   for(i=0;i<NR_INIT_ROUNDS;i++) {
      /*iv0,iv1,iv2 are now free, use them for t1-t2*/
         transform(iv0,iv1,iv2,z);
   }

#ifdef NR_OUTPUT_BITS
   #if NR_OUTPUT_BITS==32
      #pragma unroll 32 
   #elif NR_OUTPUT_BITS==16
      #pragma unroll 16
   #elif NR_OUTPUT_BITS==8
      #pragma unroll 8
   #elif NR_OUTPUT_BITS==4
      #pragma unroll 4
   #elif NR_OUTPUT_BITS==2
      #pragma unroll 2
   #endif
#endif
   for(i=0;i<nr_output_bits;i++) {
      transform(iv0,iv1,iv2,z);
      output|=(z&1)<<i;//(31-i);
   }
   return output;
}


#define black_box_def_key(key)                                  \
   u32 (key)[3];                                                \
   (key)[0]=(key)[1]=(key)[2]=0;                                \

#define black_box_def_iv(iv)                                    \
   u32 (iv)[3];                                                 \
   (iv)[0]=(iv)[1]=(iv)[2]=0;                                   \

#define black_box_def_state(key,iv)                             \
   black_box_def_key(key);                                      \
   black_box_def_iv(iv);                                        \

#define black_box_id2iv(iv,I,nr_idx,threadID)                   \
{                                                               \
  u32 i;                                                        \
  u64 tid=(threadID);                                           \
  (iv)[0]=(iv)[1]=(iv)[2]=0;                                    \
  for(i=0;i<(nr_idx);i++) {                                     \
     (iv)[((I)[i]/32)]|= (tid&0x1)<<((I)[i]-32*((I)[i]/32));    \
     tid>>=1;                                                   \
  }                                                             \
}                                                    

#define black_box_key_set_bitpos(key,bitpos)                    \
{                                                               \
  (key)[0]=(key)[1]=(key)[2]=0;                                 \
  (key)[((bitpos)/32)]|= (0x1)<<((bitpos)-32*((bitpos)/32));    \
}                                                    

#define black_box_key_random(key,rand)                          \
{                                                               \
  (key)[0]=(rand)(); (key)[1]=(rand)(); (key)[2]=(rand)();      \
}

#define black_box_add_keys(keyD,key0,key1)                      \
{                                                               \
   (keyD)[0]=(key0)[0]^(key1)[0];                               \
   (keyD)[1]=(key0)[1]^(key1)[1];                               \
   (keyD)[2]=(key0)[2]^(key1)[2];                               \
}

#define black_box_query(key,iv,nr_output_bits)                  \
   d_trivium(key,iv,nr_output_bits)

#endif
