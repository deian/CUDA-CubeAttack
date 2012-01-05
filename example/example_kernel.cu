#ifndef __D_EXAMPLE_KERNEL__
#define __D_EXAMPLE_KERNEL__

__device__ __host__ u32 d_example(u32 key[1],u32 iv[1],u32 nr_output_bits) {
   u32 p=0;

   #define x1 (key[0]&1)
   #define x2 ((key[0]>>1)&1)
   #define x3 ((key[0]>>2)&1)
   #define x4 ((key[0]>>3)&1)

   #define v1 (iv[0]&1)
   #define v2 ((iv[0]>>1)&1)
   #define v3 ((iv[0]>>2)&1)
   #define v4 ((iv[0]>>3)&1)

   p = (v1&v2)
      ^(v2&v3&v4)
      ^(v2&v3&v4&x3)
      ^(v1&x2)
      ^(v1&v2&x2)
      ^(v2&v3&x2&x3)
      ^(v1&v2&x1&x3)
      ^(v1&v2&x4)
      ^(v4&x4)
      ^(v2&v3&v4&x4)
      ^(v3&v4&x1&x2&x3&x4);

   #undef x1
   #undef x2
   #undef x3
   #undef x4
   #undef v1
   #undef v2
   #undef v3
   #undef v4

   return p;
}


#define black_box_def_key(key)                                  \
   u32 (key)[1];                                                \
   (key)[0]=0;                                                  \

#define black_box_def_iv(iv)                                    \
   u32 (iv)[1];                                                 \
   (iv)[0]=0;                                                   \

#define black_box_def_state(key,iv)                             \
   black_box_def_key(key);                                      \
   black_box_def_iv(iv);                                        \

#define black_box_id2iv(iv,I,nr_idx,threadID)                   \
{                                                               \
  u32 i;                                                        \
  u64 tid=(threadID);                                           \
  (iv)[0]=0;                                                    \
  for(i=0;i<(nr_idx);i++) {                                     \
     (iv)[((I)[i]/32)]|= (tid&0x1)<<((I)[i]-32*((I)[i]/32));    \
     tid>>=1;                                                   \
  }                                                             \
}                                                    

#define black_box_key_set_bitpos(key,bitpos)                    \
{                                                               \
  (key)[0]=0;                                                   \
  (key)[((bitpos)/32)]|= (0x1)<<((bitpos)-32*((bitpos)/32));    \
}                                                    

#define black_box_key_random(key,rand)                          \
{                                                               \
  (key)[0]=(rand)(); (key)[1]=(rand)(); (key)[2]=(rand)();      \
}

#define black_box_add_keys(keyD,key0,key1)                      \
{                                                               \
   (keyD)[0]=(key0)[0]^(key1)[0];                               \
}

#define black_box_query(key,iv,nr_output_bits)                  \
   d_example(key,iv,nr_output_bits)

#endif
