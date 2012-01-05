#ifndef __D_MICKEY_KERNEL__
#define __D_MICKEY_KERNEL__
#include "mickey.h"


#define R_Mask0   0x1279327b
#define R_Mask1   0xb5546660
#define R_Mask2   0xdf87818f
#define R_Mask3   0x00000003

#define Comp00    0x6aa97a30
#define Comp01    0x7942a809
#define Comp02    0x057ebfea
#define Comp03    0x00000006
#define Comp10    0xdd629e9a
#define Comp11    0xe3a21d63
#define Comp12    0x91c23dd7
#define Comp13    0x00000001

#define S_Mask00  0x9ffa7faf
#define S_Mask01  0xaf4a9381
#define S_Mask02  0x9cec5802
#define S_Mask03  0x00000001

#define S_Mask10  0x4c8cb877
#define S_Mask11  0x4911b063
#define S_Mask12  0x40fbc52b
#define S_Mask13  0x00000008

#define d_mickey_state(R,S) \
   R##0,R##1,R##2,R##3,S##0,S##1,S##2,S##3

#define state(X,a)      X##a


#define CLOCK_R(R,input_bit,control_bit)                                   \
{                                                                          \
    int Feedback_bit;                                                      \
    int Carry0, Carry1, Carry2;                                            \
                                                                           \
    Feedback_bit = ((state(R,3) >> 3) & 1) ^ (input_bit);                  \
    Carry0 = (state(R,0) >> 31) & 1;                                       \
    Carry1 = (state(R,1) >> 31) & 1;                                       \
    Carry2 = (state(R,2) >> 31) & 1;                                       \
                                                                           \
    if ((control_bit)) {                                                   \
        state(R,0) ^= (state(R,0) << 1);                                   \
        state(R,1) ^= (state(R,1) << 1) ^ Carry0;                          \
        state(R,2) ^= (state(R,2) << 1) ^ Carry1;                          \
        state(R,3) ^= (state(R,3) << 1) ^ Carry2;                          \
    } else {                                                               \
        state(R,0) = (state(R,0) << 1);                                    \
        state(R,1) = (state(R,1) << 1) ^ Carry0;                           \
        state(R,2) = (state(R,2) << 1) ^ Carry1;                           \
        state(R,3) = (state(R,3) << 1) ^ Carry2;                           \
    }                                                                      \
                                                                           \
    if (Feedback_bit) {                                                    \
       state(R,0) ^= R_Mask0;                                              \
       state(R,1) ^= R_Mask1;                                              \
       state(R,2) ^= R_Mask2;                                              \
       state(R,3) ^= R_Mask3;                                              \
    }                                                                      \
}

#define CLOCK_S(S,input_bit,control_bit)                                   \
{                                                                          \
    int Feedback_bit;                                                      \
    int Carry0, Carry1, Carry2;                                            \
                                                                           \
    Feedback_bit = ((state(S,3) >> 3) & 1) ^ (input_bit);                  \
    Carry0 = (state(S,0) >> 31) & 1;                                       \
    Carry1 = (state(S,1) >> 31) & 1;                                       \
    Carry2 = (state(S,2) >> 31) & 1;                                       \
                                                                           \
    state(S,0) = (state(S,0) << 1) ^ ((state(S,0) ^ Comp00) &              \
          ((state(S,0) >> 1) ^ (state(S,1) << 31) ^ Comp10) & 0xfffffffe); \
    state(S,1) = (state(S,1) << 1) ^ ((state(S,1) ^ Comp01) &              \
          ((state(S,1) >> 1) ^ (state(S,2) << 31) ^ Comp11)) ^ Carry0;     \
    state(S,2) = (state(S,2) << 1) ^ ((state(S,2) ^ Comp02) &              \
          ((state(S,2) >> 1) ^ (state(S,3) << 31) ^ Comp12)) ^ Carry1;     \
    state(S,3) = (state(S,3) << 1) ^ ((state(S,3) ^ Comp03) &              \
          ((state(S,3) >> 1) ^ Comp13) & 0x7) ^ Carry2;                    \
                                                                           \
    if (Feedback_bit) {                                                    \
        if ((control_bit)) {                                               \
            state(S,0) ^= S_Mask10;                                        \
            state(S,1) ^= S_Mask11;                                        \
            state(S,2) ^= S_Mask12;                                        \
            state(S,3) ^= S_Mask13;                                        \
        } else {                                                           \
            state(S,0) ^= S_Mask00;                                        \
            state(S,1) ^= S_Mask01;                                        \
            state(S,2) ^= S_Mask02;                                        \
            state(S,3) ^= S_Mask03;                                        \
        }                                                                  \
    }                                                                      \
}

#define CLOCK_KG_INIT(R,S,input_bit)                                       \
{                                                                          \
    int control_bit_r;                                                     \
    int control_bit_s;                                                     \
                                                                           \
    control_bit_r = ((state(S,1) >> 2) ^ (state(R,2) >> 3)) & 1;           \
    control_bit_s = ((state(R,1) >> 1) ^ (state(S,2) >> 3)) & 1;           \
                                                                           \
    CLOCK_R(R, ((state(S,1) >> 18) & 1) ^ (input_bit), control_bit_r);     \
    CLOCK_S(S, (input_bit), control_bit_s);                                \
                                                                           \
}


#define CLOCK_KG_KS(ks,R,S)                                                \
{                                                                          \
    int control_bit_r;                                                     \
    int control_bit_s;                                                     \
                                                                           \
    ks = (state(R,0) ^ state(S,0)) & 1;                                    \
    control_bit_r = ((state(S,1) >> 2) ^ (state(R,2) >> 3)) & 1;           \
    control_bit_s = ((state(R,1) >> 1) ^ (state(S,2) >> 3)) & 1;           \
                                                                           \
    CLOCK_R(R, 0, control_bit_r);                                          \
    CLOCK_S(S, 0, control_bit_s);                                          \
}


#define min(a,b) ((a)<(b))?(a):(b)

__device__ __host__ u32 d_mickey(u32 key0, u32 key1, u32 key2,
                                 u32  iv0, u32  iv1, u32  iv2,
                                 u32 nr_output_bits) {

   int i,z;
   int iv_size=MICKEY_IV_SIZE;
   u32 output=0;



   /* define state */
   u32 d_mickey_state(R,S);

   /* initialize R and S */
   state(R,0)=state(R,1)=state(R,2)=state(R,3)=0;
   state(S,0)=state(S,1)=state(S,2)=state(S,3)=0;

   /* - load IV ----------------------------------------------------- */
   {
      int nr_iv_clks=min(iv_size,32);
#if (MICKEY_IV_SIZE>=32)
#pragma unroll 32
#endif
      for(i=0;i<nr_iv_clks;i++) { CLOCK_KG_INIT(R,S,(iv0>>(31-i))&1); }
      iv_size-=nr_iv_clks; nr_iv_clks=min(iv_size,32);
#if (MICKEY_IV_SIZE>=64)
#pragma unroll 32
#endif
      for(i=0;i<nr_iv_clks;i++) { CLOCK_KG_INIT(R,S,(iv1>>(31-i))&1); }
      iv_size-=nr_iv_clks; nr_iv_clks=min(iv_size,32);
#if (MICKEY_IV_SIZE==80)
#pragma unroll 16
#endif
      for(i=0;i<nr_iv_clks;i++) { CLOCK_KG_INIT(R,S,(iv2>>(31-i))&1); }
   }
   /* --------------------------------------------------------------- */


   /* - load key ---------------------------------------------------- */
   {
#pragma unroll 32 
      for(i=0;i<32;i++) { CLOCK_KG_INIT(R,S,(key0>>(31-i))&1); }
#pragma unroll 32 
      for(i=0;i<32;i++) { CLOCK_KG_INIT(R,S,(key1>>(31-i))&1); }
#pragma unroll 16 
      for(i=0;i<16;i++) { CLOCK_KG_INIT(R,S,(key2>>(31-i))&1); }
   }
   /* --------------------------------------------------------------- */

   /* - preclock ---------------------------------------------------- */
#if ((NR_INIT_ROUNDS%100)==0)
   #pragma unroll 100
#elif ((NR_INIT_ROUNDS%32)==0)
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
    for(i=0;i<NR_INIT_ROUNDS;i++) { CLOCK_KG_INIT(R,S,0); }
   /* --------------------------------------------------------------- */

   /* - generate keystream ------------------------------------------ */

#if (NR_TEST_ITERATIONS>1)
#pragma unroll 1
   for(int nr_test_iter=0;nr_test_iter<NR_TEST_ITERATIONS;nr_test_iter++)  {
   output=0;
#endif

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
   for(i=0;i<nr_output_bits-1;i++) {
      CLOCK_KG_KS(z,R,S)
      output|=(z<<(31-i));
   }
   z=(state(R,0) ^ state(S,0)) & 1;
   output|=(z<<(31-i));
#if (NR_TEST_ITERATIONS>1)
   }
#endif
   /* --------------------------------------------------------------- */

   return output;
}


/*
   For our application we a only care about the word-length on
   the GPU, which is 32-bits, so this implementation is slightly
   limited though it's quite easy to extend.
 */
int d_mickey_test(char *fname) {
   u32 z;
   FILE *fp;

   int i,nr_tests;
   u32 test_key[3],test_iv[3],test_z;

   if(!(fp=fopen(fname,"r"))) {
      fprintf(stderr,"d_mickey_test: Failed to open \'%s\'\n", fname);
      return -1;
   }

   fscanf(fp,"%u\n",&nr_tests);
   debug("nr_tests=%d\n",nr_tests);

   for(i=0;i<nr_tests;i++) {
    fscanf(fp,"%x,%x,%x,%x,%x,%x,%x,\n",&test_key[0],&test_key[1],&test_key[2],
                                         &test_iv[0],&test_iv[1],&test_iv[2],
                                         &test_z);

#define ch_endianess(a)                                 \
{                                                       \
   a=(((a>>24)&0x000000ff) | ((a>> 8)&0x0000ff00)       \
    | ((a<< 8)&0x00ff0000) | ((a<<24)&0xff000000));     \
}
      ch_endianess(test_key[0]);        ch_endianess(test_iv[0]);
      ch_endianess(test_key[1]);        ch_endianess(test_iv[1]);
      ch_endianess(test_key[2]);        ch_endianess(test_iv[2]);
      ch_endianess(test_z);

#undef ch_endianess


      z=d_mickey(test_key[0],test_key[1],test_key[2],
                  test_iv[0],test_iv[1],test_iv[2],NR_OUTPUT_BITS);

      if(z^( test_z & (0xFFFFFFFF>>(32-NR_OUTPUT_BITS)))) {
         fprintf(stderr,"d_mickey_test: Failed test number %2d:"
               "%08x!=%08x\n",i,z,test_z);
         fclose(fp);
         return -1;
      }
   }

   fclose(fp);
   return 0;
}

/*
int main() {
         d_mickey_test("mickey.long.test.32");
   return 0;
}
*/
#define black_box_key(key)                                      \
   key##0,key##1,key##2

#define black_box_iv(iv)                                        \
   iv##0,iv##1,iv##2

#define black_box_clear_key(key)                                \
   key##0=key##1=key##2=0;                                      \

#define black_box_clear_iv(iv)                                  \
   iv##0=iv##1=iv##2=0;                                         \

#define black_box_id2iv(iv,I,nr_idx,threadID)                   \
{                                                               \
  u32 _i;                                                       \
  u64 tid=(threadID);                                           \
  black_box_clear_iv(iv);                                       \
  for(_i=0;_i<(nr_idx);_i++) {                                  \
     u32 mask = (tid&0x1)<<((I)[_i]-32*((I)[_i]/32));           \
     switch(((I)[_i]/32)) {                                     \
      case 0: iv##0 |= mask; break;                             \
      case 1: iv##1 |= mask; break;                             \
      case 2: iv##2 |= mask; break;                             \
     }                                                          \
     tid>>=1;                                                   \
  }                                                             \
}                                                    

#define black_box_key_set_bitpos(key,bitpos)                    \
{                                                               \
   u32 mask=(0x1)<<((bitpos)-32*((bitpos)/32));                 \
   switch(((bitpos)/32)) {                                      \
      case 0: key##0 |= mask; break;                            \
      case 1: key##1 |= mask; break;                            \
      case 2: key##2 |= mask; break;                            \
   }                                                            \
}                                                    

#define black_box_key_random(key,rand)                          \
{                                                               \
   key##0=rand();                                               \
   key##1=rand();                                               \
   key##2=rand();                                               \
}

#define black_box_add_keys(keyD,key0,key1)                      \
{                                                               \
   keyD##0=key0##0^key1##0;                                     \
   keyD##1=key0##1^key1##1;                                     \
   keyD##2=key0##2^key1##2;                                     \
}

/* - general macros ------------------------------------------ */
#define black_box_def_key(key)                                  \
   u32 black_box_key(key);

#define black_box_def_iv(iv)                                    \
   u32 black_box_iv(iv);

#define black_box_def_state(key,iv)                             \
   black_box_def_key(key);                                      \
   black_box_def_iv(iv);                                        \
   black_box_clear_key(key);                                    \
   black_box_clear_iv(iv);                                      

#define black_box_test(fname)                                   \
   d_mickey_test(fname)

#define black_box_query(key,iv,nr_output_bits)                  \
   d_mickey(black_box_key(key),black_box_iv(iv),nr_output_bits)
/* ----------------------------------------------------------- */
#endif
