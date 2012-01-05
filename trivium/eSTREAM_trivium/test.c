#include <stdio.h>
#include <string.h>
#include <time.h>
#include "ecrypt-sync.h"
#include "d_trivium.h"

typedef struct {
   unsigned long y;
} xs_rng;

void xs_seed(xs_rng *rng, unsigned long seed){
   rng->y=seed;
}

unsigned long xs_rand(xs_rng *rng) {
   rng->y^=rng->y<<13;
   rng->y^=rng->y>>17;
   rng->y^=rng->y<<5;
   return rng->y;
}


#define def_print_bin(utype)            \
void print_bin_##utype(utype w) {       \
   int i;                               \
   for(i=0;i<sizeof(w)*CHAR_BIT;i++) {  \
      printf("%d",(w&1)?1:0);           \
      w>>=1;                            \
   }                                    \
}

#define def_print_bin_array(utype)                              \
void print_bin_array_##utype(utype *w,int count) {              \
   while(count--) { print_bin_##utype(*w++); printf("\n");}      \
}

def_print_bin(u8)
def_print_bin_array(u8)

def_print_bin(u32)
def_print_bin_array(u32)

void print_x32_array(u32 *w,int nr) {
   int i;
   for(i=0;i<nr;i++) {
      printf("%08x,",w[i]);
   }
}
void print_x32_rev_bits(u32 *out) {
   int i;
   u32 w=0;
   for(i=0;i<32;i++) { w|=((out[0]>>i)&1)<<(31-i); }
   printf("%08x",w);
}


#define black_box_id2iv(iv,I,nr_idx,threadID)                   \
{                                                               \
  u32 i;                                                        \
  u32 tid=(threadID);                                           \
  for(i=0;i<(nr_idx);i++) {                                     \
     (iv)[((I)[i]/32)]|= (tid&0x1)<<((I)[i]-32*((I)[i]/32));    \
     tid>>=1;                                                   \
  }                                                             \
}                                                    

#define black_box_key_set_bitpos(key,bitpos)                    \
{                                                               \
   (key)[((bitpos)/32)]|= (0x1)<<((bitpos)-32*((bitpos)/32));    \
}     

#define NR_PROC_BYTES 4
#define NR_TESTS 1000000
int main()
{
  ECRYPT_ctx ctx;
  int i,t;
  u8 key[10];
  u8 iv[10];
  u8 in[NR_PROC_BYTES],out[NR_PROC_BYTES];
  u32 dkey[3],div[3];

  xs_rng rng;

  xs_seed(&rng,time(0));
//  xs_seed(&rng,1337);

  memset(in,0x00,sizeof(in)); memset(out,0x00,sizeof(out));

  memset(dkey,0,3*sizeof(u32));
  memset(div,0,3*sizeof(u32));

  printf("%d\n",NR_TESTS);

  for(t=0;t<NR_TESTS;t++) {

     /*
        memset(key,0xff,sizeof(key));
        memset(iv,0xff,sizeof(iv));
      */
     for(i=0;i<sizeof(key);i++) {
        int tmp=xs_rand(&rng);
        key[i]=tmp&0xff;
        iv[i]=(tmp>>8)&0xff;
     }
     //  memcpy(dkey,key,sizeof(key));
     //  memcpy(div,iv,sizeof(iv));

     //  printf("key:\n"); print_bin_array_u8(key,10); printf("\n");
     //  printf("iv:\n"); print_bin_array_u8(iv,10); printf("\n");

     ECRYPT_init();
     ECRYPT_keysetup(&ctx,key,80,80);
     ECRYPT_ivsetup(&ctx,iv);


     //  printf("state:\n"); print_bin_array_u8(ctx.s,40); printf("\n");
     ECRYPT_process_bytes(0,&ctx,in,out,sizeof(in));
     print_x32_array(key,3);
     print_x32_array(iv,3);
     print_x32_rev_bits(out);
     printf("\n");
     //  printf("state:\n"); print_bin_array_u8(ctx.s,40); printf("\n");

     //  printf("output:\n"); print_bin_array_u8(out,NR_PROC_BYTES); printf("\n");

     //  printf("\n-----------------------\n");
     //  printf("dkey:\n"); print_bin_array_u32(dkey,3); printf("\n");
     //  printf("div:\n"); print_bin_array_u32(div,3); printf("\n");

#if 0
     {
        int term[12]={3,13,18,26,38,40,47,49,55,57,66,79};
        int nr_terms=12;
        u32 const_term=0;
        int tm;
        int k;
        for(tm=-1;tm<80;tm++) {
           int sum=0,output=0;
           memset(dkey,0,3*sizeof(u32));
           if(tm!=-1) {
              black_box_key_set_bitpos(dkey,tm);
           }
           //        printf("dkey:\n"); print_bin_array_u32(dkey,3); printf("\n");
           for(k=0;k<(1<<nr_terms);k++)
           {
              memset(div,0,3*sizeof(u32));
              black_box_id2iv(div,term,12,k);
              //        printf("div[%3d]:\n",k); print_bin_array_u32(div,3); printf("\n");
              output=d_trivium(dkey,div,32);
              sum^=output;
           }
           if(tm==-1) {
              const_term=sum;
           } else {
              printf("x_%02d=%08x\n",tm,(sum^const_term)&1);
           }
        }
     }
#endif
  }


  return 0;

}

