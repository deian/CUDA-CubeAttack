
#include <stdio.h>
#include <string.h>

#include "ecrypt-sync.h"
#include "../../../xsr_rng.h"

void perform_test (u8 *key, u8* iv, int iv_length_in_bits)
{
    ECRYPT_ctx ctx;
    u8 keystream[4];

    ECRYPT_keysetup (&ctx, key, 80, iv_length_in_bits);
    ECRYPT_ivsetup (&ctx, iv);
    ECRYPT_keystream_bytes (&ctx, keystream, 4);

    printf("0x%08x,0x%08x,0x%04x,", ((u32*)key)[0], ((u32*)key)[1], ((u32*)key)[2]);
    printf("0x%08x,0x%08x,0x%04x,", ((u32*)iv)[0], ((u32*)iv)[1], ((u32*)iv)[2]);
    printf("0x%08x\n", ((u32*)keystream)[0]);

}


int main()
{

   u32 key[4];//= { 0x78563412, 0xf0debc9a, 0x0003412};
   u32 iv[4];//=  { 0x8a2f539c, 0x2e4beac3, 0x000f5a0};
   int nr_tests=1000000,i;
   ECRYPT_init ();

   {
      FILE *fp;
      u64 seed=time(0);
      fp = fopen("/dev/urandom", "r");
      fread(&seed,sizeof(seed),1,fp);
      fclose(fp);
      xsr_srand_u32((u32)(seed));
      xsr_srand_u64(seed);
   }

   printf("%d\n",nr_tests);
   for(i=0;i<nr_tests;i++) {
      key[0]=xsr_rand_u32(); key[1]=xsr_rand_u32(); key[2]=xsr_rand_u32()&0xffff;
      iv[0]=xsr_rand_u32(); iv[1]=xsr_rand_u32(); iv[2]=xsr_rand_u32()&0xffff;
      perform_test((u8*)key,(u8*)iv, 80);
   }
   return 0;
}
