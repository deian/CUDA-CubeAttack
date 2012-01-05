#include "xsr_rng.h"
#include  <stdio.h>

/* see Marsaglia's "Xorshift RNGs" paper */

static uint32_t __xsr_seed_u32=2463534242;

void xsr_srand_u32(uint32_t seed) { __xsr_seed_u32=seed; }

uint32_t xsr_rand_u32(void) {
   __xsr_seed_u32^=(__xsr_seed_u32<<13);
   __xsr_seed_u32^=(__xsr_seed_u32>>17);
   __xsr_seed_u32^=(__xsr_seed_u32<<5);
   return __xsr_seed_u32;

}

/* ---------------------------------------------------------------- */

static uint64_t __xsr_seed_u64=88172645463325252LL;

void xsr_srand_u64(uint64_t seed) { __xsr_seed_u64=seed; }

uint64_t xsr_rand_u64(void){
   __xsr_seed_u64^=(__xsr_seed_u64<<13);
   __xsr_seed_u64^=(__xsr_seed_u64>>7);
   __xsr_seed_u64^=(__xsr_seed_u64<<17);
   return __xsr_seed_u64;
}
/* random integer [a,b] */
int rand_int(int a, int b) {
   if(sizeof(int)==sizeof(uint64_t)) {
      return a+(xsr_rand_u64()%(b-a+1));
   } else {
      return a+(xsr_rand_u32()%(b-a+1));
   }
}
