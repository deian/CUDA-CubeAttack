#ifndef __XSR_RNG_H__
#define __XSR_RNG_H__
#include <stdint.h>

void xsr_srand_u32(uint32_t seed);
uint32_t xsr_rand_u32(void);

void xsr_srand_u64(uint64_t seed);
uint64_t xsr_rand_u64(void);
int rand_int(int a, int b);

#define inline_xsr_def_u32()                            \
   u32 __inline_xsr_seed_u32;

#define inline_xsr_srand_u32(seed)                      \
   { __inline_xsr_seed_u32=seed; }


/* TODO: a functional approach (which is less efficent
   and slightly ugly) to keep the API simpler */

#define xor_lsh(s,f)                                    \
   ((s)^((s)<<(f)))

#define xor_rsh(s,f)                                    \
   ((s)^((s)>>(f)))

#define inline_xsr_rand_u32()                           \
   __inline_xsr_seed_u32=                               \
   xor_lsh(xor_rsh(xor_lsh(__inline_xsr_seed_u32,13),17),5)

#define inline_rand_int(a,b) \
      (a)+((inline_xsr_rand_u32())%((b)-(a)+1));

#endif
