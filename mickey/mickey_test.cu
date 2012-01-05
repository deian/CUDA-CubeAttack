#include <cutil_inline.h>
#include <stdio.h>
#include <stdint.h>
#include "../xsr_rng.h"


//------
typedef uint32_t u64;
typedef uint32_t u32;
typedef uint8_t u8;

#define debug printf
//------
#define NR_TEST_ITERATIONS 1//0000
#include "mickey_kernel.cu"


__global__ static void test_mickey(u32 *key, u32 *iv, u32 *out) {
   int tid=blockIdx.x*blockDim.x+threadIdx.x;
   int off=tid*3;
#if 0
   printf("gpu [%4d] -> key=[%08x,%08x,%04x] iv=[%08x,%08x,%04x]\n"
         ,tid,key[off+0],key[off+1],key[off+2],iv[off+0],iv[off+1],iv[off+2]);
#endif
   out[tid]=d_mickey(key[off+0],key[off+1],key[off+2],
                      iv[off+0], iv[off+1], iv[off+2],NR_OUTPUT_BITS);
}


int test_mickey_gpu(int dev, int nr_threads, int nr_blocks) {
   int i;
   u32 *key_h,*key_d;
   u32 *iv_h,*iv_d;
   u32 *out_h,*out_d;
   cudaDeviceProp deviceProp;
   unsigned int timer = 0;
   double proc_time=-1.0;

   size_t key_size=(nr_threads*nr_blocks*3)*sizeof(u32);
   size_t iv_size=(nr_threads*nr_blocks*3)*sizeof(u32);
   size_t out_size=(nr_threads*nr_blocks)*sizeof(u32);

   cudaGetDeviceProperties(&deviceProp, dev);
   printf("\nUsing device %d: \"%s\"\n", dev, deviceProp.name);
   printf("\nClock rate: %d\n",deviceProp.clockRate);
   cudaSetDevice(dev);

   cutilSafeCall(cudaHostAlloc((void**)&(key_h),key_size,
                                          cudaHostAllocPortable));
   cutilSafeCall(cudaMalloc((void**)&(key_d),key_size));

   cutilSafeCall(cudaHostAlloc((void**)&(iv_h),iv_size,
                                          cudaHostAllocPortable));
   cutilSafeCall(cudaMalloc((void**)&(iv_d),iv_size));

   cutilSafeCall(cudaHostAlloc((void**)&(out_h),out_size,
                                          cudaHostAllocPortable));
   cutilSafeCall(cudaMalloc((void**)&(out_d),out_size));

   {
      FILE *fp;
      u64 seed=time(0);
      fp = fopen("/dev/urandom", "r");
      fread(&seed,sizeof(seed),1,fp);
      fclose(fp);
      xsr_srand_u32((u32)(seed));
      xsr_srand_u64(seed);
   }

   {
      /* create the random keys and random ivs */
      u32 *pkey=&key_h[0];
      u32 *piv=&iv_h[0];
      for(i=0;i<nr_threads*nr_blocks;i++) {
         pkey[0]=xsr_rand_u32();        piv[0]=xsr_rand_u32();
         pkey[1]=xsr_rand_u32();        piv[1]=xsr_rand_u32();
         pkey[2]=xsr_rand_u32()&0xffff; piv[2]=xsr_rand_u32()&0xffff;
         pkey+=3;                       piv+=3;
      }
   }

   cutilCheckError(cutCreateTimer( &timer));
   cutilCheckError(cutStartTimer( timer));

   cutilSafeCall(cudaMemcpy(key_d,key_h,key_size,cudaMemcpyHostToDevice));
   cutilSafeCall(cudaMemcpy(iv_d,iv_h,iv_size,cudaMemcpyHostToDevice));

   test_mickey<<<nr_blocks,nr_threads>>>(key_d,iv_d,out_d);
   cutilCheckMsg("Kernel execution failed");
   cudaThreadSynchronize();

   cutilSafeCall(cudaMemcpy(out_h,out_d,out_size,cudaMemcpyDeviceToHost));


   cutilCheckError(cutStopTimer( timer));
   proc_time=cutGetTimerValue( timer);
   cutilCheckError(cutDeleteTimer( timer));

   {
      u32 *pkey=key_h;
      u32 *piv=iv_h;
      for(i=0;i<nr_threads*nr_blocks;i++) {
         u32 out_val=0;
         out_val=d_mickey(pkey[0],pkey[1],pkey[2],
                          piv[0],piv[1],piv[2],NR_OUTPUT_BITS);
         if(out_val!=out_h[i]) {
            printf("failed @ [%4d] -> key=[%08x,%08x,%04x] iv=[%08x,%08x,%04x]"
                  " cpu=%08x gpu=%08x\n",i,pkey[0],pkey[1],pkey[2],
                                          piv[0],piv[1],piv[2],
                                          out_val,out_h[i]);
         }

         pkey+=3;
         piv+=3;
      }
   }


   cutilSafeCall(cudaFree(key_d));
   cutilSafeCall(cudaFree(iv_d));
   cutilSafeCall(cudaFree(out_d));
   cutilSafeCall(cudaFreeHost(key_h));
   cutilSafeCall(cudaFreeHost(iv_h));
   cutilSafeCall(cudaFreeHost(out_h));

   printf( "\nProcessing time: %8g (ms), %g (us/byte) %g (cycles/byte)\n"
         ,proc_time
         ,(1000.0*proc_time)/(NR_TEST_ITERATIONS)/
                           (nr_threads*nr_blocks*sizeof(u32))
         ,((proc_time)*(deviceProp.clockRate))/(NR_TEST_ITERATIONS)/
                           (nr_threads*nr_blocks*sizeof(u32)));
   return 0;
}
int main(void) {
   return test_mickey_gpu(3,128,512);
}
