#include <cutil_inline.h>
#include <stdio.h>
#include <stdint.h>
#include "../xsr_rng.h"

typedef uint32_t u32;

__global__ static void test_inline_xsr_rand_u32(u32 *seed, int N, u32 *out) {
   int i;
   int tid=blockIdx.x*blockDim.x+threadIdx.x;
   inline_xsr_def_u32();
   inline_xsr_srand_u32(seed[tid]);
   for(i=0;i<N-1;i++) {
      inline_xsr_rand_u32();
   }
   out[tid]=inline_xsr_rand_u32();
}


int test_inline_xsr(int dev, int nr_threads, int nr_blocks,int N) {
   int i,j;
   u32 *seed_h,*seed_d;
   u32 *out_h,*out_d;
   cudaDeviceProp deviceProp;
   unsigned int timer = 0;
   double proc_time=-1.0;
   size_t size=(nr_threads*nr_blocks)*sizeof(u32);

   cudaGetDeviceProperties(&deviceProp, dev);
   printf("\nUsing device %d: \"%s\"\n", dev, deviceProp.name);
   printf("\nClock rate: %d\n",deviceProp.clockRate);
   cudaSetDevice(dev);

   cutilSafeCall(cudaHostAlloc((void**)&(seed_h),size,cudaHostAllocPortable));
   cutilSafeCall(cudaMalloc((void**)&(seed_d),size));

   cutilSafeCall(cudaHostAlloc((void**)&(out_h),size,cudaHostAllocPortable));
   cutilSafeCall(cudaMalloc((void**)&(out_d),size));

   {
      FILE *fp;
      if(!(fp = fopen("/dev/urandom", "r"))) {
         fprintf(stderr,"Failed to open /dev/urandom");
         return -1;
      }
      fread(seed_h,sizeof(u32),nr_threads*nr_blocks,fp);
      fclose(fp);
   }

   cutilCheckError(cutCreateTimer( &timer));
   cutilCheckError(cutStartTimer( timer));
#define NR_TEST_ITERATIONS 1//000
   for(int nr_test_iter=0;nr_test_iter<NR_TEST_ITERATIONS;nr_test_iter++) 
   {


      cutilSafeCall(cudaMemcpy(seed_d,seed_h,size,cudaMemcpyHostToDevice));

      test_inline_xsr_rand_u32<<<nr_blocks,nr_threads>>>(seed_d,N,out_d);
      cutilCheckMsg("Kernel execution failed");
      cudaThreadSynchronize();

      cutilSafeCall(cudaMemcpy(out_h,out_d,size,cudaMemcpyDeviceToHost));


   }
   cutilCheckError(cutStopTimer( timer));
   proc_time=cutGetTimerValue( timer);
   cutilCheckError(cutDeleteTimer( timer));

   for(i=0;i<nr_threads*nr_blocks;i++) {
      u32 out_val=0;
      xsr_srand_u32(seed_h[i]);
      for(j=0;j<N;j++) { out_val=xsr_rand_u32(); }
      if(out_val!=out_h[i]) {
         printf("failed @ [%4d] -> seed=%08x cpu=%08x gpu=%08x\n",i,
                                             seed_h[i],out_val,out_h[i]);
      }
   }


   cutilSafeCall(cudaFree(seed_d));
   cutilSafeCall(cudaFree(out_d));
   cutilSafeCall(cudaFreeHost(seed_h));
   cutilSafeCall(cudaFreeHost(out_h));

   printf( "\nProcessing time: %8g (ms), %g (us/byte) %g (cycles/byte)\n"
         ,proc_time
         ,(1000.0*proc_time)/(NR_TEST_ITERATIONS*N)/
                           (nr_threads*nr_blocks*sizeof(u32))
         ,((proc_time)*(deviceProp.clockRate))/(NR_TEST_ITERATIONS*N)/
                           (nr_threads*nr_blocks*sizeof(u32)));


   return 0;
}
int main(void) {
   return test_inline_xsr(3,256,256,100000);
}
