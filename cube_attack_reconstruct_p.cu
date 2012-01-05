#include <cutil_inline.h>
#include <stdio.h>
#include <multithreading.h>
#include "cube_attack.h"
#include "xsr_rng.h"
//#include "xsr_rng_kernel.cu"

//#include "mickey/mickey_kernel.cu"
#include "trivium/trivium_kernel.cu"
//#include "example/example_kernel.cu"


extern __shared__ __align__ (__alignof(void*)) u32 smem_cache[];

/* - reconstruct linear superpoly ---------------------------------------- */

__global__ static void __kernel_reconstruct_linsuperpoly_h(u32 devId,
                                                          u32 devDim,
                                                          u32 *I,
                                                          int nr_idx,
                                                          u32 *out,
                                                          int term,
                                                          u32 nr_iter) {

   u32* reduce_csh=(u32*) smem_cache;
   int i;
   u32 iter;
   u64 threadID=devId*devDim*blockDim.x+blockIdx.x*blockDim.x+threadIdx.x;
   u32 sum=0;

   black_box_def_state(key,iv);

   if(term!=CONSTANT_TERM) {
      black_box_clear_key(key);
      black_box_key_set_bitpos(key,term);
   }
   
   for(iter=0;iter<nr_iter;iter++) {
      black_box_id2iv(iv,I,nr_idx,threadID);
      sum^=black_box_query(key,iv,NR_OUTPUT_BITS);

      threadID+=devDim*blockDim.x; //threadID+=(total # threads)
   }
   /* write result to shared memory */
   reduce_csh[threadIdx.x]=sum;
   __syncthreads();

   /* reduce results of threads in same block */
   for(i=blockDim.x/2;i>0;i/=2) {
      if(threadIdx.x<i) {
         reduce_csh[threadIdx.x]^=reduce_csh[threadIdx.x+i];
      }
      __syncthreads();
   }

   /* write result from first thread thread */
   if(threadIdx.x==0) {
      out[blockIdx.x]=reduce_csh[0];
   }
}

__global__ static void __kernel_reconstruct_linsuperpoly_m(u32 devId,
                                                          u32 devDim,
                                                          u32 *I,
                                                          int nr_idx,
                                                          u32 *out,
                                                          u32 nr_iter) {

   u32* reduce_csh=(u32*) smem_cache;
   u16* smemI=(u16*) &smem_cache[blockDim.x]; /* reduce_csh uses 0->blockDim.x-1 */
   int i;
   u32 iter;
   u64 threadID=threadIdx.x;
   int term;
   u32 sum=0;

   black_box_def_state(key,iv);

   if(threadIdx.x<nr_idx) {
      /* if there are enough threads (there should be), read in parallel */
      smemI[threadIdx.x]=I[threadIdx.x];
   }
   if(nr_idx>blockDim.x && threadIdx.x==0) {
      /* highly unlikely branch, but write the rest by thread 0 */
      for(int i=nr_idx-blockDim.x;i<nr_idx;i++) {
            smemI[i]=I[i];
      }
   }
   __syncthreads();


   term=devId*devDim+blockIdx.x-1; /* block 0 handles constant term */

   if(term!=CONSTANT_TERM) {
      black_box_key_set_bitpos(key,term);
   }
   
   for(iter=0;iter<nr_iter;iter++) {
      black_box_id2iv(iv,smemI,nr_idx,threadID);
      sum^=black_box_query(key,iv,NR_OUTPUT_BITS);

      threadID+=blockDim.x; //threadID+=(# threads/block)
   }
   /* write result to shared memory */
   reduce_csh[threadIdx.x]=sum;
   __syncthreads();

   /* reduce results of threads in same block */
   for(i=blockDim.x/2;i>0;i/=2) {
      if(threadIdx.x<i) {
         reduce_csh[threadIdx.x]^=reduce_csh[threadIdx.x+i];
      }
      __syncthreads();
   }

   /* write result from first thread thread */
   if(threadIdx.x==0) {
      out[blockIdx.x]=reduce_csh[0];
   }
}

CUT_THREADPROC do_reconstruct_linsuperpoly_GPU_h(CA_work_reconstruct *w_mterm) {
   int i;
   cudaDeviceProp deviceProp;
   u32 *idx; int nr_idx;
   size_t out_size,smem_size;
   u32 *out_d,*out_h,*I_d;

   cudaGetDeviceProperties(&deviceProp, w_mterm->pDev);
   cudaSetDevice(w_mterm->pDev);

   debug("Using physical device %d: %s\n",w_mterm->pDev,deviceProp.name);

   nr_idx=(*(w_mterm->I)).nr_idx;
   idx=(*(w_mterm->I)).idx;

   debug("I={"); for(i=0;i<nr_idx;i++) { debug("%2d, ",idx[i]); } debug("}\n");

   /* allocate output buffer -> one output per block */
   out_size=(w_mterm->nBlocks)*sizeof(u32);
   cutilSafeCall(cudaHostAlloc((void**)&(out_h),out_size,
                                                      cudaHostAllocPortable));
   cutilSafeCall(cudaMalloc((void**)&(out_d),out_size));

   /* allocate and copy maxterm */
   cutilSafeCall(cudaMalloc((void**)&(I_d),nr_idx*sizeof(u32)));
   cutilSafeCall(cudaMemcpy(I_d,idx,nr_idx*sizeof(u32),
                                                      cudaMemcpyHostToDevice));
   /* allocate dynamic shared memory size */
   smem_size=(w_mterm->nThreads)*sizeof(u32);

   debug("Device %2d: %3d blocks : %3d threads : %3d iterations...\n",
    w_mterm->devId, w_mterm->nBlocks, w_mterm->nThreads, w_mterm->nIterations);

   __kernel_reconstruct_linsuperpoly_h<<<w_mterm->nBlocks,
                                        w_mterm->nThreads,
                                        smem_size>>> (w_mterm->devId,
                                                      w_mterm->nBlocks,
                                                      I_d,
                                                      nr_idx,
                                                      out_d,
                                                      w_mterm->data.hcube.term,
                                                      w_mterm->nIterations );

   cutilCheckMsg("Kernel execution failed");
   cudaThreadSynchronize();

   /* copy output from device to host */
   cutilSafeCall(cudaMemcpy(out_h,out_d,out_size,cudaMemcpyDeviceToHost));

   /* reduce from all thread blocks */
   w_mterm->data.hcube.partial_sum=0;
   for(i=0;i<w_mterm->nBlocks;i++) {
      w_mterm->data.hcube.partial_sum^=out_h[i];
   }

   debug("Device %2d: output=%08x\n",w_mterm->devId,
         w_mterm->data.hcube.partial_sum);

   /* free memory */
   cutilSafeCall(cudaFree(I_d));
   cutilSafeCall(cudaFree(out_d));
   cutilSafeCall(cudaFreeHost(out_h));


   CUT_THREADEND;
}

CUT_THREADPROC do_reconstruct_linsuperpoly_GPU_m(CA_work_reconstruct *w_mterm) {
   int i;
   cudaDeviceProp deviceProp;
   u32 *idx; int nr_idx;
   size_t out_size,smem_size;
   u32 *out_d,*out_h,*I_d;

   cudaGetDeviceProperties(&deviceProp, w_mterm->pDev);
   cudaSetDevice(w_mterm->pDev);

   debug("Using physical device %d: %s\n",w_mterm->pDev,deviceProp.name);

   nr_idx=(*(w_mterm->I)).nr_idx;
   idx=(*(w_mterm->I)).idx;

   debug("I={"); for(i=0;i<nr_idx;i++) { debug("%2d, ",idx[i]); } debug("}\n");

   /* allocate output buffer -> one output per block */
   out_size=(w_mterm->nBlocks)*sizeof(u32);
   cutilSafeCall(cudaHostAlloc((void**)&(out_h),out_size,
                                                      cudaHostAllocPortable));
   cutilSafeCall(cudaMalloc((void**)&(out_d),out_size));

   /* allocate and copy maxterm */
   cutilSafeCall(cudaMalloc((void**)&(I_d),nr_idx*sizeof(u32)));
   cutilSafeCall(cudaMemcpy(I_d,idx,nr_idx*sizeof(u32),
                                                      cudaMemcpyHostToDevice));
   /* allocate dynamic shared memory size */
   smem_size=(w_mterm->nThreads)*sizeof(u32)+(nr_idx)*sizeof(u16);

   debug("Device %2d: %3d blocks : %3d threads : %3d iterations...\n",
    w_mterm->devId, w_mterm->nBlocks, w_mterm->nThreads, w_mterm->nIterations);

   __kernel_reconstruct_linsuperpoly_m<<<w_mterm->nBlocks,w_mterm->nThreads,
                               smem_size>>> (w_mterm->devId,
                                              w_mterm->data.mcube.blkOffset,
                                              I_d,
                                              nr_idx,
                                              out_d,
                                              w_mterm->nIterations );

   cutilCheckMsg("Kernel execution failed");
   cudaThreadSynchronize();

   /* copy output from device to host */
   cutilSafeCall(cudaMemcpy(out_h,out_d,out_size,cudaMemcpyDeviceToHost));

   /* reduce from all thread blocks */

   for(i=0;i<w_mterm->nBlocks;i++) {
   w_mterm->data.mcube.S_I->idx[
      w_mterm->devId*(w_mterm->data.mcube.blkOffset)+i]=out_h[i];
      debug("Device %2d: %2d: output=%08x\n",w_mterm->devId,i,out_h[i]);
   }


   /* free memory */
   cutilSafeCall(cudaFree(I_d));
   cutilSafeCall(cudaFree(out_d));
   cutilSafeCall(cudaFreeHost(out_h));


   CUT_THREADEND;
}

u32 do_reconstruct_linsuperpoly_CPU(u32 *I,int nr_idx, int term) {

   int i;
   u64 count=0;
   u32 sum=0,output=0;
   black_box_def_state(key,iv);


   debug("I={"); for(i=0;i<nr_idx;i++) { debug("%2d, ",I[i]); } debug("}\n");

//   printf("trying to reconstruct term %d..\n",term+1);

   if(term!=CONSTANT_TERM) {
      black_box_key_set_bitpos(key,term);
   }
   for(count=0;count<(1<<nr_idx);count++) {
      black_box_id2iv(iv,I,nr_idx,count);
      output=black_box_query(key,iv,NR_OUTPUT_BITS);
      sum^=output;
   }
 //  printf("final sum=%d\n\n",sum);
   return sum;
}

/* - find maxterm -------------------------------------------------------- */

int do_test_maxterm_CPU(CA_ext_term *test_term) {
   int t,i,nr_idx;
   CA_term *tI;
   u32 *I;
   u32 p0,p1,p2,p12;
   u32 failed_blr,nconst_poly; /* masks for the failed blr/nonconstant polys*/
   u64 count;
   
#if defined(CA_VERBOSE_STAT)
   u32 nr_ones[NR_OUTPUT_BITS], nr_aff[NR_OUTPUT_BITS];
#endif

   black_box_def_iv(iv);     
   black_box_def_key(key1);  
   black_box_def_key(key2);  
   black_box_def_key(key12); 

   tI=&test_term->I;

   I=tI->idx; /* easier to work with I */

   nr_idx=tI->nr_idx;



   debug("I(%02d)={",nr_idx);
   for(i=0;i<nr_idx;i++) { debug("%2d, ",I[i]); }
   debug("}\n");

   /* clear counters */
   failed_blr=nconst_poly=0;

#if defined(CA_VERBOSE_STAT)
   for(i=0;i<NR_OUTPUT_BITS;i++) { nr_ones[i]=nr_aff[i]=0; }
#endif


   black_box_clear_iv(iv);
   black_box_clear_key(key1);
   black_box_clear_key(key2);
   black_box_clear_key(key12);



   /* calculate p(0) */
   /* black_box_clear_key(key1); already clear */

   /* cube sum */
   p0=0;
   for(count=0;count<(1<<nr_idx);count++) {
      black_box_id2iv(iv,I,nr_idx,count);
      p0^=black_box_query(key1,iv,NR_OUTPUT_BITS);
   }

#if defined(CA_VERBOSE_STAT)
   for(i=0;i<NR_OUTPUT_BITS;i++) { if(p0&(1<<i))  { nr_ones[i]++; } }
#endif

   for(t=0;t<CA_NR_TESTS;t++) {
      black_box_key_random(key1,xsr_rand_u32);
      black_box_key_random(key2,xsr_rand_u32);
      black_box_add_keys(key12,key1,key2); /* k0+=k1 */

      /* cube sums */
      p1=p2=p12=0;
      for(count=0;count<(1<<nr_idx);count++) {
         black_box_id2iv(iv,I,nr_idx,count);

         p1^=black_box_query(key1,iv,NR_OUTPUT_BITS);
         p2^=black_box_query(key2,iv,NR_OUTPUT_BITS);
         p12^=black_box_query(key12,iv,NR_OUTPUT_BITS);

      }

      /* bit in mask is set if the output is not linear
         i.e., p(0)+p(k1)+p(k2) != p(k1+k2)
       */
      failed_blr|=(p0^p1^p2^p12);

      /* bit in mask is set if the output is not constant */
      nconst_poly|=(p0^p1)|(p0^p2);

#if defined(CA_VERBOSE_STAT)
      for(i=0;i<NR_OUTPUT_BITS;i++) {
         if(p1&(1<<i))  { nr_ones[i]++; }
         if(p2&(1<<i))  { nr_ones[i]++; }
         if(!((p0^p1^p2^p12)&(1<<i))) {
            nr_aff[i]++;
         } 
      }
#endif
      if(! ((~(failed_blr))&OUTPUT_BIT_MASK)) {
         /* all the ANFs failed the BLR test, so */
         break;
      }
   }


   //-------------------------------------------

#if defined(CA_DEBUG)

   debug("number of tests=%3d\n",t);
   for(i=0;i<NR_OUTPUT_BITS;i++) {
#if !defined(CA_VERBOSE_STAT)
      if(((nconst_poly&(~failed_blr))&(1<<i))) 
#endif
      {
         debug("%3d: constant=%d, failed_blr=%d", i/*+NR_INIT_ROUNDS*/,
               ((nconst_poly&(1<<i))==0),((failed_blr&(1<<i))>0));
#if defined(CA_VERBOSE_STAT)
         debug(", nr_ones=%3d/%3d,  affine=%3d",
               nr_ones[i],1+CA_NR_TESTS*2,nr_aff[i]);
#endif 
         debug(", p(0)=%d\n",((p0&(1<<i))>0));
      }
   }
#endif

   test_term->out_bits=
      (nconst_poly&(~failed_blr))&OUTPUT_BIT_MASK;

   return 0;
}

int do_find_maxterm_CPU(CA_ext_term *term,int min_dim,int max_dim, int pub_size) {
   int t,i,nr_idx;
   CA_term *tI;
   u32 *I;
   u32 p0,p1,p2,p12;
   u32 failed_blr,nconst_poly; /* masks for the failed blr/nonconstant polys*/
   u32 nr_tries;
   u64 count;
#if defined(CA_VERBOSE_STAT)
   u32 nr_ones[NR_OUTPUT_BITS], nr_aff[NR_OUTPUT_BITS];
#endif

   black_box_def_iv(iv);     
   black_box_def_key(key1);  
   black_box_def_key(key2);  
   black_box_def_key(key12); 

   tI=&term->I;//&w_info->data.find_maxterm.I;

   I=tI->idx; /* easier to work with I */

   /* start aglrothithm with a fresh term */
   nr_idx=rand_int(min_dim,max_dim); /* choose a random (bounded) size */
   array_rnd_fill(I,nr_idx,0,pub_size,rand_int);


   for(nr_tries=0;nr_tries<CA_MAX_TRIES_PER_I;nr_tries++) {

      debug("I(%02d)={",nr_idx);
      for(i=0;i<nr_idx;i++) { debug("%2d, ",I[i]); }
      debug("}\n");

      /* clear counters */
      failed_blr=nconst_poly=0;

#if defined(CA_VERBOSE_STAT)
      for(i=0;i<NR_OUTPUT_BITS;i++) { nr_ones[i]=nr_aff[i]=0; }
#endif


      black_box_clear_iv(iv);
      black_box_clear_key(key1);
      black_box_clear_key(key2);
      black_box_clear_key(key12);



      /* calculate p(0) */
      /* black_box_clear_key(key1); already clear */

      /* cube sum */
      p0=0;
      for(count=0;count<(1<<nr_idx);count++) {
         black_box_id2iv(iv,I,nr_idx,count);
         p0^=black_box_query(key1,iv,NR_OUTPUT_BITS);
      }

#if defined(CA_VERBOSE_STAT)
      for(i=0;i<NR_OUTPUT_BITS;i++) { if(p0&(1<<i))  { nr_ones[i]++; } }
#endif

      for(t=0;t<CA_NR_TESTS;t++) {
         black_box_key_random(key1,xsr_rand_u32);
         black_box_key_random(key2,xsr_rand_u32);
         black_box_add_keys(key12,key1,key2); /* k0+=k1 */

         /* cube sums */
         p1=p2=p12=0;
         for(count=0;count<(1<<nr_idx);count++) {
            black_box_id2iv(iv,I,nr_idx,count);

            p1^=black_box_query(key1,iv,NR_OUTPUT_BITS);
            p2^=black_box_query(key2,iv,NR_OUTPUT_BITS);
            p12^=black_box_query(key12,iv,NR_OUTPUT_BITS);

         }

         /* bit in mask is set if the output is not linear
            i.e., p(0)+p(k1)+p(k2) != p(k1+k2)
          */
         failed_blr|=(p0^p1^p2^p12);

         /* bit in mask is set if the output is not constant */
         nconst_poly|=(p0^p1)|(p0^p2);

#if defined(CA_VERBOSE_STAT)
         for(i=0;i<NR_OUTPUT_BITS;i++) {
            if(p1&(1<<i))  { nr_ones[i]++; }
            if(p2&(1<<i))  { nr_ones[i]++; }
            if(!((p0^p1^p2^p12)&(1<<i))) {
               nr_aff[i]++;
            } 
         }
#endif
         if(! ((~(failed_blr))&OUTPUT_BIT_MASK)) {
            /* all the ANFs failed the BLR test, so */
            break;
         }
      }

      if(t!=CA_NR_TESTS) {
         debug("nonlinear...\n");
         /* all the output ANFs are non-linear */
         if(nr_idx<tI->nr_idx)  {
            array_rnd_add_el(I,nr_idx,0,pub_size,rand_int);
         } else {
            term->out_bits=0; //w_info->data.find_maxterm.out_bits=0;
            tI->nr_idx=0;
            return -1; /* Try different I */
         }
      } else if(! ((nconst_poly)&OUTPUT_BIT_MASK)) {
         debug("constant...\n");
         /* all the output ANFs are constant */
         if(nr_idx>1) {
            array_rnd_rm_el(I,nr_idx,rand_int);
         } else {
            term->out_bits=0; //w_info->data.find_maxterm.out_bits=0;
            tI->nr_idx=0;
            return -1; /* Try different I */
         }
      } else {
            term->out_bits= //w_info->data.find_maxterm.out_bits=
            (nconst_poly&(~failed_blr))&OUTPUT_BIT_MASK;
           if(term->out_bits) {//w_info->data.find_maxterm.out_bits
            tI->nr_idx=nr_idx;
            break; /* Found a maxterm */
         } else {
            /* TODO: maybe compute the Hamming weight of
            the failed_blr and ~nconst_poly. if there
            are more non-linear ANFs then add a term
            else remove a term */
            tI->nr_idx=0;
            return -1; /* no maxterm found */
         }
      }
   }


   //-------------------------------------------

#if defined(CA_DEBUG)
   debug("number of tests=%3d\n",t);

   for(i=0;i<NR_OUTPUT_BITS;i++) {
      if(((nconst_poly&(~failed_blr))&(1<<i))) {
         debug("%3d: constant=%d, failed_blr=%d", i/*+NR_INIT_ROUNDS*/,
               ((nconst_poly&(1<<i))==0),((failed_blr&(1<<i))>0));
#if defined(CA_VERBOSE_STAT)
         debug(", nr_ones=%3d/%3d, affine=%3d",
               nr_ones[i],1+CA_NR_TESTS*2,nr_aff[i]);
#endif 
         debug(", p(0)=%d\n",((p0&(1<<i))>0));
      }
   }

#endif



   return 0;
}

int CA_test_black_box_CPU(char *fname) {
   int rc;

   printf("Testing black-box with vectors from \'%s\'...",fname);
   if((rc=black_box_test(fname))) {
      printf("FAILED!\n");
   } else {
      printf("PASSED!\n");
   }
   return rc;
}

/* see p.78 of H. S. Warren, Jr.'s "Hacker's Delight" */
__device__ int ilog2(i32 x) {
   int n;
   if(x==0) { return 0; } /* actually, -1, but should never get here! */
   n=1;
   if((x>>16)==0) { n+=16; x<<=16;}
   if((x>>24)==0) { n+=8;  x<<=8; }
   if((x>>28)==0) { n+=4;  x<<=4; }
   if((x>>30)==0) { n+=2;  x<<=2; }
   n-=(x>>31);
   return (31-n);
}

#if defined(CONSTANT_TEST_MAXTERM_GPU)
__constant__ u16 test_I[7][15]={
/*
{ 0,  1,  2,  7,  9, 11, 14, 16, 20, 25, 43, 60, 67},// 672+3: x_54
{ 1,  5, 14, 19, 22, 35, 37, 40, 51, 53, 56, 75, 78},// 672+4: 1+x67
{ 0,  7, 11, 14, 18, 21, 24, 26, 33, 36, 62, 65, 78},// 672+3: x_65
{ 4, 10, 13, 19, 24, 26, 41, 53, 56, 61, 64, 71, 73},//672+3: x_62
{ 4, 22, 25, 34, 36, 40, 42, 50, 53, 57, 61, 68, 78},//672+5:1 + x_61
{ 1,  3, 14, 20, 23, 40, 43, 46, 52, 53, 55, 57, 65}, //672:  x_57
{ 3, 10, 14, 16, 18, 20, 21, 32, 41, 53, 58, 72, 77},//672+1: x64
{ 4, 13, 15, 16, 19, 23, 27, 29, 42, 43, 52, 56, 79},//672+5: x_65
{ 2,  6,  8, 12, 15, 23, 33, 35, 50, 58, 64, 67, 79},//672+7: x_60
*/
{ 0, 12, 16, 18, 20, 24, 25, 26, 36, 37, 47, 49, 52, 55, 70},//672+11:x_57+x_59
{ 0,  4, 12, 16, 18, 20, 24, 25, 26, 36, 37, 47, 52, 55, 70},//672+8: x_57
{ 0,  2, 14, 19, 20, 22, 29, 30, 36, 53, 58, 64, 70, 72, 79},//672+2: x_55
{ 0, 16, 19, 21, 39, 42, 51, 58, 61, 63, 65, 66, 70, 73, 78},//672+7: x_59
{ 0,  2, 15, 19, 23, 27, 30, 35, 36, 45, 46, 50, 60, 65, 72},//672: x_65
{ 1,  6 , 9, 16, 17, 22, 37, 40, 46, 53, 56, 58, 60, 68, 71},//672+4: x_58
{ 0,  3,  4,  7, 15, 29, 32, 37, 40, 49, 54, 64, 70, 77, 78},//672+8: x_57
};
#endif

#define FIND_MAXTERM_FAIL (0x01<<16)
#define __pck_lo(a) ((a)&0x0000ffff)
#define __pck_hi(a) ((a)&0xffff0000)
__global__ static void __kernel_find_maxterm_m(u32 *seed,    
                                               i32 min_dim,
                                               i32 max_dim,   
                                               int pub_size,   
                                               u16 *glob_nr_idx,
                                               u16 *glob_I,     
                                               u32 *passANF) {  
/*
seed         [in] rng seed (per block)
min_dim      [in] min dimension of I 
max_dim      [in] max dimension of I 
pub_size     [in] #public variables 
glob_nr_idx [out] size of maxterms (per block) 
glob_I      [out] actual maxterms (per block) 
passANF     [out] the ANFs which are linear & non-constant*/

   u32* reduce_csh=(u32*) smem_cache;
   u32* pck_control_idx=(u32*) &smem_cache[blockDim.x];
   u32* new_seed=(u32*) &smem_cache[blockDim.x+1];
   u32* failed_blr=(u32*) &smem_cache[blockDim.x+2];
   u32* nconst_poly=(u32*) &smem_cache[blockDim.x+3];
   u16* smemI=(u16*) &smem_cache[blockDim.x+4]; /* reduce_csh uses 0->blockDim.x-1 */
   int t;
   int nr_idx; /* definitely <16-bits */
   u64 threadID;
   u32 log_blockDim=ilog2(blockDim.x);
   u32 nr_iter;
   u32 sum;
   u32 p0,p1,p2,p12;

   /* clear control bits */
   pck_control_idx[0]=0;

   /* define and seed the rng */
   inline_xsr_def_u32();
   inline_xsr_srand_u32(seed[blockIdx.x]);
   /* declare the keys and ivs for black box */
   black_box_def_key(key1);
   black_box_def_key(key2);
   black_box_def_iv(iv);


   nr_idx=inline_rand_int(min_dim,max_dim); /* choose a random (bounded) size*/
   if(threadIdx.x==0) {
      array_rnd_fill(smemI,nr_idx,0,pub_size,inline_rand_int);
      new_seed[0]=inline_xsr_rand_u32();
   }
   __syncthreads();
   inline_xsr_srand_u32(new_seed[0]); /* broadcast read new seed */

#if defined(CONSTANT_TEST_MAXTERM_GPU)
   /* lazy test approach, change to more CPU-like test*/
   nr_idx=max_dim;

   if(threadIdx.x<nr_idx) {
      /* if there are enough threads (there should be), read in parallel */
      smemI[threadIdx.x]=test_I[blockIdx.x][threadIdx.x];
   }
   __syncthreads();
#endif

   nr_iter=1<<(nr_idx-log_blockDim);


   /* given a key and iv do a cube sum, reduced to thread 0 of the block
      and save the result in pi */
#define cube_sum(pi,key,iv)                                             \
   {                                                                    \
      threadID=threadIdx.x;                                             \
      sum=0;                                                            \
      for(int iter=0;iter<nr_iter;iter++) {                             \
         black_box_id2iv(iv,smemI,nr_idx,threadID);                     \
         sum^=black_box_query(key,iv,NR_OUTPUT_BITS);                   \
                                                                        \
         threadID+=blockDim.x;                                          \
      }                                                                 \
      /* write result to shared memory */                               \
      reduce_csh[threadIdx.x]=sum;                                      \
      __syncthreads();                                                  \
                                                                        \
      /* reduce results of threads in same block */                     \
      for(int i=blockDim.x/2;i>0;i/=2) {                                \
         if(threadIdx.x<i) {                                            \
            reduce_csh[threadIdx.x]^=reduce_csh[threadIdx.x+i];         \
         }                                                              \
         __syncthreads();                                               \
      }                                                                 \
                                                                        \
      /* write result from first thread thread */                       \
      if(threadIdx.x==0) {                                              \
         pi=reduce_csh[0];                                              \
      }                                                                 \
   }

   for(int nr_tries=0;nr_tries<CA_MAX_TRIES_PER_I;nr_tries++) {

      /* clear counters */
      if(threadIdx.x==0) {
         nconst_poly[0]=failed_blr[0]=0;
      }
      __syncthreads();
      p0=p1=p2=p12=0;

      /* compute p0 */
      black_box_clear_key(key1);                      cube_sum(p0, key1, iv);

      for(t=0;t<CA_NR_TESTS;t++) {
         /* compute p1,p2,p12 */
         black_box_key_random(key1,inline_xsr_rand_u32); cube_sum(p1, key1, iv);
         black_box_key_random(key2,inline_xsr_rand_u32); cube_sum(p2, key2, iv);
         black_box_add_keys(key1,key1,key2);             cube_sum(p12,key1, iv);
         if(threadIdx.x==0) {
            /* bit in mask is set if the output is not linear
               i.e., p(0)+p(k1)+p(k2) != p(k1+k2)
             */
            failed_blr[0]|=(p0^p1^p2^p12);

            /* bit in mask is set if the output is not constant */
            nconst_poly[0]|=(p0^p1)|(p0^p2);

         }

         __syncthreads();

         if(!((~(failed_blr[0]))&OUTPUT_BIT_MASK)) {
            /* all the ANFs failed the BLR test, so */
            break;
         }

      }

      /* broke out of/finished the BLR tests */
      bool is_nonlin=(t!=CA_NR_TESTS);
      bool is_const=(!(nconst_poly[0]&OUTPUT_BIT_MASK));

      if(is_nonlin || is_const) { 
         if(threadIdx.x==0) {
            if( is_nonlin && (nr_idx<max_dim)) {
               /* all the output ANFs are non-linear, add an element */
               array_rnd_add_el(smemI,nr_idx,0,pub_size,inline_rand_int);
            } else if( is_const && (nr_idx>1)) {
               /* all the output ANFs are constant, remove term */
               array_rnd_rm_el(smemI,nr_idx,inline_rand_int);
            } else {
               /* try with a different I */
               pck_control_idx[0]|=FIND_MAXTERM_FAIL;
               glob_nr_idx[blockIdx.x]=0;
               passANF[blockIdx.x]=0;
            }
            if(!(pck_control_idx[0]&FIND_MAXTERM_FAIL)) {
               pck_control_idx[0]=__pck_hi(pck_control_idx[0])|__pck_lo(nr_idx);
               new_seed[0]=inline_xsr_rand_u32();
            }
         }

         __syncthreads();
         if(!(pck_control_idx[0]&FIND_MAXTERM_FAIL)) {
            /* update seed and nr_idx for all other threads */
            inline_xsr_srand_u32(new_seed[0]);
            nr_idx=__pck_lo(pck_control_idx[0]);
         } else {
            /* failed to find maxterm with this I, exit*/
            break;
         }

      } else {
         /* maybe found maxterm */
         if(threadIdx.x==0) {

            passANF[blockIdx.x]=(nconst_poly[0]&(~failed_blr[0]))&
                                                      OUTPUT_BIT_MASK;
            if(passANF[blockIdx.x]) {
               /* found a maxterm */
               glob_nr_idx[blockIdx.x]=nr_idx;
            } else {
               /* did not find a maxterm */
               glob_nr_idx[blockIdx.x]=0;
            }
         }
         __syncthreads();
         if(passANF[blockIdx.x]) {
            /* write found maxterm I to global mem */
            if(threadIdx.x<nr_idx) {
               /* write most in parallel */
               glob_I[blockIdx.x*max_dim+threadIdx.x]=smemI[threadIdx.x];
            }
            if(nr_idx>blockDim.x && threadIdx.x==0) {
               /* highly unlikely branch, but write the rest by thread 0 */
               for(int i=nr_idx-blockDim.x;i<nr_idx;i++) {
                  glob_I[blockIdx.x*max_dim+i]=smemI[i];
               }
            }
         }
         break;
      }

   }

//#undef cube_sum

}

__global__ static void __kernel_test_maxterm_m(u32 *seed,    
                                               i32 max_dim,   
                                               u16 *glob_nr_idx,
                                               u16 *glob_I,     
                                               u32 *passANF) {  
/*
seed         [in] rng seed (per block)
max_dim      [in] max dimension of I  (also the stride)
glob_nr_idx  [in] size of maxterms (per block) 
glob_I       [in] actual maxterms (per block) 
passANF     [out] the ANFs which are linear & non-constant*/

   u32* reduce_csh=(u32*) smem_cache;
   u32* failed_blr=(u32*) &smem_cache[blockDim.x+1];
   u16* smemI=(u16*) &smem_cache[blockDim.x+2]; /* reduce_csh uses 0->blockDim.x-1 */
   u32 nconst_poly;
   int t;
   int nr_idx; /* definitely <16-bits */
   u64 threadID;
   u32 log_blockDim=ilog2(blockDim.x);
   u32 nr_iter;
   u32 sum;
   u32 p0,p1,p2,p12;

   /* define and seed the rng */
   inline_xsr_def_u32();
   inline_xsr_srand_u32(seed[blockIdx.x]);
   /* declare the keys and ivs for black box */
   black_box_def_key(key1);
   black_box_def_key(key2);
   black_box_def_iv(iv);


   nr_idx=glob_nr_idx[blockIdx.x];

   /* read maxterm I from global mem */
   if(threadIdx.x<nr_idx) {
      /* read most in parallel */
      smemI[threadIdx.x]=glob_I[blockIdx.x*max_dim+threadIdx.x];
   }
   if(nr_idx>blockDim.x && threadIdx.x==0) {
      /* highly unlikely branch, but read the rest by thread 0 */
      for(int i=nr_idx-blockDim.x;i<nr_idx;i++) {
         smemI[i]=glob_I[blockIdx.x*max_dim+i];
      }
   }
   __syncthreads();

   nr_iter=1<<(nr_idx-log_blockDim);


   /* clear counters */
   if(threadIdx.x==0) { failed_blr[0]=0; } __syncthreads();
   nconst_poly=0;
   p0=p1=p2=p12=0;

   /* compute p0 */
   black_box_clear_key(key1);                      cube_sum(p0, key1, iv);

   for(t=0;t<CA_NR_TESTS;t++) {
      /* compute p1,p2,p12 */
      black_box_key_random(key1,inline_xsr_rand_u32); cube_sum(p1, key1, iv);
      black_box_key_random(key2,inline_xsr_rand_u32); cube_sum(p2, key2, iv);
      black_box_add_keys(key1,key1,key2);             cube_sum(p12,key1, iv);

      if(threadIdx.x==0) {
         /* bit in mask is set if the output is not linear
            i.e., p(0)+p(k1)+p(k2) != p(k1+k2)
          */
         failed_blr[0]|=(p0^p1^p2^p12);

         /* bit in mask is set if the output is not constant */
         nconst_poly|=(p0^p1)|(p0^p2);

      }
      __syncthreads();

      if(!((~(failed_blr[0]))&OUTPUT_BIT_MASK)) {
         /* all the ANFs failed the BLR test, so */
         break;
      }

   }

   if(threadIdx.x==0) {
      passANF[blockIdx.x]=(nconst_poly&(~failed_blr[0]))&OUTPUT_BIT_MASK;
   }


#undef cube_sum

}


CUT_THREADPROC do_find_maxterm_GPU_m(CA_work_search *w_sterm) {
   cudaDeviceProp deviceProp;
   size_t seed_size,smem_size,nr_idx_size,I_size,output_bits_size;
   u32 *seed_h,*seed_d;
   u16 *nr_idx_h,*nr_idx_d;
   u16 *I_h,*I_d;
   u32 *output_bits_h,*output_bits_d;


   /* set the device */
   cudaGetDeviceProperties(&deviceProp, w_sterm->pDev);
   cudaSetDevice(w_sterm->pDev);

   debug("Using physical device %d: %s\n",w_sterm->pDev,deviceProp.name);
   debug("%3d,%3d\n",w_sterm->nBlocks,w_sterm->nThreads);

   /* shared memory:
      cache for the reduce state=nr_threads*sizeof(u32)
      cache for storing I=nr_idx*sizeof(u16)
      u32 pck_control_idx=[control|nr_idx],new_seed,failed_blr,nconst_poly
    */
   smem_size=(w_sterm->nThreads)*sizeof(u32)+
             (w_sterm->max_dim)*sizeof(u16)+4*sizeof(u32);
   seed_size=(w_sterm->nBlocks)*sizeof(u32);
   nr_idx_size=(w_sterm->nBlocks)*sizeof(u16);
   I_size=(w_sterm->nBlocks*w_sterm->max_dim)*sizeof(u16);
   output_bits_size=(w_sterm->nBlocks)*sizeof(u32);
   
   /* input: seed (per block) */
   cutilSafeCall(cudaHostAlloc((void**)&(seed_h),seed_size,cudaHostAllocPortable));
   cutilSafeCall(cudaMalloc((void**)&(seed_d),seed_size));

   /* output: nr_idx is the size of the maxterms (if found, else 0) */
   cutilSafeCall(cudaHostAlloc((void**)&(nr_idx_h),nr_idx_size,cudaHostAllocPortable));
   cutilSafeCall(cudaMalloc((void**)&(nr_idx_d),nr_idx_size));

   /* output: actual maxterms */
   cutilSafeCall(cudaHostAlloc((void**)&(I_h),I_size,cudaHostAllocPortable));
   cutilSafeCall(cudaMalloc((void**)&(I_d),I_size));

   /* output: the output bits for which the tests passed */
   cutilSafeCall(cudaHostAlloc((void**)&(output_bits_h),output_bits_size,cudaHostAllocPortable));
   cutilSafeCall(cudaMalloc((void**)&(output_bits_d),output_bits_size));

   {
      /* read a random seed */
      FILE *fp;
      if(!(fp = fopen("/dev/urandom", "r"))) {
         fprintf(stderr,"Failed to open /dev/urandom");
         exit(-1);
      }
      fread(seed_h,sizeof(u32),w_sterm->nBlocks,fp);
      fclose(fp);
   }

   cutilSafeCall(cudaMemcpy(seed_d,seed_h,seed_size,cudaMemcpyHostToDevice));
   __kernel_find_maxterm_m<<<w_sterm->nBlocks,
                             w_sterm->nThreads,
                             smem_size>>>(seed_d,
                                          w_sterm->min_dim,
                                          w_sterm->max_dim,
                                          w_sterm->pub_size,
                                          nr_idx_d,
                                          I_d,
                                          output_bits_d);
   cutilCheckMsg("Kernel execution failed");
   cudaThreadSynchronize();

   /* copy dimensions, terms, and output bit masks back */
   cutilSafeCall(cudaMemcpy(nr_idx_h,nr_idx_d,nr_idx_size,cudaMemcpyDeviceToHost));
   cutilSafeCall(cudaMemcpy(I_h,I_d,I_size,cudaMemcpyDeviceToHost));
   cutilSafeCall(cudaMemcpy(output_bits_h,output_bits_d,
                                 output_bits_size,cudaMemcpyDeviceToHost));

   for(int i=0;i<w_sterm->nBlocks;i++) {
      debug("device %3d block %3d -> nr_idx=%2d, output_bits=%08x ",w_sterm->devId,i, nr_idx_h[i], output_bits_h[i]);
      debug("{ ");
      if(nr_idx_h[i]) {
         for(int j=0;j<nr_idx_h[i];j++) {
            debug("%2d, ",I_h[i*w_sterm->max_dim+j]);
         }
      }
      debug("}\n");
   }

   for(int i=0;i<w_sterm->nBlocks;i++) {
      w_sterm->search_terms[i].out_bits=output_bits_h[i];
      w_sterm->search_terms[i].I.nr_idx=nr_idx_h[i];
      for(int j=0;j<nr_idx_h[i];j++) {
      w_sterm->search_terms[i].I.idx[j]=I_h[i*w_sterm->max_dim+j];
      }
   }

   cutilSafeCall(cudaFree(seed_d));
   cutilSafeCall(cudaFreeHost(seed_h));

   cutilSafeCall(cudaFree(I_d));
   cutilSafeCall(cudaFreeHost(I_h));

   cutilSafeCall(cudaFree(nr_idx_d));
   cutilSafeCall(cudaFreeHost(nr_idx_h));

   cutilSafeCall(cudaFree(output_bits_d));
   cutilSafeCall(cudaFreeHost(output_bits_h));


   CUT_THREADEND;
}

CUT_THREADPROC do_test_maxterm_GPU_m(CA_work_search *w_sterm) {
   cudaDeviceProp deviceProp;
   size_t seed_size,smem_size,nr_idx_size,I_size,output_bits_size;
   u32 *seed_h,*seed_d;
   u16 *nr_idx_h,*nr_idx_d;
   u16 *I_h,*I_d;
   u32 *output_bits_h,*output_bits_d;


   /* set the device */
   cudaGetDeviceProperties(&deviceProp, w_sterm->pDev);
   cudaSetDevice(w_sterm->pDev);

   debug("Using physical device %d: %s\n",w_sterm->pDev,deviceProp.name);
   debug("%3d,%3d\n",w_sterm->nBlocks,w_sterm->nThreads);

   /* shared memory:
      cache for the reduce state=nr_threads*sizeof(u32)
      cache for storing I=nr_idx*sizeof(u16)
      u32 failed_blr
    */
   smem_size=(w_sterm->nThreads)*sizeof(u32)+
             (w_sterm->max_dim)*sizeof(u16)+1*sizeof(u32);
   seed_size=(w_sterm->nBlocks)*sizeof(u32);
   nr_idx_size=(w_sterm->nBlocks)*sizeof(u16);
   I_size=(w_sterm->nBlocks*w_sterm->max_dim)*sizeof(u16);
   output_bits_size=(w_sterm->nBlocks)*sizeof(u32);
   
   /* input: seed (per block) */
   cutilSafeCall(cudaHostAlloc((void**)&(seed_h),seed_size,cudaHostAllocPortable));
   cutilSafeCall(cudaMalloc((void**)&(seed_d),seed_size));

   /* output: nr_idx is the size of the maxterms (if found, else 0) */
   cutilSafeCall(cudaHostAlloc((void**)&(nr_idx_h),nr_idx_size,cudaHostAllocPortable));
   cutilSafeCall(cudaMalloc((void**)&(nr_idx_d),nr_idx_size));

   /* output: actual maxterms */
   cutilSafeCall(cudaHostAlloc((void**)&(I_h),I_size,cudaHostAllocPortable));
   cutilSafeCall(cudaMalloc((void**)&(I_d),I_size));

   /* output: the output bits for which the tests passed */
   cutilSafeCall(cudaHostAlloc((void**)&(output_bits_h),output_bits_size,cudaHostAllocPortable));
   cutilSafeCall(cudaMalloc((void**)&(output_bits_d),output_bits_size));

   {
      /* read a random seed */
      FILE *fp;
      if(!(fp = fopen("/dev/urandom", "r"))) {
         fprintf(stderr,"Failed to open /dev/urandom");
         exit(-1);
      }
      fread(seed_h,sizeof(u32),w_sterm->nBlocks,fp);
      fclose(fp);
   }


   for(int i=0;i<w_sterm->nBlocks;i++) {
      nr_idx_h[i]=w_sterm->search_terms[i].I.nr_idx;
      for(int j=0;j<nr_idx_h[i];j++) {
         I_h[i*w_sterm->max_dim+j]=w_sterm->search_terms[i].I.idx[j];
      }
   }

   cutilSafeCall(cudaMemcpy(seed_d,seed_h,seed_size,cudaMemcpyHostToDevice));
   cutilSafeCall(cudaMemcpy(I_d,I_h,I_size,cudaMemcpyHostToDevice));
   cutilSafeCall(cudaMemcpy(nr_idx_d,nr_idx_h,nr_idx_size,
                                          cudaMemcpyHostToDevice));

   __kernel_test_maxterm_m<<<w_sterm->nBlocks,
                             w_sterm->nThreads,
                             smem_size>>>(seed_d,
                                          w_sterm->max_dim,
                                          nr_idx_d,
                                          I_d,
                                          output_bits_d);
   cutilCheckMsg("Kernel execution failed");
   cudaThreadSynchronize();

   /* copy dimensions, terms, and output bit masks back */
   cutilSafeCall(cudaMemcpy(output_bits_h,output_bits_d,
                                 output_bits_size,cudaMemcpyDeviceToHost));

   for(int i=0;i<w_sterm->nBlocks;i++) {
      debug("device %3d block %3d -> nr_idx=%2d, output_bits=%08x ",w_sterm->devId,i, nr_idx_h[i], output_bits_h[i]);
      debug("{ ");
      if(nr_idx_h[i]) {
         for(int j=0;j<nr_idx_h[i];j++) {
            debug("%2d, ",I_h[i*w_sterm->max_dim+j]);
         }
      }
      debug("}\n");
   }

   for(int i=0;i<w_sterm->nBlocks;i++) {
      w_sterm->search_terms[i].out_bits=output_bits_h[i];
   }

   cutilSafeCall(cudaFree(seed_d));
   cutilSafeCall(cudaFreeHost(seed_h));

   cutilSafeCall(cudaFree(I_d));
   cutilSafeCall(cudaFreeHost(I_h));

   cutilSafeCall(cudaFree(nr_idx_d));
   cutilSafeCall(cudaFreeHost(nr_idx_h));

   cutilSafeCall(cudaFree(output_bits_d));
   cutilSafeCall(cudaFreeHost(output_bits_h));


   CUT_THREADEND;
}
