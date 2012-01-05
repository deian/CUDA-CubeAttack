#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include <stdarg.h>

#include <multithreading.h>
#include <cutil_inline.h>
#include <cuda_runtime_api.h>

#include "cube_attack.h"
/*TODO: memory leak-> free work_info */

void CA_alloc_idx(CA_term *term,int nr_idx) {
   if(!(term->idx=(u32*)malloc(sizeof(u32)*nr_idx))) {
      fprintf(stderr,"CA_alloc_idx: malloc idx failed "
                                                "with %s\n",strerror(errno));
      exit(-1);
   }
   term->nr_idx=nr_idx;
}

void CA_free_idx(CA_term *term) {
   free(term->idx);
}

void CA_alloc_search_terms(CA_ext_term **term,int nr_terms,int max_dim) {
   CA_ext_term *t;
   int i;

   if(!(t=(CA_ext_term *)malloc(sizeof(CA_ext_term)*nr_terms))) {
      fprintf(stderr,"CA_alloc_search_terms: malloc term failed "
                                                "with %s\n",strerror(errno));
      exit(-1);
   }

   for(i=0;i<nr_terms;i++) {
      CA_alloc_idx(&t[i].I,max_dim);
      t[i].out_bits=0;
   }
   *term=t;
}

void CA_free_search_terms(CA_ext_term *term,int nr_terms) {
   int i;
   for(i=0;i<nr_terms;i++) {
      CA_free_idx(&term[i].I);
   }
   free(term);
}

void CA_alloc_test_terms(CA_work_info *w_info, int nr_terms) {
   if(!(w_info->data.test_maxterm.test_terms=
                        (CA_ext_term *)malloc(sizeof(CA_ext_term)*nr_terms))) {
      fprintf(stderr,"CA_alloc_search_terms: malloc term failed "
                                                "with %s\n",strerror(errno));
      exit(-1);
   }

   w_info->data.test_maxterm.min_dim=1000;
   w_info->data.test_maxterm.max_dim=0;
   w_info->data.test_maxterm.cnt_terms=0;
   w_info->data.test_maxterm.nr_terms=nr_terms;

}

void CA_free_test_terms(CA_work_info *w_info) {
   int i;
   for(i=0;i<w_info->data.test_maxterm.cnt_terms;i++) {
      CA_free_idx(&w_info->data.test_maxterm.test_terms[i].I);
   }
   free(w_info->data.test_maxterm.test_terms);
}

int __limit_physical_devs_count=0;
int *__limit_physical_devs=NULL;

int CA_limit_dev(int nr_dev, ...) {
      va_list ap;
      int i;
      int deviceCount;
      cudaGetDeviceCount(&deviceCount);
      if(nr_dev>deviceCount) {
         fprintf(stderr,"CA_limit_dev: number of devices is "
               "invalid (>%d)", deviceCount);
         return -1;
      }
      __limit_physical_devs_count=nr_dev;
      if(!(__limit_physical_devs=(int*)malloc(sizeof(int)*nr_dev))) {
         fprintf(stderr,"CA_limit_dev: malloc devs count failed "
               "with %s\n", strerror(errno));
         exit(-1);
      }
      va_start(ap,nr_dev);
      for(i=0;i<nr_dev;i++) {
         __limit_physical_devs[i]=va_arg(ap,int);
         if(__limit_physical_devs[i]>=deviceCount) {
            fprintf(stderr,"CA_limit_dev: device %d number (%d) is "
                  "invalid (>%d)", i, __limit_physical_devs[i], deviceCount);
         }
      }
      va_end(ap);
      return 0;
}


int CA_alloc_v_devices(CA_devs *v_devs,int strict_physical) {
   int i;
   int deviceCount;
   int limit_flag=0;

   cudaGetDeviceCount(&deviceCount);
   if(deviceCount == 0) {
      v_devs->devID=NULL;
      return -1;
   }
   if(__limit_physical_devs_count && __limit_physical_devs) {
      limit_flag=1;
      deviceCount=__limit_physical_devs_count;
   }

   v_devs->nr_pdev=deviceCount;
   if(!strict_physical) {
      /* number of virtual devices is closest power of 2 */
      v_devs->nr_vdev=1<<(int)(ceil(log2(deviceCount+1)));
   } else {
      /* strictly limit the physical devices */
      v_devs->nr_vdev=v_devs->nr_pdev;
   }

   if(!(v_devs->devID=(int*)malloc(sizeof(int)*v_devs->nr_vdev))) {
      fprintf(stderr,"CA_alloc_v_devices: malloc devID failed "
                                                "with %s\n",strerror(errno));
      exit(-1);
   }

   for(i=0;i<v_devs->nr_vdev;i++) {
      /* assign physical devices in round-robin fashion */
      if(!limit_flag) {
         v_devs->devID[i]=i%(v_devs->nr_pdev);
      } else {
         v_devs->devID[i]=__limit_physical_devs[i%(v_devs->nr_pdev)];
      }
   }

   return 0;
}

void CA_free_v_devices(CA_devs *v_devs) {
   if(v_devs->devID) {
      free(v_devs->devID);
   }
}

#define min(a,b)\
   ( ( (a)>=(b) ) ? (b):(a) )

#define max(a,b)\
   ( ( (a)>=(b) ) ? (a):(b) )

/* reconstruct superpoly for a high-dimensional cube */
void CA_reconstruct_linsuperpoly_GPU_h(CA_work_info *w_info, CA_devs *v_devs) {
   int i,term;
   u32 const_term=0;
   int tmp,D,B,T,J;
   CA_work_reconstruct *work_items;
   CUTThread *thread_IDs;

   if(!(work_items=(CA_work_reconstruct*)
            malloc(sizeof(CA_work_reconstruct)*v_devs->nr_vdev))) {
      fprintf(stderr,"CA_reconstruct_linsuperpoly_GPU: malloc work_items failed"
                                                " with %s\n",strerror(errno));
      exit(-1);
   }
   if(!(thread_IDs=(CUTThread*)malloc(sizeof(CUTThread)*v_devs->nr_vdev))) {
      fprintf(stderr,"CA_reconstruct_linsuperpoly_GPU: malloc thread_IDs failed"
                                                " with %s\n",strerror(errno));
      exit(-1);
   }

   D=(int)(log2(v_devs->nr_vdev));
   tmp=w_info->data.reconstruct_linsuperpoly.I.nr_idx-D;/*|I|-D p eval/device*/
   T=min(8,tmp); /* max 2**8=256 threads/block */
   tmp-=T;
   B=min(9,max(tmp,0)); /* max 2**9=512 blocks/device */
   tmp-=B;
   J=max(tmp,0);        /* at least 1 iteration/thread */

   for(i=0;i<v_devs->nr_vdev;i++) {
      work_items[i].pDev=v_devs->devID[i];
      work_items[i].devId=i;
      work_items[i].nDev=1<<D;
      work_items[i].nBlocks=1<<B;
      work_items[i].nThreads=1<<T;
      work_items[i].nIterations=1<<J;
      work_items[i].I=&w_info->data.reconstruct_linsuperpoly.I;
   }

   /* calculate constant term */
   for(i=0;i<v_devs->nr_vdev;i++) {
      work_items[i].data.hcube.term=CONSTANT_TERM;
      work_items[i].data.hcube.partial_sum=0;
      thread_IDs[i]=cutStartThread((CUT_THREADROUTINE)
               do_reconstruct_linsuperpoly_GPU_h,(void *)(&work_items[i]));
   }
   cutWaitForThreads(thread_IDs, v_devs->nr_vdev);
   /* reduce from all devices */
   for(i=0;i<v_devs->nr_vdev;i++) {
      const_term^=work_items[i].data.hcube.partial_sum;
   }
   debug("constant_term = %08x\n",const_term);
   w_info->data.reconstruct_linsuperpoly.S_I.idx[
      w_info->data.reconstruct_linsuperpoly.S_I.nr_idx-1]=const_term;

   /* Can of course, start from -1 (CONSTANT_TERM) */
   for(term=0;term<w_info->priv_size;term++) {
      u32 sum=0;
      /* calculate term */
      for(i=0;i<v_devs->nr_vdev;i++) {
         work_items[i].data.hcube.term=term;
         work_items[i].data.hcube.partial_sum=0;
         thread_IDs[i]=cutStartThread((CUT_THREADROUTINE)
               do_reconstruct_linsuperpoly_GPU_h,(void *)(&work_items[i]));
      }
      cutWaitForThreads(thread_IDs, v_devs->nr_vdev);
      /* reduce from all devices */
      for(i=0;i<v_devs->nr_vdev;i++) {
         sum^=work_items[i].data.hcube.partial_sum;
      }
      debug("x_%02d = %08x\n",term,(sum^const_term)/*&0x1*/);
      w_info->data.reconstruct_linsuperpoly.S_I.idx[term]=sum^const_term;
   }

   free(work_items);
   free(thread_IDs);
}

/* reconstruct superpoly for a medium-dimensional cube */
void CA_reconstruct_linsuperpoly_GPU_m(CA_work_info *w_info, CA_devs *v_devs) {
   int i;
   u32 const_term=0, *S_I;
   int tmp,D,B,T,J,K;
   CA_work_reconstruct *work_items;
   CUTThread *thread_IDs;

   if(!(work_items=(CA_work_reconstruct*)
            malloc(sizeof(CA_work_reconstruct)*v_devs->nr_vdev))) {
      fprintf(stderr,"CA_reconstruct_linsuperpoly_GPU: malloc work_items failed "
                                                "with %s\n",strerror(errno));
      exit(-1);
   }
   if(!(thread_IDs=(CUTThread*)malloc(sizeof(CUTThread)*v_devs->nr_vdev))) {
      fprintf(stderr,"CA_reconstruct_linsuperpoly_GPU: malloc thread_IDs failed "
                                                "with %s\n",strerror(errno));
      exit(-1);
   }


   D=(int)(log2(v_devs->nr_vdev));
   K=w_info->priv_size;
   B=(K+1)>>D;          /* (K+1)/2^D blocks per device */
   /* each block evaluates a full cube sum |I|: */
   T=min(8,w_info->data.reconstruct_linsuperpoly.I.nr_idx);
   tmp=w_info->data.reconstruct_linsuperpoly.I.nr_idx-T;
   J=max(tmp,0);

   for(i=0;i<v_devs->nr_vdev;i++) {
      /* assign work */
      work_items[i].pDev=v_devs->devID[i];
      work_items[i].devId=i;
      work_items[i].nDev=1<<D;
      work_items[i].nBlocks=B; /* NOTE: not 2^B */
      work_items[i].nThreads=1<<T;
      work_items[i].nIterations=1<<J;
      work_items[i].I=&w_info->data.reconstruct_linsuperpoly.I;
      work_items[i].data.mcube.blkOffset=B;
      work_items[i].data.mcube.S_I=&w_info->data.reconstruct_linsuperpoly.S_I;
   }
   /* handle the odd case; final device has more blocks: */
   work_items[v_devs->nr_vdev-1].nBlocks=B+(K+1)-(B<<D);

   for(i=0;i<v_devs->nr_vdev;i++) {
      /* spawn threads */
      thread_IDs[i]=cutStartThread((CUT_THREADROUTINE)
            do_reconstruct_linsuperpoly_GPU_m,(void *)(&work_items[i]));
   }
   cutWaitForThreads(thread_IDs, v_devs->nr_vdev);

   /* for simplicity the constant term is in pos 0, 
      xor the constant term and move it to the last position: */
   
   S_I=w_info->data.reconstruct_linsuperpoly.S_I.idx;
   const_term=S_I[0];
   for(i=0;i<K;i++) {
      S_I[i]=S_I[i+1]^const_term;
      debug("x_%02d = %08x\n",i,S_I[i]/*&0x1*/);
   }
   S_I[K]=const_term;

   free(work_items);
   free(thread_IDs);
}

void CA_reconstruct_linsuperpoly_CPU(CA_work_info *w_info) {
   u32 const_term=0;
   int term;
   u32 *idx,nr_idx;

   idx=w_info->data.reconstruct_linsuperpoly.I.idx;
   nr_idx=w_info->data.reconstruct_linsuperpoly.I.nr_idx;
   
   const_term=do_reconstruct_linsuperpoly_CPU(idx,nr_idx,CONSTANT_TERM);
   debug("cons = %08x\n",const_term/*&1*/);

   w_info->data.reconstruct_linsuperpoly.S_I.idx[
               w_info->data.reconstruct_linsuperpoly.S_I.nr_idx-1]=const_term;
   /* Can of course, start from -1 (CONSTANT_TERM) */
   for(term=0;term<w_info->priv_size;term++) {
      u32 sum=0;
      /* calculate term */
      sum=do_reconstruct_linsuperpoly_CPU(idx,nr_idx,term);
      debug("x_%02d = %08x\n",term,(sum^const_term)/*&0x1*/);
      w_info->data.reconstruct_linsuperpoly.S_I.idx[term]=sum^const_term;
   }
}

int CA_reconstruct_linsuperpoly(CA_work_info *w_info) {
   CA_devs v_devices;
   int dim=0;


   dim=w_info->data.reconstruct_linsuperpoly.I.nr_idx;

   if( (dim>=CA_CUBE_DIM_MED) && (!CA_alloc_v_devices(&v_devices,0))) {
      /* if we have a GPU and |I| is not of a low dimension */
      if(dim>=CA_CUBE_DIM_HIGH) {
         printf("Reconstructing high-dimensional superpoly on GPUs...\n");
         CA_reconstruct_linsuperpoly_GPU_h(w_info,&v_devices);
      } else {
         printf("Reconstructing medium-dimensional superpoly on GPUs...\n");
         CA_reconstruct_linsuperpoly_GPU_m(w_info,&v_devices);
      }
      CA_free_v_devices(&v_devices);
   } else {
      /* do it on the CPU */
      printf("Reconstructing low-dimensional superpoly on CPU...\n");
      CA_reconstruct_linsuperpoly_CPU(w_info);
   }


   return 0;
}

int CA_test_maxterm_add_term(CA_work_info *w_info,int idx[], int nr_idx) {
   int i,j;
   if(w_info->data.test_maxterm.cnt_terms>=
         w_info->data.test_maxterm.nr_terms) {
      return -1;
   }

   j=w_info->data.test_maxterm.cnt_terms++;
   CA_ext_term *eterm=&w_info->data.test_maxterm.test_terms[j];
   CA_alloc_idx(&eterm->I,nr_idx);
   for(i=0;i<nr_idx;i++) {
      eterm->I.idx[i]=idx[i];
   }
   w_info->data.test_maxterm.test_terms[j].out_bits=0;
   w_info->data.test_maxterm.min_dim=
      min(w_info->data.test_maxterm.min_dim,nr_idx);
   w_info->data.test_maxterm.max_dim=
      max(w_info->data.test_maxterm.max_dim,nr_idx);

   return 0;
}

void CA_test_maxterm_GPU_m(CA_work_info *w_info, CA_devs *v_devs) {
   int i;
   int D,B,T;
   CA_work_search *work_items;
   CUTThread *thread_IDs;

   if(!(work_items=(CA_work_search*)
            malloc(sizeof(CA_work_search)*v_devs->nr_vdev))) {
      fprintf(stderr,"CA_find_maxterm_GPU: malloc work_items failed "
                                                "with %s\n",strerror(errno));
      exit(-1);
   }
   if(!(thread_IDs=(CUTThread*)malloc(sizeof(CUTThread)*v_devs->nr_vdev))) {
      fprintf(stderr,"CA_find_maxterm_GPU: malloc thread_IDs failed "
                                                "with %s\n",strerror(errno));
      exit(-1);
   }


   /* if we're doing a small-set search, then don't use many devies */
   D=min(v_devs->nr_vdev,w_info->data.test_maxterm.nr_terms);
   /*divide the tests among the devices */
   B=w_info->data.test_maxterm.nr_terms/D;
   /* each block evaluates a full cube sum |I| (max 256 threads): */
   T=min(8,w_info->data.test_maxterm.min_dim);

   for(i=0;i<D;i++) {
      /* assign work */
      work_items[i].pDev=v_devs->devID[i];
      work_items[i].devId=i;
      work_items[i].nDev=D; /* NOTE: not 2^D */
      work_items[i].nBlocks=B; /* NOTE: not 2^B */
      work_items[i].nThreads=1<<T;
      work_items[i].pub_size=w_info->pub_size;
      work_items[i].min_dim=w_info->data.test_maxterm.min_dim;
      work_items[i].max_dim=w_info->data.test_maxterm.max_dim;
      work_items[i].search_terms=&w_info->data.test_maxterm.test_terms[i*B];
   }
   /* handle the odd case; final device has more blocks: */
   work_items[v_devs->nr_vdev-1].nBlocks=B+(w_info->data.test_maxterm.nr_terms-(B*D));

   for(i=0;i<D;i++) {
      /* spawn threads */
      thread_IDs[i]=cutStartThread((CUT_THREADROUTINE)
            do_test_maxterm_GPU_m,(void *)(&work_items[i]));
   }
   cutWaitForThreads(thread_IDs, D);
   free(work_items);
   free(thread_IDs);

}

int CA_test_maxterm_CPU(CA_work_info *w_info) {
   int rc=-1;
   int i;

   for(i=0;i<w_info->data.test_maxterm.cnt_terms;i++) {
      debug("Testing a maxterm [%3d] ...\n",i);
      if((rc=do_test_maxterm_CPU(&w_info->data.test_maxterm.test_terms[i]))) {
         debug("failed!\n");
         return rc;
      }
   }
   return rc;
}

int CA_test_maxterm(CA_work_info *w_info) {
   CA_devs v_devices;
   int min_dim;


   min_dim=max(w_info->data.test_maxterm.min_dim,1);
   if( (!CA_alloc_v_devices(&v_devices,1)) && (min_dim>=CA_CUBE_DIM_MED) ) {
      printf("Testing maxterms on GPUs...\n");
      CA_test_maxterm_GPU_m(w_info,&v_devices);
   } else {
      /* do it on the CPU */
      printf("Testing maxterms on CPU...\n");
      return CA_test_maxterm_CPU(w_info);
   }
   return 0;
}

void CA_find_maxterm_GPU_m(CA_work_info *w_info, CA_devs *v_devs) {
   int i;
   int D,B,T;
   CA_work_search *work_items;
   CUTThread *thread_IDs;

   if(!(work_items=(CA_work_search*)
            malloc(sizeof(CA_work_search)*v_devs->nr_vdev))) {
      fprintf(stderr,"CA_find_maxterm_GPU: malloc work_items failed "
                                                "with %s\n",strerror(errno));
      exit(-1);
   }
   if(!(thread_IDs=(CUTThread*)malloc(sizeof(CUTThread)*v_devs->nr_vdev))) {
      fprintf(stderr,"CA_find_maxterm_GPU: malloc thread_IDs failed "
                                                "with %s\n",strerror(errno));
      exit(-1);
   }


   /* if we're doing a small-set search, then don't use many devies */
   D=min(v_devs->nr_vdev,w_info->data.find_maxterm.nr_searches);
   /*divide the searches among the devices */
   B=w_info->data.find_maxterm.nr_searches/D;
   /* each block evaluates a full cube sum |I| (max 256 threads): */
   T=min(8,w_info->data.find_maxterm.min_dim);

   for(i=0;i<D;i++) {
      /* assign work */
      work_items[i].pDev=v_devs->devID[i];
      work_items[i].devId=i;
      work_items[i].nDev=D; /* NOTE: not 2^D */
      work_items[i].nBlocks=B; /* NOTE: not 2^B */
      work_items[i].nThreads=1<<T;
      work_items[i].pub_size=w_info->pub_size;
      work_items[i].min_dim=w_info->data.find_maxterm.min_dim;
      work_items[i].max_dim=w_info->data.find_maxterm.max_dim;
      work_items[i].search_terms=&w_info->data.find_maxterm.search_terms[i*B];
   }
   /* handle the odd case; final device has more blocks: */
   work_items[v_devs->nr_vdev-1].nBlocks=B+(w_info->data.find_maxterm.nr_searches-(B*D));

   for(i=0;i<D;i++) {
      /* spawn threads */
      thread_IDs[i]=cutStartThread((CUT_THREADROUTINE)
            do_find_maxterm_GPU_m,(void *)(&work_items[i]));
   }
   cutWaitForThreads(thread_IDs, D);
   free(work_items);
   free(thread_IDs);

}

int CA_find_maxterm_CPU(CA_work_info *w_info) {
   int rc=-1;
   int i;

   for(i=0;i<w_info->data.find_maxterm.nr_searches;i++) {
      debug("Trying to find a maxterm [%3d] ...\n",i);
      if(!(rc=do_find_maxterm_CPU(&w_info->data.find_maxterm.search_terms[i],
                              w_info->data.find_maxterm.min_dim,
                              w_info->data.find_maxterm.max_dim,
                              w_info->pub_size))) {
         debug("found!\n");
      } else { debug("failed!\n"); }
   }
   return rc;
}

int CA_find_maxterm(CA_work_info *w_info) {
   CA_devs v_devices;
   int min_dim;


   min_dim=max(w_info->data.find_maxterm.min_dim,1);
   if( (min_dim>=CA_CUBE_DIM_MED) && (!CA_alloc_v_devices(&v_devices,1)) ) {
      printf("Find maxterms on GPUs...\n");
      CA_find_maxterm_GPU_m(w_info,&v_devices);
   } else {
      /* do it on the CPU */
      printf("Find maxterms on CPU...\n");
      return CA_find_maxterm_CPU(w_info);
   }
   return 0;
}

void  CA_init(CA_env *CA, void (*add_job)(CA_work_info *),
                         void (*del_job)(CA_work_info *)) {
   CA->add_job=add_job;
   CA->del_job=del_job;
}

int CA_work(CA_env *CA) {

   (CA->add_job)(&CA->w_info);
   switch(CA->w_info.desc) {
      case RECONSTRUCT_LINSUPERPOLY:
         CA_reconstruct_linsuperpoly(&CA->w_info);
         break;
      case FIND_MAXTERM:
         CA_find_maxterm(&CA->w_info);
         break;
      case TEST_MAXTERM:
         CA_test_maxterm(&CA->w_info);
         break;
      default:
         fprintf(stderr,"Unknown job description id %d.\n",CA->w_info.desc);
         return -1;
   }
   return 0;
}

void CA_exit(CA_env *CA) {
   (CA->del_job)(&CA->w_info);
}

void CA_print_results(CA_env *CA,int nr_output_bits, FILE *fp) {


   switch(CA->w_info.desc) {
      case RECONSTRUCT_LINSUPERPOLY:
         {
            int i,k;
            u32 *S_I=CA->w_info.data.reconstruct_linsuperpoly.S_I.idx;
            int nr_idx=CA->w_info.data.reconstruct_linsuperpoly.S_I.nr_idx;

            fprintf(fp,"I (%2d) ={",
                  CA->w_info.data.reconstruct_linsuperpoly.I.nr_idx);
            for(i=0;i<CA->w_info.data.reconstruct_linsuperpoly.I.nr_idx;i++) {
               fprintf(fp,"%2d, ",
                     CA->w_info.data.reconstruct_linsuperpoly.I.idx[i]);
            }
            fprintf(fp,"}\n");

//               fprintf(fp,"p0 = %08x ",S_I[nr_idx-1]);
            for(k=0;k<nr_output_bits;k++) {
               fprintf(fp,"SuperPoly(%02d) = $ %2d ",k+672,(S_I[nr_idx-1]&(1<<k))>0);
               for(i=0;i<nr_idx-1;i++) {
                  if(S_I[i]&(1<<k)) {
                     fprintf(fp,"+ x_{%02d} ",i);
                  }
               }
               fprintf(fp,"$\n");
            }
         }
         break;
      case FIND_MAXTERM:
         {
            int t;

            for(t=0;t<CA->w_info.data.find_maxterm.nr_searches;t++) {
               int i;
               CA_ext_term *maxterm =
                  &CA->w_info.data.find_maxterm.search_terms[t];
               if(maxterm->out_bits) {
                  /* valid maxterm */
                  fprintf(fp,"I (%2d) ={", maxterm->I.nr_idx);
                  for(i=0;i<maxterm->I.nr_idx;i++) {
                     fprintf(fp,"%2d, ", maxterm->I.idx[i]);
                  }
                  fprintf(fp,"} @ [ ");
                  for(i=0;i<nr_output_bits;i++) {
                     if(maxterm->out_bits&(1<<i)) {
                        fprintf(fp,"%2d, ",i);
                     }
                  }
                  fprintf(fp,"]\n");
               }
            }
         }
         break;
      case TEST_MAXTERM:
         {
            int t;

            for(t=0;t<CA->w_info.data.test_maxterm.cnt_terms;t++) {
               int i;
               CA_ext_term *maxterm =
                  &CA->w_info.data.test_maxterm.test_terms[t];
               if(maxterm->out_bits) {
                  /* valid maxterm */
                  fprintf(fp,"I (%2d) ={", maxterm->I.nr_idx);
                  for(i=0;i<maxterm->I.nr_idx;i++) {
                     fprintf(fp,"%2d, ", maxterm->I.idx[i]);
                  }
                  fprintf(fp,"} @ [ ");
                  for(i=0;i<nr_output_bits;i++) {
                     if(maxterm->out_bits&(1<<i)) {
                        fprintf(fp,"%2d, ",i);
                     }
                  }
                  fprintf(fp,"]\n");
               }
            }
         }
         break;
      default:
         break;
   }
}

