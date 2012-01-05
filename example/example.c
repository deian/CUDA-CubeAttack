#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include "../xsr_rng.h"
#include "../cube_attack.h"
#include "example.h"

#define CUBE_SIZE 2

void example_CA_add_job(CA_work_info *w_info) {
   int j;
   int I[CUBE_SIZE]={0,1};
   int I_siz=CUBE_SIZE;

   w_info->key_size=4;
   w_info->iv_size=4;

   w_info->desc=RECONSTRUCT_LINSUPERPOLY;
   CA_alloc_idx(&w_info->data.reconstruct_linsuperpoly.I,I_siz);
   /*allocate for the whole key+const term: makes reading easier */
   CA_alloc_idx(&w_info->data.reconstruct_linsuperpoly.S_I,w_info->key_size+1);
   for(j=0;j<I_siz;j++) {
      w_info->data.reconstruct_linsuperpoly.I.idx[j]=I[j];
   }

}

void example_CA_del_job(CA_work_info *w_info) {
   CA_free_idx(&w_info->data.reconstruct_linsuperpoly.I);
   CA_free_idx(&w_info->data.reconstruct_linsuperpoly.S_I);
}

int benchmark(void) {
   CA_env CA;
   struct timeval tv_start,tv_end;
   double proc_time;

   {
      FILE *fp;
      u64 seed=time(0);
      fp = fopen("/dev/urandom", "r");
      fread(&seed,sizeof(seed),1,fp);
      fclose(fp);
      xsr_srand_u32((u32)(seed));
      xsr_srand_u64(seed);
   }

   CA_init(&CA,&example_CA_add_job,&example_CA_del_job);

   gettimeofday(&tv_start,NULL);

   if(CA_work(&CA)) {
      fprintf(stderr,"CA_init failed!\n");
      CA_exit(&CA);
      return -1;
   }
   gettimeofday(&tv_end,NULL);

   proc_time=(double)(tv_end.tv_sec-tv_start.tv_sec)+
             (double)(tv_end.tv_usec-tv_start.tv_usec)/1000000.0;


   CA_print_results(&CA,stdout);
   printf("total time=%5.7g sec\n",proc_time);

   CA_exit(&CA);
   return 0;
}

int try_find(void) {
   CA_env CA;
   struct timeval tv_start,tv_end;
   double proc_time;

   {
      FILE *fp;
      u64 seed=time(0);
      fp = fopen("/dev/urandom", "r");
      fread(&seed,sizeof(seed),1,fp);
      fclose(fp);
      xsr_srand_u32((u32)(seed));
      xsr_srand_u64(seed);
   }

   CA_init(&CA,&example_CA_add_job,&example_CA_del_job);

   gettimeofday(&tv_start,NULL);
   {
      int i;
      int I[CUBE_SIZE];//={1,2};
      int nr=CUBE_SIZE;
      CA_term tI;
      CA_alloc_idx(&tI,nr);
      for(i=0;i<nr;i++) { tI.idx[i]=I[i]; }
      do_find_maxterm_CPU(&tI);
   }


/*
   if(CA_work(&CA)) {
      fprintf(stderr,"CA_init failed!\n");
      CA_exit(&CA);
      return -1;
   }
 */
   gettimeofday(&tv_end,NULL);

   proc_time=(double)(tv_end.tv_sec-tv_start.tv_sec)+
             (double)(tv_end.tv_usec-tv_start.tv_usec)/1000000.0;


/*   CA_print_results(&CA,stdout); */
   printf("total time=%5.7g sec\n",proc_time);

//   CA_exit(&CA);
   return 0;
}

int main(void) {
   return benchmark();
//   return try_find();
}
