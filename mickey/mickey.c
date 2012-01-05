#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include "../xsr_rng.h"
#include "../cube_attack.h"
#include "mickey.h"



/* - find maxterm ------------------------------------------------- */
void mickey_CA_add_job_findmaxterm(CA_work_info *w_info) {
   int min_dim=20;
   int max_dim=23;
   int nr_searches=60;

   w_info->priv_size=80;
   w_info->pub_size=MICKEY_IV_SIZE;

   w_info->desc=FIND_MAXTERM;
   w_info->data.find_maxterm.min_dim=min_dim;
   w_info->data.find_maxterm.max_dim=max_dim;
   w_info->data.find_maxterm.nr_searches=nr_searches;
   CA_alloc_search_terms(&w_info->data.find_maxterm.search_terms,
                                                   nr_searches,max_dim);
}

void mickey_CA_del_job_findmaxterm(CA_work_info *w_info) {
   CA_free_search_terms(w_info->data.find_maxterm.search_terms,
                                 w_info->data.find_maxterm.nr_searches);
}
/* ---------------------------------------------------------------- */

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


   CA_init(&CA,&mickey_CA_add_job_findmaxterm,&mickey_CA_del_job_findmaxterm);

   gettimeofday(&tv_start,NULL);

   if(CA_work(&CA)) {
      fprintf(stderr,"CA_init failed!\n");
      CA_exit(&CA);
      return -1;
   }

   gettimeofday(&tv_end,NULL);

   proc_time=(double)(tv_end.tv_sec-tv_start.tv_sec)+
             (double)(tv_end.tv_usec-tv_start.tv_usec)/1000000.0;


   CA_print_results(&CA,NR_OUTPUT_BITS,stdout);
   printf("total time=%5.7g sec\n",proc_time);

   CA_exit(&CA);
   return 0;
}



int main(void) {
   CA_limit_dev(2,2,3);
   return benchmark();
}

