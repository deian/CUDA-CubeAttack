#include "cube_attack.h"
#include "trivium.h"

void trivium_CA_add_job(CA_work_info *w_info) {
   int j;
//   int term[12]={2,13,20,24,37,42,43,46,53,55,57,7};
   int term[48]={
   2,13,20,24,37,42,43,46,53,55,57,7,
   2,13,20,24,37,42,43,46,53,55,57,7,
   2,13,20,24,37,42,43,46,53,55,57,7,
   2,13,20,24,37,42,43,46,53,55,57,7,
   };
   w_info->desc=RECONSTRUCT_MAXTERM;
   CA_alloc_idx(&w_info->I,48);
   for(j=0;j<w_info->I.nr_idx;j++) {
      w_info->I.idx[j]=term[j];
   }
}
