#ifndef __MICKEY__H_
#define __MICKEY__H_

#define MICKEY_IV_SIZE  80
#define NR_INIT_ROUNDS   1
#define NR_OUTPUT_BITS  32
#define OUTPUT_BIT_MASK (0xFFFFFFFF>>(32-(NR_OUTPUT_BITS)))
#define TEST_FILE "mickey/mickey.long.test.100"
#include "../cube_attack.h"
void mickey_CA_add_job(CA_work_info *w_info);
void mickey_CA_del_job(CA_work_info *w_info);
#endif
