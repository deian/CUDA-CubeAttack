#ifndef __TRIVIUM__H_
#define __TRIVIUM__H_

#define NR_INIT_ROUNDS   672//75
#define NR_OUTPUT_BITS    32
#define OUTPUT_BIT_MASK (0xFFFFFFFF>>(32-(NR_OUTPUT_BITS)))
#define TEST_FILE "trivium/trivium.long.test.672"

#include "../cube_attack.h"
void trivium_CA_add_job(CA_work_info *w_info);
void trivium_CA_del_job(CA_work_info *w_info);
#endif
