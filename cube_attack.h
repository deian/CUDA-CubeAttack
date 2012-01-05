#ifndef __CUBE_ATTACK_H__
#define __CUBE_ATTACK_H__

#include <stdint.h>


typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int32_t  i32;

//#define CA_DEBUG
//#define CA_VERBOSE_STAT

#define CA_CUBE_DIM_MED    0 /* cubes of medium dimension are above this */
#define CA_CUBE_DIM_HIGH 100 /* and below this */

#define CA_NR_TESTS  128  /* number of BLR tests to run during maxterm search*/
#define CA_MAX_TRIES_PER_I 20 /* number of tries before starting with a new
                                 in searching for maxterms */

enum CA_work_type { RECONSTRUCT_LINSUPERPOLY, 
                    FIND_MAXTERM, 
                    TEST_MAXTERM/*, ONLINE_ATTACK*/ };

typedef struct {
   u32 *idx; /* array of index terms */
   int nr_idx; /* number of index terms */
} CA_term;

typedef struct {
   CA_term I;    /* the term */
   u32 out_bits; /* the bit positions for which it passed the BLR tests*/
   /*TODO: allow for more than 32-bit outputs */
} CA_ext_term;


typedef struct {
   int *devID;  /*array of device id's (can be repeated) */
   int nr_pdev; /* number of physical devices */
   int nr_vdev; /* number of virtual devices (power of 2) */
} CA_devs;

typedef struct {
   enum CA_work_type desc; /* work description */
   int priv_size,pub_size; /* number of private and public variables */
   union {
      struct {
         CA_term I;   /*input: maxterm indexes*/
         CA_term S_I; /*output: indexes of terms in superpoly + const term*/
      } reconstruct_linsuperpoly;

      /* search for maxterms */
      struct {
         int min_dim; /*in: minimum dimension of maxterm */
         int max_dim; /*in: maximum dimension of maxterm */
         int nr_searches; /*in: the number of searches (i.e., nr blocks) */
         CA_ext_term *search_terms; /*out: the maxterms, if any */
      } find_maxterm;

      /* test existing maxterms */
      struct {
         int min_dim; /*in: minimum dimension of maxterm */
         int max_dim; /*in: maximum dimension of maxterm */
         int nr_terms; /*in: the number of terms (i.e., nr blocks) */
         int cnt_terms; /* counter used when adding new terms to list */
         CA_ext_term *test_terms; /*in/out: the maxterms in, out_bits out */
      } test_maxterm;

   } data;
} CA_work_info;

typedef struct {
   int pDev;            /* physical device */
   u32 devId,           /* virtual device id */
       nDev,            /* number of virtual devices */
       nBlocks,         /* number of blocks/device */
       nThreads,        /* number of threads/block */
       nIterations;     /* number of iterations/thread */
   CA_term *I;
   union {
      struct {
         int term;      /*-1=const, 0,..key_size for the rest*/
         u32 partial_sum;
      } hcube;
      struct {
         int blkOffset; /* nr blocks/device excluding the odd case*/
         CA_term *S_I;  /* pointer to S_I */
      } mcube;
   } data;
} CA_work_reconstruct;


typedef struct {
   int pDev;            /* physical device */
   u32 devId,           /* virtual device id */
       nDev,            /* number of virtual devices */
       nBlocks,         /* number of blocks/device */
       nThreads;        /* number of threads/block */
   int pub_size;        /* number of public variables */
   int min_dim;         /* minimum dimension of term */
   int max_dim;         /* maximum dimension of term */
   CA_ext_term *search_terms; /*out: the maxterms, if any */
} CA_work_search;
#define CONSTANT_TERM -1


struct CA_env {
   void (*add_job)(CA_work_info *);
   void (*del_job)(CA_work_info *);

   CA_work_info w_info;
};


/* ------------------------------------------------------------------------- */
/* Outside API */
void CA_init(CA_env *CA, void (*add_job)(CA_work_info *),
                         void (*del_job)(CA_work_info *));
void CA_exit(CA_env *CA);
int  CA_work(CA_env *CA);
void CA_print_results(CA_env *CA,int nr_output_bits, FILE *fp);

/* limit the physical devices to a subset */
int CA_limit_dev(int nr_dev, ...);
/* ------------------------------------------------------------------------- */

void CA_alloc_idx(CA_term *term,int nr_idx);
void CA_free_idx(CA_term *term);

int CA_alloc_v_devices(CA_devs *v_devs,int strict_physical);
void CA_free_v_devices(CA_devs *v_devs);

void CA_alloc_search_terms(CA_ext_term **term,int nr_terms,int max_dim);
void CA_free_search_terms(CA_ext_term *term,int nr_terms);

void CA_alloc_test_terms(CA_work_info *w_info, int nr_terms);
void CA_free_test_terms(CA_work_info *w_info);

/* ------------------------------------------------------------------------- */
int  CA_reconstruct_linsuperpoly(CA_work_info *w_info);
void CA_reconstruct_linsuperpoly_GPU_h(CA_work_info *w_info, CA_devs *v_devs);
void CA_reconstruct_linsuperpoly_GPU_m(CA_work_info *w_info, CA_devs *v_devs);

void do_reconstruct_linsuperpoly_GPU_h(CA_work_reconstruct *w_mterm);
void do_reconstruct_linsuperpoly_GPU_m(CA_work_reconstruct *w_mterm);
u32  do_reconstruct_linsuperpoly_CPU(u32 *I,int nr_idx, int term);

/* ------------------------------------------------------------------------- */
int  CA_find_maxterm(CA_work_info *w_info);
int  CA_find_maxterm_CPU(CA_work_info *w_info);
void CA_find_maxterm_GPU_m(CA_work_info *w_info, CA_devs *v_devs);

int do_find_maxterm_CPU(CA_ext_term *term,int min_dim,int max_dim,int pub_size);
void do_find_maxterm_GPU_m(CA_work_search *w_sterm);

/* ------------------------------------------------------------------------- */
int CA_test_maxterm(CA_work_info *w_info);
int CA_test_maxterm_CPU(CA_work_info *w_info);
int CA_test_maxterm_add_term(CA_work_info *w_info,int idx[], int nr_idx);

int do_test_maxterm_CPU(CA_ext_term *test_term);
void do_test_maxterm_GPU_m(CA_work_search *w_sterm);


void CA_test_maxterm_GPU_m(CA_work_info *w_info, CA_devs *v_devs);
/* ------------------------------------------------------------------------- */
int  CA_test_black_box_CPU(char *fname);

int  test_inline_xsr(int dev, int nr_threads, int nr_blocks,int N);



#if defined(CA_VERBOSE_STAT)
#define CA_DEBUG
#endif


#ifdef CA_DEBUG
#define debug(...) fprintf(stderr, __VA_ARGS__)
#else
#define debug(...) ;
#endif

#define swap(a,b,type)          \
{                               \
   type tmp=a; a=b;b=tmp;       \
}

/* see p. 103 of Cormen, et al's "Introduction to Algorithms" */
#define randomize_in_place(A,n,type)                    \
{                                                       \
   int rip_i;                                           \
   for(rip_i=0;rip_i<n;rip_i++) {                       \
      swap((A)[rip_i],(A)[rand_int(rip_i,(n)-1)],type); \
   }                                                    \
}

/* fill array A[0:n-1] with random intergs\in[start,end[ */
#define array_rnd_fill(A,n,start,end,rnd_int)                   \
   {  /* very lazy array filler */                              \
      int __arf_i,__arf_j;                                      \
      for(__arf_i=0;__arf_i<(n);__arf_i++) {                    \
         do {                                                   \
            (A)[__arf_i]=rnd_int((start),(end)-1);              \
            for(__arf_j=0;__arf_j<__arf_i;__arf_j++) {          \
               if((A)[__arf_i]==(A)[__arf_j]) break;            \
            }                                                   \
         } while(__arf_i!=__arf_j);                             \
      }                                                         \
   }

#define array_rnd_add_el(A,n,start,end,rnd_int) \
   {                                            \
      int __ara_j;                              \
      do {                                      \
         (A)[(n)]=rnd_int((start),(end)-1);     \
         for(__ara_j=0;__ara_j<(n);__ara_j++) { \
            if((A)[(n)]==(A)[__ara_j]) break;   \
         }                                      \
      } while((n)!=__ara_j);                    \
      (n)++;                                    \
   }

/* remove random element from u32 array A */
#define array_rnd_rm_el(A,n,rnd_int)    \
   {                                    \
      int el=rnd_int(0,(n)-1);          \
      (A)[el]=(A)[(n)-1];               \
      (n)--;                            \
   }

#endif
