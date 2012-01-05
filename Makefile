################################################################################

CUBEATTACK=cube_attack
EXAMPLE=trivium
#example
#trivium
# Add source files here
EXECUTABLE	:= $(CUBEATTACK)
# CUDA source files (compiled with cudacc)
CUFILES		:= $(CUBEATTACK)_reconstruct_p.cu
# CUDA dependency files
CU_DEPS		:= $(CUBEATTACK).h $(EXAMPLE)/$(EXAMPLE)_kernel.cu 

# C/C++ source files (compiled with gcc / c++)
CCFILES		:= $(CUBEATTACK).cpp

C_DEPS		:= $(CUBEATTACK).h \
                   $(EXAMPLE)/$(EXAMPLE).h $(EXAMPLE)/$(EXAMPLE).c \

OBJS            += xsr_rng.o $(EXAMPLE)/$(EXAMPLE).o

#keep            =1
verbose         =1

NVCCFLAGS := -arch sm_13
#-Xptxas -v --maxrregcount 16

#CFLAGS = -fopenmp
#CXXFLAGS = -fopenmp 


################################################################################
# Rules and targets
include $(HOME)/NVIDIA_GPU_Computing_SDK/C/common/common.mk

xsr_rng.o: xsr_rng.c xsr_rng.h
	g++ -o xsr_rng.o -c xsr_rng.c
$(EXAMPLE)/$(EXAMPLE).o: $(EXAMPLE)/$(EXAMPLE).c  $(EXAMPLE)/$(EXAMPLE).h
	g++ -o $(EXAMPLE)/$(EXAMPLE).o -c $(EXAMPLE)/$(EXAMPLE).c
