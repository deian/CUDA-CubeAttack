################################################################################

TARGET=xsr_rng_test
# Add source files here
EXECUTABLE	:= $(TARGET)
# CUDA source files (compiled with cudacc)
CUFILES		:= $(TARGET).cu
# CUDA dependency files
CU_DEPS		:= ../xsr_rng.c ../xsr_rng.h

OBJS            += ../xsr_rng.o

# C/C++ source files (compiled with gcc / c++)
CCFILES		:= 

keep            =1
verbose         =1
NVCCFLAGS := -Xptxas -v
#--maxrregcount 16



################################################################################
# Rules and targets

#include ../../common/common.mk
include /home/d/NVIDIA_GPU_Computing_SDK/C/common/common.mk
../xsr_rng.o: ../xsr_rng.c ../xsr_rng.h
	g++ -o ../xsr_rng.o -c ../xsr_rng.c
