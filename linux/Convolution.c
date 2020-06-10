#include <stdio.h>  
#include <stdlib.h>  
#include <unistd.h>  
#include <string.h>  
#include <signal.h>
#include <errno.h>  
#include <fcntl.h>  
#include <ctype.h>  
#include <sys/types.h>  
#include <sys/mman.h>  
#include <stdint.h>
#include "Convolution.h"

void Open_Conv()
{
    if((fd_c = open("/dev/mem", O_RDWR | O_SYNC)) == -1) FATAL;
//    printf("/dev/mem opened.\n"); fflush(stdout);
    /* Map one page */
    map_base_conv = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_SHARED,
     fd_c, TARGET_CONV & ~MAP_MASK);
    if(map_base_conv == (void *) -1) FATAL;
//    printf("Memory mapped at address %p.\n", map_base_conv); fflush(stdout);
}

void RunConv(unsigned int CHin,unsigned int Hin,unsigned int Win,unsigned int CHout,
		unsigned int Kx,unsigned int Ky,unsigned int Sx,unsigned int Sy,unsigned int mode,unsigned int relu_en,
		float *feature_in,float *W,float *bias,float *feature_out)
{
	CHin_C = CHin;
	Hin_C = Hin;
	Win_C = Win;
	CHout_C = CHout;
	Kx_C = Kx;
	Ky_C = Ky;
	Sx_C = Sx;
	Sy_C = Sy;
	mode_C = mode;
	relu_en_C = relu_en;
	feature_in_C = (unsigned int)feature_in;
	W_C = (unsigned int)W;
	bias_C = (unsigned int)bias;
	feature_out_C = (unsigned int)feature_out;

	CTRL_C = (CTRL_C & 0x80 | 0x01);
	int tp = CTRL_C;
	while (!((tp>>1) & 0x1)){
		tp = CTRL_C;
	}
}

void Close_Conv()
{
    if(munmap(map_base_conv, MAP_SIZE) == -1) FATAL;  
    close(fd_c);
}
