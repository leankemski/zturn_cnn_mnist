#include <stdio.h>  
#include <stdlib.h>  
#include <unistd.h>  
#include <string.h>
#include <errno.h>  
#include <signal.h>  
#include <fcntl.h>  
#include <ctype.h>   
#include <sys/types.h>  
#include <sys/mman.h>  
#include <stdint.h>
#include "Pool.h"

void Open_Pool()
{
    if((fd_p = open("/dev/mem", O_RDWR | O_SYNC)) == -1) FATAL;
//    printf("/dev/mem opened.\n"); fflush(stdout);
    /* Map one page */
    map_base_pool = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_SHARED,
     fd_p, TARGET_POOL & ~MAP_MASK);
    if(map_base_pool == (void *) -1) FATAL;
//    printf("Memory mapped at address %p.\n", map_base_pool); fflush(stdout);
}

void RunPool(unsigned int CHin,unsigned int Hin,unsigned int Win,
		unsigned int Kx,unsigned int Ky,unsigned int mode,
		float *feature_in,float *feature_out)
{
	CHin_P = CHin;
	Hin_P = Hin;
	Win_P = Win;
	Kx_P = Kx;
	Ky_P = Ky;
	mode_P = mode;
	feature_in_P = (unsigned int)feature_in;
	feature_out_P = (unsigned int)feature_out;

	CTRL_P = CTRL_P & 0x80 | 0x01;
    int tp = CTRL_P;
	while (!((tp>>1) & 0x1)){
		tp = CTRL_P;
	}
}

void Close_Pool()
{
    if(munmap(map_base_pool, MAP_SIZE) == -1) FATAL;  
    close(fd_p);
}
