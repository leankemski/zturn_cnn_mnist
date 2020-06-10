#ifndef POOL_H
#define POOL_H

#include <stdio.h>

#define MAP_SIZE 4096UL
#define MAP_MASK (MAP_SIZE - 1)
#define TARGET_POOL 0x43C10000UL
#define FATAL do { fprintf(stderr, "Error at line %d, file %s (%d) [%s]\n", \
__LINE__, __FILE__, errno, strerror(errno)); exit(1); } while(0)  

int fd_p;
void *map_base_pool;

#define CTRL_P        	*(uint32_t*)(map_base_pool + 0x00)
#define GIER_P        	*(uint32_t*)(map_base_pool + 0x04)
#define IP_IER_P	    *(uint32_t*)(map_base_pool + 0x08)
#define IP_ISR_P	    *(uint32_t*)(map_base_pool + 0x0c)
#define CHin_P	    	*(uint32_t*)(map_base_pool + 0x10)
#define Hin_P	    	*(uint32_t*)(map_base_pool + 0x18)
#define Win_P	    	*(uint32_t*)(map_base_pool + 0x20)
#define Kx_P	    	*(uint32_t*)(map_base_pool + 0x28)
#define Ky_P	    	*(uint32_t*)(map_base_pool + 0x30)
#define mode_P	    	*(uint32_t*)(map_base_pool + 0x38)
#define feature_in_P  	*(uint32_t*)(map_base_pool + 0x40)
#define feature_out_P 	*(uint32_t*)(map_base_pool + 0x48)

void Open_Pool();
void RunPool(unsigned int CHin,unsigned int Hin,unsigned int Win,
		unsigned int Kx,unsigned int Ky,unsigned int mode,
		float *feature_in,float *feature_out);//mode: 0:MEAN, 1:MIN, 2:MAX
void Close_Pool();

#endif  /* POOL_H */
