#ifndef CONV_H
#define CONV_H

#include <stdio.h>

#define MAP_SIZE 4096UL
#define MAP_MASK (MAP_SIZE - 1)
#define TARGET_CONV 0x43C00000UL
#define FATAL do { fprintf(stderr, "Error at line %d, file %s (%d) [%s]\n", \
__LINE__, __FILE__, errno, strerror(errno)); exit(1); } while(0)  

int fd_c;
void *map_base_conv;

#define CTRL_C        	*(uint32_t*)(map_base_conv + 0x00)
#define GIER_C        	*(uint32_t*)(map_base_conv + 0x04)
#define IP_IER_C	    *(uint32_t*)(map_base_conv + 0x08)
#define IP_ISR_C	    *(uint32_t*)(map_base_conv + 0x0c)
#define CHin_C	    	*(uint32_t*)(map_base_conv + 0x10)
#define Hin_C	    	*(uint32_t*)(map_base_conv + 0x18)
#define Win_C	    	*(uint32_t*)(map_base_conv + 0x20)
#define CHout_C	    	*(uint32_t*)(map_base_conv + 0x28)
#define Kx_C	    	*(uint32_t*)(map_base_conv + 0x30)
#define Ky_C	    	*(uint32_t*)(map_base_conv + 0x38)
#define Sx_C	    	*(uint32_t*)(map_base_conv + 0x40)
#define Sy_C	    	*(uint32_t*)(map_base_conv + 0x48)
#define mode_C	    	*(uint32_t*)(map_base_conv + 0x50)
#define relu_en_C   	*(uint32_t*)(map_base_conv + 0x58)
#define feature_in_C 	*(uint32_t*)(map_base_conv + 0x60)
#define W_C		    	*(uint32_t*)(map_base_conv + 0x68)
#define bias_C	    	*(uint32_t*)(map_base_conv + 0x70)
#define feature_out_C 	*(uint32_t*)(map_base_conv + 0x78)



void Open_Conv();
void RunConv(unsigned int CHin,unsigned int Hin,unsigned int Win,unsigned int CHout,
		unsigned int Kx,unsigned int Ky,unsigned int Sx,unsigned int Sy,unsigned int mode,unsigned int relu_en,
		float *feature_in,float *W,float *bias,float *feature_out);
void Close_Conv();

#endif  /* CONV_H */
