#include <stdio.h>  
#include <stdlib.h>  
#include <unistd.h>   //sleep() s
#include <string.h>  
#include <signal.h>
#include <errno.h>  
#include <fcntl.h>  
#include <ctype.h>  
#include <sys/types.h>  
#include <sys/mman.h>  
#include <stdint.h> 
#include <time.h>

#include "Convolution.h"
#include "Pool.h"
#include "load.h"

#define TARGET 0x20000000UL

//Conv1
#define IN_WIDTH1 28
#define IN_HEIGHT1 28
#define IN_CH1 1

#define KERNEL_WIDTH1 3
#define KERNEL_HEIGHT1 3
#define X_STRIDE1 1
#define Y_STRIDE1 1

#define RELU_EN1 1
#define MODE1 1  //0:VALID, 1:SAME
#define X_PADDING1 (MODE1? (KERNEL_WIDTH1-1)/2 : 0)
#define Y_PADDING1 (MODE1? (KERNEL_HEIGHT1-1)/2 : 0)

#define OUT_CH1 16
/* #define OUT_WIDTH1 (IN_WIDTH1+2*X_PADDING1-KERNEL_WIDTH1)/X_STRIDE1+1
#define OUT_HEIGHT1 (IN_HEIGHT1+2*Y_PADDING1-KERNEL_HEIGHT1)/Y_STRIDE1+1 */
#define OUT_WIDTH1 28
#define OUT_HEIGHT1 28

//Pool1
#define MODE11 2  //mode: 0:MEAN, 1:MIN, 2:MAX
#define IN_WIDTH11 OUT_WIDTH1
#define IN_HEIGHT11 OUT_HEIGHT1
#define IN_CH11 OUT_CH1

#define KERNEL_WIDTH11 2
#define KERNEL_HEIGHT11 2

/* #define OUT_CH11 IN_CH11
#define OUT_WIDTH11 IN_WIDTH11/KERNEL_WIDTH11
#define OUT_HEIGHT11 IN_HEIGHT11/KERNEL_HEIGHT11 */
#define OUT_CH11 16
#define OUT_WIDTH11 14
#define OUT_HEIGHT11 14

//Conv2
#define IN_WIDTH2 OUT_WIDTH11
#define IN_HEIGHT2 OUT_HEIGHT11
#define IN_CH2 OUT_CH11

#define KERNEL_WIDTH2 3
#define KERNEL_HEIGHT2 3
#define X_STRIDE2 1
#define Y_STRIDE2 1

#define RELU_EN2 1
#define MODE2 1  //0:VALID, 1:SAME
#define X_PADDING2 (MODE2? (KERNEL_WIDTH2-1)/2 : 0)
#define Y_PADDING2 (MODE2? (KERNEL_HEIGHT2-1)/2 : 0)

#define OUT_CH2 32
/* #define OUT_WIDTH2 (IN_WIDTH2+2*X_PADDING2-KERNEL_WIDTH2)/X_STRIDE2+1
#define OUT_HEIGHT2 (IN_HEIGHT2+2*Y_PADDING2-KERNEL_HEIGHT2)/Y_STRIDE2+1 */
#define OUT_WIDTH2 14
#define OUT_HEIGHT2 14

//Pool2
#define MODE21 2  //mode: 0:MEAN, 1:MIN, 2:MAX
#define IN_WIDTH21 OUT_WIDTH2
#define IN_HEIGHT21 OUT_HEIGHT2
#define IN_CH21 OUT_CH2

#define KERNEL_WIDTH21 2
#define KERNEL_HEIGHT21 2

/* #define OUT_CH21 IN_CH21
#define OUT_WIDTH21 IN_WIDTH21/KERNEL_WIDTH21
#define OUT_HEIGHT21 IN_HEIGHT21/KERNEL_HEIGHT21 */
#define OUT_CH21 32
#define OUT_WIDTH21 7
#define OUT_HEIGHT21 7

//Fc1
#define IN_WIDTH3 OUT_WIDTH21
#define IN_HEIGHT3 OUT_HEIGHT21
#define IN_CH3 OUT_CH21

#define KERNEL_WIDTH3 7
#define KERNEL_HEIGHT3 7
#define X_STRIDE3 1
#define Y_STRIDE3 1

#define RELU_EN3 1
#define MODE3 0  //0:VALID, 1:SAME
#define X_PADDING3 (MODE3? (KERNEL_WIDTH3-1)/2 : 0)
#define Y_PADDING3 (MODE3? (KERNEL_HEIGHT3-1)/2 : 0)

#define OUT_CH3 128
/* #define OUT_WIDTH3 (IN_WIDTH3+2*X_PADDING3-KERNEL_WIDTH3)/X_STRIDE3+1
#define OUT_HEIGHT3 (IN_HEIGHT3+2*Y_PADDING3-KERNEL_HEIGHT3)/Y_STRIDE3+1 */
#define OUT_WIDTH3 1
#define OUT_HEIGHT3 1

//Fc2
#define IN_WIDTH4 OUT_WIDTH3
#define IN_HEIGHT4 OUT_HEIGHT3
#define IN_CH4 OUT_CH3

#define KERNEL_WIDTH4 1
#define KERNEL_HEIGHT4 1
#define X_STRIDE4 1
#define Y_STRIDE4 1

#define RELU_EN4 1
#define MODE4 0  //0:VALID, 1:SAME
#define X_PADDING4 (MODE4? (KERNEL_WIDTH4-1)/2 : 0)
#define Y_PADDING4 (MODE4? (KERNEL_HEIGHT4-1)/2 : 0)

#define OUT_CH4 10
/* #define OUT_WIDTH4 (IN_WIDTH4+2*X_PADDING4-KERNEL_WIDTH4)/X_STRIDE4+1
#define OUT_HEIGHT4 (IN_HEIGHT4+2*Y_PADDING4-KERNEL_HEIGHT4)/Y_STRIDE4+1 */
#define OUT_WIDTH4 1
#define OUT_HEIGHT4 1

//Weight of Conv1
float *image;
float *W_conv1;
float *b_conv1;
float *h_conv1;
float *h_pool1;

//Weight of Conv2
float *W_conv2;
float *b_conv2;
float *h_conv2;
float *h_pool2;

//Weight of FC1
float *W_fc1;
float *b_fc1;
float *h_fc1;

//Weight of FC2
float *W_fc2;
float *b_fc2;
float *h_fc2;

int main(int argc, char *argv[])
{
/* 	printf("out_height1 = %d, out_width1 = %d, out_ch1 = %d\n", OUT_HEIGHT1, OUT_WIDTH1, OUT_CH1);
	printf("in_height11 = %d, in_width11 = %d, in_ch11 = %d\n", IN_HEIGHT11, IN_WIDTH11, IN_CH11);
	printf("out_height11 = %d, out_width11 = %d, out_ch11 = %d\n", OUT_HEIGHT11, OUT_WIDTH11, OUT_CH11);
	printf("out_height2 = %d, out_width2 = %d, out_ch2 = %d\n", OUT_HEIGHT2, OUT_WIDTH2, OUT_CH2);
	printf("out_height21 = %d, out_width21 = %d, out_ch21 = %d\n", OUT_HEIGHT21, OUT_WIDTH21, OUT_CH21);
    printf("out_height3 = %d, out_width3 = %d, out_ch3 = %d\n", OUT_HEIGHT3, OUT_WIDTH3, OUT_CH3);
	printf("out_height4 = %d, out_width4 = %d, out_ch4 = %d\n", OUT_HEIGHT4, OUT_WIDTH4, OUT_CH4); */
	int fd;
	void *map_base;
    float *phy0;
    float *phy1;
    float *phy2;
    float *phy3;
    float *phy4;
    float *phy5;
    float *phy6;
    float *phy7;
    float *phy8;
    float *phy9;
    float *phy10;
    float *phy11;
    float *phy12;
    float *phy13;
    float *phy14;
    float *phy15;

    Open_Conv();
    Open_Pool();

    if((fd = open("/dev/mem", O_RDWR | O_SYNC)) == -1) FATAL;
//    printf("/dev/mem opened.\n"); fflush(stdout);
    map_base = mmap(0, 1024*1024*4, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_SHARED, fd, TARGET);
    if(map_base == (void *) -1) FATAL;
//    printf("Memory mapped at address %p.\n", map_base); fflush(stdout);

	phy1 = (float *)TARGET; //image
	phy2 = phy1 + IN_HEIGHT1*IN_WIDTH1*IN_CH1;   //w_conv1
	phy3 = phy2 + KERNEL_HEIGHT1*KERNEL_WIDTH1*IN_CH1*OUT_CH1;  //b_conv1
	phy4 = phy3 + OUT_CH1;        //h_conv1
	phy5 = phy4 + OUT_HEIGHT1*OUT_WIDTH1*OUT_CH1;  //h_pool1
	phy6 = phy5 + OUT_HEIGHT11*OUT_WIDTH11*OUT_CH11;  //w_conv2
	phy7 = phy6 + KERNEL_HEIGHT2*KERNEL_WIDTH2*IN_CH2*OUT_CH2; //b_conv2 
	phy8 = phy7 + OUT_CH2;	   //h_conv2
	phy9 = phy8 + OUT_HEIGHT2*OUT_WIDTH2*OUT_CH2;  //h_pool2
	phy10 = phy9 + OUT_HEIGHT21*OUT_WIDTH21*OUT_CH21;   //w_fc1
	phy11 = phy10 + KERNEL_HEIGHT3*KERNEL_WIDTH3*IN_CH3*OUT_CH3;  //b_fc1
	phy12 = phy11 + OUT_CH3;	       //h_fc1
	phy13 = phy12 + OUT_HEIGHT3*OUT_WIDTH3*OUT_CH3;         //w_fc2
	phy14 = phy13 + KERNEL_HEIGHT4*KERNEL_WIDTH4*IN_CH4*OUT_CH4;      //b_fc2
	phy15 = phy14 + OUT_CH4;          //h_fc2

	image = map_base;
	W_conv1 = image + IN_HEIGHT1*IN_WIDTH1*IN_CH1;
	b_conv1 = W_conv1 + KERNEL_HEIGHT1*KERNEL_WIDTH1*IN_CH1*OUT_CH1;
	h_conv1 = b_conv1 + OUT_CH1;
	h_pool1 = h_conv1 + OUT_HEIGHT1*OUT_WIDTH1*OUT_CH1;
	W_conv2 = h_pool1 + OUT_HEIGHT11*OUT_WIDTH11*OUT_CH11;
	b_conv2 = W_conv2 + KERNEL_HEIGHT2*KERNEL_WIDTH2*IN_CH2*OUT_CH2;
	h_conv2 = b_conv2 + OUT_CH2;
	h_pool2 = h_conv2 + OUT_HEIGHT2*OUT_WIDTH2*OUT_CH2;
	W_fc1   = h_pool2 + OUT_HEIGHT21*OUT_WIDTH21*OUT_CH21;
	b_fc1   = W_fc1   + KERNEL_HEIGHT3*KERNEL_WIDTH3*IN_CH3*OUT_CH3;
	h_fc1   = b_fc1   + OUT_CH3;
	W_fc2   = h_fc1   + OUT_HEIGHT3*OUT_WIDTH3*OUT_CH3;
	b_fc2   = W_fc2   + KERNEL_HEIGHT4*KERNEL_WIDTH4*IN_CH4*OUT_CH4;
	h_fc2   = b_fc2   + OUT_CH4;

/*  	printf("image virtual addr = %p , physical addr = %#x\n", image, phy1);
	printf("W_conv1 virtual addr = %p , physical addr = %#x\n", W_conv1, phy2);
	printf("b_conv1 virtual addr = %p , physical addr = %#x\n", b_conv1, phy3);
	printf("h_conv1 virtual addr = %p , physical addr = %#x\n", h_conv1, phy4);
	printf("h_pool1 virtual addr = %p , physical addr = %#x\n", h_pool1, phy5);
	printf("W_conv2 virtual addr = %p , physical addr = %#x\n", W_conv2, phy6);
	printf("b_conv2 virtual addr = %p , physical addr = %#x\n", b_conv2, phy7);
	printf("h_conv2 virtual addr = %p , physical addr = %#x\n", h_conv2, phy8);
	printf("h_pool2 virtual addr = %p , physical addr = %#x\n", h_pool2, phy9);
	printf("W_fc1 virtual addr = %p , physical addr = %#x\n", W_fc1, phy10);
	printf("b_fc1 virtual addr = %p , physical addr = %#x\n", b_fc1, phy11);
	printf("h_fc1 virtual addr = %p , physical addr = %#x\n", h_fc1, phy12);
	printf("W_fc2 virtual addr = %p , physical addr = %#x\n", W_fc2, phy13);
	printf("b_fc2 virtual addr = %p , physical addr = %#x\n", b_fc2, phy14);
	printf("h_fc2 virtual addr = %p , physical addr = %#x\n", h_fc2, phy15); */

    LoadWeight("data/W_conv1.dat",KERNEL_HEIGHT1*KERNEL_WIDTH1*IN_CH1*OUT_CH1,W_conv1);
    LoadWeight("data/b_conv1.dat",OUT_CH1,b_conv1);	
    LoadWeight("data/W_conv2.dat",KERNEL_HEIGHT2*KERNEL_WIDTH2*IN_CH2*OUT_CH2,W_conv2);	
	LoadWeight("data/b_conv2.dat",OUT_CH2,b_conv2);
	LoadWeight("data/W_fc1.dat",KERNEL_HEIGHT3*KERNEL_WIDTH3*IN_CH3*OUT_CH3,W_fc1);
	LoadWeight("data/b_fc1.dat",OUT_CH3,b_fc1);
	LoadWeight("data/W_fc2.dat",KERNEL_HEIGHT4*KERNEL_WIDTH4*IN_CH4*OUT_CH4,W_fc2);
	LoadWeight("data/b_fc2.dat",OUT_CH4,b_fc2);
	
	printf("Initialize done!\n");
	
	unsigned char frameGet[28*28];
	LoadBmp(argv[1], frameGet);

	printf("Load bmp done!\n");

	for(int j=0;j<28*28;j++)
	{
		*(image+(27-j/28)*28+j%28) = (255-frameGet[j]*1.0)/255;
	}
	
	clock_t start, stop;
	start = clock();
	
	//Conv1
	RunConv(IN_CH1, IN_HEIGHT1, IN_WIDTH1, OUT_CH1,  //1,28,28,16 CHin,Hin,Win,CHout
			KERNEL_WIDTH1, KERNEL_HEIGHT1, X_STRIDE1, Y_STRIDE1, MODE1, RELU_EN1,  //3,3,1,1,1,1 Kx,Ky,Sx,Sy,mode,relu_en
			phy1, phy2, phy3, phy4);  //feature_in,W,bias,feature_out

	RunPool(IN_CH11, IN_HEIGHT11, IN_WIDTH11,  //16,28,28 CHin,Hin,Win
			KERNEL_WIDTH11, KERNEL_HEIGHT11, MODE11,  //2,2,2 Kx,Ky,mode
			phy4, phy5);  //feature_in,feature_out

	//Conv2
	RunConv(IN_CH2, IN_HEIGHT2, IN_WIDTH2, OUT_CH2,  //16,14,14,32 CHin,Hin,Win,CHout
			KERNEL_WIDTH2, KERNEL_HEIGHT2, X_STRIDE2, Y_STRIDE2, MODE2, RELU_EN2,  //3,3,1,1,1,1 Kx,Ky,Sx,Sy,mode,relu_en
			phy5, phy6, phy7, phy8);  //feature_in,W,bias,feature_out

	RunPool(IN_CH21, IN_HEIGHT21, IN_WIDTH21,  //32,14,14, CHin,Hin,Win
			KERNEL_WIDTH21, KERNEL_HEIGHT21, MODE21,  //2,2,2 Kx,Ky,mode
			phy8, phy9);  //feature_in,feature_out

	//Fc1
	RunConv(IN_CH3, IN_HEIGHT3, IN_WIDTH3, OUT_CH3,  //32,7,7,128 CHin,Hin,Win,CHout
			KERNEL_WIDTH3, KERNEL_HEIGHT3, X_STRIDE3, Y_STRIDE3, MODE3, RELU_EN3,  //7,7,1,1,0,1 Kx,Ky,Sx,Sy,mode,relu_en
			phy9, phy10, phy11, phy12);  //feature_in,W,bias,feature_out

	//Fc2
	RunConv(IN_CH4, IN_HEIGHT4, IN_WIDTH4, OUT_CH4,  //128,1,1,10 CHin,Hin,Win,CHout
			KERNEL_WIDTH4, KERNEL_HEIGHT4, X_STRIDE4, Y_STRIDE4, MODE4, RELU_EN4,  //1,1,1,1,0,1 Kx,Ky,Sx,Sy,mode,relu_en
			phy12, phy13, phy14, phy15);  //feature_in,W,bias,feature_out

	printf("CNN done!\n");
	
	float max = *h_fc2;
	int num = 1;
	for(int m=0; m<OUT_CH4; m++)
	{
//		printf("%d = %f\n", m, *(h_fc2+m));
		if(*(h_fc2+m) > max)
		{
			max=*(h_fc2+m);
			num=m;
		}
	}
	printf("%d\r\n", num);
	stop = clock();
	printf("time cost = %f s\n", (double)(stop-start)/CLOCKS_PER_SEC);
	
    munmap(map_base, 1024*1024*4);	

    close(fd);
    Close_Conv();
    Close_Pool();

    return 0;
}

