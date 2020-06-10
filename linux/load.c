#include <stdio.h>
#include <stdlib.h>

/* void LoadWeight(char *filename, float *dp)
{
	float value;
	FILE *fp;
	
	if (!(fp=fopen(filename,"r")))
	{
	 printf("Error in open file!\n");
	 exit(1);
	}
	
	int i = 0;
	while (fscanf(fp, "%f", &value))
	{
 		// if(i == 0)
			// printf("value = %f\n", value);
		*(dp+4*i) = value;
		i++;
		if (feof(fp))
		{
			printf("%d\n",i);
			break;
		}
	}
	fclose(fp);
} */

void LoadWeight(char *filename, int num, float *dp)
{
	float value;
	FILE *fp;
	
	if (!(fp=fopen(filename,"r")))
	{
	 printf("Error in open file!\n");
	 exit(1);
	}
	
	int i = 0;
	while (fscanf(fp, "%f", &value))
	{
 		if(i < num)
		{
			*(dp+i) = value;
			i++;
		}
		else
			break;
	}
	fclose(fp);
}

void LoadBmp(char *filename, unsigned char *line_buf)
{
	FILE *fp;

	/* 文件打开 */
	if((fp = fopen(filename, "rb")) == NULL)
	{
		printf("Can't open the bmp!\n");
	}
	
	fseek(fp, 1078, SEEK_SET);  //文件头加调色板
	fread(line_buf, 784, 1, fp);
	
/* 	int i;
    for(i = 0 ;i < 784; i++)
    {
        printf("%d = %d\n",i,line_buf[i]);
    } */
	
	fclose(fp);
}



