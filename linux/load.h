#ifndef LOAD_H
#define LOAD_H

#include <stdio.h>

//void LoadWeight(char *filename, float *dp);
void LoadWeight(char *filename, int num, float *dp);
void LoadBmp(char *filename, unsigned char *line_buf);

#endif  /* LOAD_H */
