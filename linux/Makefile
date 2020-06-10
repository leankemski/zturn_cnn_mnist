cc=gcc
target=mnist
obj=mnist.o Convolution.o Pool.o load.o
$(target):$(obj)
	$(cc) $(obj) -Wall -o $(target) -std=c99
mnist.o:mnist.c
	$(cc) -c mnist.c -std=c99
Convolution.o:Convolution.c
	$(cc) -c Convolution.c -std=c99
Pool.o:Pool.c
	$(cc) -c Pool.c -std=c99
load.o:load.c
	$(cc) -c load.c -std=c99
clean:
	$(RM) *.o $(target)
