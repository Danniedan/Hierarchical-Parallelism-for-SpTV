#include "mpi.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>


// Kernel that executes on the CUDA device
void computePI(int nproc,int rank);



int main(int argc, char** argv)
{
	int rank, nproc;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	printf("We are running %d processes.\n", nproc);
	
	
	computePI(nproc, rank);
	MPI_Finalize();
	
	
	return 0;
}

