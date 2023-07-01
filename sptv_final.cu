 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "device_functions.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <omp.h>
#include <math.h>


__global__ void SPMV(int p, int NUM_THREAD, int* Ap, int* rows, double* y, double* values, double* x) 
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int j, i, k;
	if(idx<NUM_THREAD)
	{
		for (i = (Ap[idx]-Ap[0]); i < (Ap[idx+1]-Ap[0]); i++)
		{
			y[idx] += values[i] * x[rows[i]];
		}
		
	}
}



int main(int argc, char** argv)
{


	cudaDeviceProp prop;
	int whichDevice;
	cudaGetDevice(&whichDevice);
	cudaGetDeviceProperties(&prop, whichDevice);
	if (!prop.deviceOverlap)
	{
		printf("not handle overlap \n");
		return 0;
	}



	double startt, endt;
	double time_host, time_device;
	unsigned long timer;

	FILE* f, * fw;
	int M;
	int N;
	long nz, i, j;

	char* filename = "/data/ratings-m1_c.txt";

	if ((f = fopen(filename, "r")) == NULL)
		exit(1);

	fscanf(f, "%d %d %ld", &N, &M, &nz);

	int num_row;
	num_row = N;
	int num_column;
	num_column = M;
	long nnz;
	nnz = nz;
	printf("num_row=%d, num_column=%d, nnz=%ld\n", num_row, num_column, nnz);

	/* reseve memory for matrices */

	int* rows = (int*)malloc(nnz * sizeof(int));
	//int* columns = (int*)malloc(nnz * sizeof(int));
	//double* values = (double*)malloc(nnz * sizeof(double));
	int* columns;
	double* values;
	cudaHostAlloc((int**)&columns, nnz * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((double**)&values, nnz * sizeof(double), cudaHostAllocDefault);

	int B=256*256*2;
	int row;
	int column;
	double value;
	long g = 0;
	
	
	j=0;
	for (i = 0; i < nnz; i++)
	{

		fscanf(f, "%d %d	%lf", &(row), &(column), &(values[i]));
		rows[i] = row - 1;
		columns[i] = column - 1;
	}



	fclose(f);

	
	
	//number of nonzeros in each column
	int* num_nonzeros = (int*)malloc(num_column * sizeof(int));
	for(i=0; i<num_column; i++)
		num_nonzeros[i]=0;
	printf("******************\n");
	
	for (i = 0; i < nnz; i++)
	{
		num_nonzeros[columns[i]]++;
	}


	//COO2CSC
	int* Ap;
	cudaHostAlloc((int**)&Ap, (num_column + 1) * sizeof(int), cudaHostAllocDefault);
	Ap[0] = 0;
	
	
	int maxl = 0;
	int sum = 0;
	j = 0;
	
	
	for (i = 0; i < num_column; i++)
	{
		sum += num_nonzeros[i];
		Ap[i + 1] = sum;
		//printf("Ap[%d]: %d\n", i + 1, sum);
		if (maxl < (Ap[i + 1] - Ap[i]))
		{
			maxl = Ap[i + 1] - Ap[i];
		}
	}
	printf("maxl: %d\n", maxl);
	
	g=0;
	

	double* x, * y;
	cudaHostAlloc((double**)&y, num_column * sizeof(double), cudaHostAllocDefault);
	cudaHostAlloc((double**)&x, num_row * sizeof(double), cudaHostAllocDefault);
	for (i = 0; i < num_row; i++) {
		x[i] = (double)(num_row%5);
	}
	for (i = 0; i < num_column; i++) {
		y[i] = 0.0;
	}
	
	
	
	float milliseconds = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	

	double* y_host;
	cudaHostAlloc((double**)&y_host, num_column * sizeof(double), cudaHostAllocDefault);



	dim3 block( (B+256-1) / 256 );
	dim3 grid(256);
	int* Ap_device1, * Ap_device2;
	int* rows_device1, * rows_device2;
	double* y_device1, * y_device2;
	double* values_device1, * values_device2;
	double* x_device1;


	cudaMalloc((int**)&Ap_device1, (num_column + 1) * sizeof(int));
	cudaMalloc((int**)&Ap_device2, (num_column + 1) * sizeof(int));
	cudaMalloc((int**)&rows_device1, nnz * sizeof(int));
	cudaMalloc((int**)&rows_device2, nnz * sizeof(int));
	cudaMalloc((double**)&y_device1, num_column * sizeof(double));
	cudaMalloc((double**)&y_device2, num_column * sizeof(double));
	cudaMalloc((double**)&values_device1, nnz * sizeof(double));
	cudaMalloc((double**)&values_device2, nnz * sizeof(double));
	cudaMalloc((double**)&x_device1, num_row * sizeof(double));
	
	
	
	
	cudaStream_t stream0, stream1;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);
	
	
	cudaMemcpyAsync(y_device1, y, num_column * sizeof(double), cudaMemcpyHostToDevice, stream0);
	cudaMemcpyAsync(y_device2, y, num_column * sizeof(double), cudaMemcpyHostToDevice, stream1);
	
	
	int len_locate;
	len_locate = (num_column+B-1)/B+1;
	int* locate;
	cudaHostAlloc((int**)&locate, len_locate * sizeof(int), cudaHostAllocDefault);
	locate[0]=0;
	
	for(i=1; i<len_locate; i++)
	{
		if( (locate[i-1]+B) < num_column )
			locate[i] = locate[i-1]+B;
		else
			locate[i] = num_column;
	}
	
	
	
	//sort the blocks
	int tmp, tmp_id;
	int* sort_id;
	cudaHostAlloc((int**)&sort_id, (len_locate-1) * sizeof(int), cudaHostAllocDefault);
	for(i=0; i<(len_locate-1); i++)
		sort_id[i] = i;
		
	
	for(i=0; i<(len_locate-1); i++)
	{
		tmp = Ap[locate[sort_id[i]+1]]-Ap[locate[sort_id[i]]];
		for(j=i+1; j<(len_locate-1); j++)
		{
			if( (Ap[locate[sort_id[j]+1]]-Ap[locate[sort_id[j]]]) > tmp )
			{
				tmp = Ap[locate[sort_id[j]+1]]-Ap[locate[sort_id[j]]];
				tmp_id = sort_id[i];
				sort_id[i] = sort_id[j];
				sort_id[j] = tmp_id;
			}
		}
	}
	
	
	
	cudaEventRecord(start, 0);
	cudaMemcpyAsync(x_device1, x, num_row * sizeof(double), cudaMemcpyHostToDevice, stream0);
	
	i = 0; 
	while( ((i+2)<len_locate) )
	{
		
		cudaMemcpyAsync(Ap_device1, Ap + locate[sort_id[i]], (locate[sort_id[i]+1]-locate[sort_id[i]] + 1) * sizeof(int), cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(rows_device1, rows + Ap[locate[sort_id[i]]], (Ap[locate[sort_id[i]+1]] - Ap[locate[sort_id[i]]]) * sizeof(int), cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(values_device1, values + Ap[locate[sort_id[i]]], (Ap[locate[sort_id[i]+1]] - Ap[locate[sort_id[i]]]) * sizeof(double), cudaMemcpyHostToDevice, stream0);
		

		cudaMemcpyAsync(Ap_device2, Ap + locate[sort_id[i]+1], (locate[sort_id[i]+2]-locate[sort_id[i]+1] + 1) * sizeof(int), cudaMemcpyHostToDevice, stream1);
		cudaMemcpyAsync(rows_device2, rows + Ap[locate[sort_id[i]+1]], (Ap[locate[sort_id[i]+2]] - Ap[locate[sort_id[i]+1]]) * sizeof(int), cudaMemcpyHostToDevice, stream1);
		cudaMemcpyAsync(values_device2, values + Ap[locate[sort_id[i]+1]], (Ap[locate[sort_id[i]+2]] - Ap[locate[sort_id[i]+1]]) * sizeof(double), cudaMemcpyHostToDevice, stream1);
	
		
		SPMV << < grid, block, 0, stream0 >> > (locate[sort_id[i]], locate[sort_id[i]+1]-locate[sort_id[i]], Ap_device1, rows_device1, y_device1, values_device1, x_device1);
		SPMV << < grid, block, 0, stream1 >> > (locate[sort_id[i]+1], locate[sort_id[i]+2]-locate[sort_id[i]+1], Ap_device2, rows_device2, y_device2, values_device2, x_device1);
		
		cudaMemcpyAsync(y_host + locate[sort_id[i]], y_device1, (locate[sort_id[i]+1]-locate[sort_id[i]]) * sizeof(double), cudaMemcpyDeviceToHost, stream0);
		cudaMemcpyAsync(y_host + locate[sort_id[i]+1], y_device2, (locate[sort_id[i]+2]-locate[sort_id[i]+1]) * sizeof(double), cudaMemcpyDeviceToHost, stream1);
		
		
		i+=2;
	}
	if( (i==0) || (i==len_locate) )
	{
		if(i==len_locate)
			i=i-2;
		cudaMemcpyAsync(Ap_device1, Ap + locate[sort_id[i]], (locate[sort_id[i]+1]-locate[sort_id[i]] + 1) * sizeof(int), cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(rows_device1, rows + Ap[locate[sort_id[i]]], (Ap[locate[sort_id[i]+1]] - Ap[locate[sort_id[i]]]) * sizeof(int), cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(values_device1, values + Ap[locate[sort_id[i]]], (Ap[locate[sort_id[i]+1]] - Ap[locate[sort_id[i]]]) * sizeof(double), cudaMemcpyHostToDevice, stream0);
	
		
		SPMV << < grid, block, 0, stream0 >> > (locate[sort_id[i]], locate[sort_id[i]+1]-locate[sort_id[i]], Ap_device1, rows_device1, y_device1, values_device1, x_device1);
		
		
		cudaMemcpyAsync(y_host + locate[i], y_device1, (locate[sort_id[i]+1]-locate[sort_id[i]]) * sizeof(double), cudaMemcpyDeviceToHost, stream0);
		
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("milliseconds = %lf\n", milliseconds);
	
	
	cudaStreamSynchronize(stream0);
	cudaStreamSynchronize(stream1);
	
	
	return 0;
}

