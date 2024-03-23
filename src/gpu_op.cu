#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/* TODO: Your code here */
/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    const float *input_b,
                                                    float *output) {
  // Dynamic shared memory, size provided at kernel launch.
  extern __shared__ float loss_per_row[];
  // Two dimensional thread blocks.
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input_a += y * ncol;
  input_b += y * ncol;
  float maxval = *input_a;
  // Find max for a row.
  for (int x = 1; x < ncol; ++x) {
    maxval = max(maxval, input_a[x]);
  }
  // Deduct by max for a row, and raise to exp.
  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    sum += exp(input_a[x] - maxval);
  }
  // Compute per-row loss.
  float loss = 0;
  for (int x = 0; x < ncol; ++x) {
    loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
  }
  loss_per_row[y] = loss;
  __syncthreads();
  // Compute reduce_mean across rows.
  float mean_loss = 0;
  // Use a single thread to reduce mean across rows.
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    for (int i = 0; i < nrow; ++i) {
      mean_loss += loss_per_row[i];
    }
    mean_loss /= nrow;
    output[0] = mean_loss;
  }
}

__global__ void cudaArraySetValue(int len, float *arr, float value){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < len){
		arr[index] = value;
	}
}

int DLGpuArraySet(DLArrayHandle arr, float value) { 
  /* TODO: Your code here */
  int blockThreads = 1024;
	int numThreads = 1;
	for (int i=0; i < arr->ndim; i++){
		numThreads = numThreads * arr->shape[i];
	}
  float *data = (float *)arr->data;
	dim3 threads, blocks;
	if (numThreads <= blockThreads){
		threads.x = numThreads;
		blocks.x = 1;
	}else{
		threads.x = blockThreads;
		blocks.x = (numThreads + blockThreads -1)/blockThreads;
	}
	cudaArraySetValue<<<blocks, threads>>>(numThreads, data, value);
  return 0;
}

__global__ void cudaBuildOutputArray(int in_threads, int out_threads, const float *input, float *finalOutput){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index < out_threads) {
		finalOutput[index] = input[index % in_threads];
	}
		
}

int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  int inThreads = 1; 
	int outThreads = 1;
	int blockThreads = 1024;
	for(int i = 0; i < input->ndim; i++){
		inThreads = inThreads * input->shape[i];
	}
	for(int i = 0; i < output->ndim; i++){
		outThreads = outThreads * output->shape[i];
	}
	const float *inData = (const float*)input->data; 
	float *outData = (float*)output->data;
	dim3 threads, blocks;
	if (inThreads <= blockThreads){
		threads.x = inThreads;
		blocks.x = 1;
	}else{
		threads.x = blockThreads;
		blocks.x = (outThreads + blockThreads -1)/blockThreads;
	}
	cudaBuildOutputArray<<<blocks, threads>>>(inThreads, outThreads, inData, outData);
	return 0;
}

__global__ void CudaReduceSumAxisZero(const float *inData, float *finalOutput, int rows, int input){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < input){
		finalOutput[index] = 0;
		for (int i=0; i < rows; i++){
				finalOutput[index] += inData[i * input + index];
			}
	}
}

int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  int blockThreads = 1024;
	int numThreads = 1;
	for (int i = 1; i < input->ndim; i++){
		numThreads = numThreads * input->shape[i];
	}
	dim3 threads, blocks;
	float *outData = (float *)output->data;
	const float *inData = (const float *)input->data;
	if (numThreads <= blockThreads){
		threads.x = numThreads;
		blocks.x = 1;
	}else{
		threads.x = blockThreads;
		blocks.x = (numThreads + blockThreads - 1)/blockThreads;
	}
	CudaReduceSumAxisZero<<<blocks, threads>>>(inData, outData, input->shape[0], numThreads);
  return 0;
}

__global__ void CudaMatrixElementwiseAdd(int input, const float *input_a, const float *input_b, float *output_data){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < input){
		output_data[index] = input_a[index] + input_b[index];
	}
}

int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output) {
  /* TODO: Your code here */
	int numThreads = 1;
	int blockThreads = 1024;
	dim3 threads, blocks;
	for (int i = 0; i < matA->ndim; i++){
		numThreads = numThreads * matA->shape[i];
	}
	const float *inputA = (const float *)matA->data;
	const float *inputB = (const float *)matB->data;
	float *outData = (float *)output->data;
	if (numThreads <= blockThreads){
		threads.x = numThreads;
		blocks.x = 1;
	}else{
		threads.x = blockThreads;
		blocks.x = (numThreads + blockThreads - 1)/blockThreads;
	}
	CudaMatrixElementwiseAdd<<<blocks, threads>>>(numThreads, inputA, inputB, outData);
  return 0;
}

__global__ void CudaMatrixElementwiseAddByConst(int input, const float *inputA, float val, float *finalOutput){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < input){
		finalOutput[index] = inputA[index] + val;
	}
}

int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                     DLArrayHandle output) {
  /* TODO: Your code here */
	int numThreads = 1;
	int blockThreads = 1024;
	dim3 threads, blocks;
	for (int i = 0; i < input->ndim; i++){
		numThreads = numThreads * input->shape[i];
	}
	const float *inputA = (const float *)input->data;
	float *outData = (float *)output->data;
	if (numThreads <= blockThreads){
		threads.x = numThreads;
		blocks.x = 1;
	}else{
		threads.x = blockThreads;
		blocks.x = (numThreads + blockThreads - 1)/blockThreads;
	}
	CudaMatrixElementwiseAddByConst<<<blocks, threads>>>(numThreads, inputA, val, outData);
  return 0;
}

__global__ void CudaMatrixElementwiseMultiply(int input, const float *inputA, const float *inputB, float *finalOutput){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < input){
		finalOutput[index] = inputA[index] * inputB[index];
	}
}

int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA,
                                   const DLArrayHandle matB,
                                   DLArrayHandle output) {
  /* TODO: Your code here */
	int numThreads = 1;
	int blockThreads = 1024;
	dim3 threads, blocks;
	for (int i = 0; i < matA->ndim; i++){
		numThreads = numThreads * matA->shape[i];
	}
	const float *inputA = (const float *)matA->data;
	const float *inputB = (const float *)matB->data;
	float *outData = (float *)output->data;
	if (numThreads <= blockThreads){
		threads.x = numThreads;
		blocks.x = 1;
	}else{
		threads.x = blockThreads;
		blocks.x = (numThreads + blockThreads - 1)/blockThreads;
	}
	CudaMatrixElementwiseMultiply<<<blocks, threads>>>(numThreads, inputA, inputB, outData);
  return 0;
}

__global__ void CudaMatrixElementwiseMultiplyConst(int input, const float *inputA, float val, float *finalOutput){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < input){
		finalOutput[index] = inputA[index] * val;
	}
}

int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                               DLArrayHandle output) {
  /* TODO: Your code here */
  int numThreads = 1;
	int blockThreads = 1024;
	dim3 threads, blocks;
	for (int i = 0; i < input->ndim; i++){
		numThreads = numThreads * input->shape[i];
	}
	const float *inputA = (const float *)input->data;
	float *outData = (float *)output->data;
	if (numThreads <= blockThreads){
		threads.x = numThreads;
		blocks.x = 1;
	}else{
		threads.x = blockThreads;
		blocks.x = (numThreads + blockThreads - 1)/blockThreads;
	}
	CudaMatrixElementwiseMultiplyConst<<<blocks, threads>>>(numThreads, inputA, val, outData);
  return 0;
}

int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC) {
  /* TODO: Your code here */
  // Hint: use cublas
  // cublas assume matrix is column major
  cublasHandle_t handle;
	if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS){
		return 0;
	}
	const float *matrixA = (const float *)matA->data;
	const float *matrixB = (const float *)matB->data;
	float *matrixC = (float *)matC->data;
	int i = matC->shape[1];
	int j = matC->shape[0];
	int k = transposeA ? matA->shape[0] : matA->shape[1];
	float alpha = 1.0, beta = 0.0;
	cublasSgemm(handle, transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
                      transposeA ? CUBLAS_OP_T : CUBLAS_OP_N,
                      i, j, k, &alpha,
                      matrixB,
                      transposeB ? k : i,
                      matrixA,
                      transposeA ? j : k,
                      &beta, 
                      matrixC, i);
  return 0;
}

__global__ void CudaRelu(int len, const float *inData, float *finalOutput){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < len){
		if (inData[index] > 0){
			finalOutput[index] = inData[index];
		}else{
			finalOutput[index] = 0;
		}
	}
}

int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  int numThreads = 1;
	int blockThreads = 1024;
	for (int i = 0; i < input->ndim; i++){
		numThreads = numThreads * input->shape[i];
	}
	const float *inData = (const float *)input->data;
	float *outData = (float *)output->data;
	dim3 blocks, threads;
	if (numThreads <= blockThreads){
		threads.x = numThreads;
		blocks.x = 1;
	}else{
		threads.x = blockThreads;
		blocks.x = (numThreads + blockThreads - 1)/blockThreads;
	}
	CudaRelu<<<blocks, threads>>>(numThreads, inData, outData);
  return 0;
}

__global__ void CudaReluGradient(int len, const float *inData, const float *gradient, float *finalOutput){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < len){
		if (inData[index] > 0){
			finalOutput[index] = gradient[index];
		}else{
			finalOutput[index] = 0;
		}
	}
}

int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                      DLArrayHandle output) {
  /* TODO: Your code here */
  int numThreads = 1;
	int blockThreads = 1024;
	for (int i = 0; i < input->ndim; i++){
		numThreads = numThreads * input->shape[i];
	}
	const float *inData = (const float *)input->data;
	const float *gradient = (const float *)in_grad->data;
	float *outData = (float *)output->data;
	dim3 blocks, threads;
	if (numThreads <= blockThreads){
		threads.x = numThreads;
		blocks.x = 1;
	}else{
		threads.x = blockThreads;
		blocks.x = (numThreads + blockThreads - 1)/blockThreads;
	}
	CudaReluGradient<<<blocks, threads>>>(numThreads, inData, gradient ,outData);
  return 0;
}

__global__  void CudaSoftmax(const float *input, float *finalOutput, int rows, int cols) {
	int index = blockDim.x * blockDim.y * blockIdx.x + threadIdx.y * blockDim.x + threadIdx.x;
	if (index < rows){
		input += index * cols;
		finalOutput += index * cols;
		float maximum = *input; 

		for (int i = 1; i < cols; i++) {
			maximum = max(maximum, input[i]);
		}

		float sum = 0.0;
		for (int i = 0; i < cols; i++) {
			sum += exp(input[i] - maximum);
		}

		for (int i = 0; i < cols; i++) {
			finalOutput[i] = exp(input[i] - maximum) / sum;
		}
	}

	
}

int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  int rows, cols;
	dim3 threads;
	int blockThreads = 1024;
	rows = input->shape[0];
	cols = input->shape[1];
	
	const float *inData = (const float *)input->data;
	float *outData = (float *)output->data;
	
	if (rows <= blockThreads) {
		threads.x = rows;
	}else{
		threads.x = blockThreads;
		threads.y = (rows + blockThreads - 1) / blockThreads;
	}

	CudaSoftmax<<<1, threads, rows * sizeof(float)>>>(inData, outData, rows, cols);
  return 0;
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
                             const DLArrayHandle input_b,
                             DLArrayHandle output) {
  assert(input_a->ndim == 2);
  assert(input_b->ndim == 2);
  assert(output->ndim == 1);
  assert(input_a->shape[0] == input_b->shape[0] &&
         input_a->shape[1] == input_b->shape[1]);
  int nrow = input_a->shape[0];
  // Maximum x- or y-dimension of a block = 1024
  // But we need 'nrow' shared memory, and max shared memory is 48KB.
  // Conservatively allow max 16KB shared memory.
  assert(nrow <= 1024 * 4);
  int ncol = input_a->shape[1];
  const float *input_data_a = (const float *)input_a->data;
  const float *input_data_b = (const float *)input_b->data;
  float *output_data = (float *)output->data;
  dim3 threads;
  if (nrow <= 1024) {
    threads.x = nrow;
  } else {
    threads.x = 1024;
    threads.y = (nrow + 1023) / 1024;
  }
  // 1 block, each block with 'threads' number of threads with 'nrow' shared
  // memory size
  matrix_softmax_cross_entropy_kernel<<<1, threads, nrow * sizeof(float)>>>(
      nrow, ncol, input_data_a, input_data_b, output_data);
  return 0;
}
