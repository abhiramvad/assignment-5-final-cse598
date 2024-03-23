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
  int totalElements = 1;
	for (int i=0; i < arr->ndim; i++){
		totalElements *= arr->shape[i];
	}
  float *arrData = (float *)arr->data;
	dim3 threads, blocks;
	int threadsPerBlock = 1024;
	if (totalElements <= threadsPerBlock){
		threads.x = totalElements;
		blocks.x = 1;
	}else{
		threads.x = threadsPerBlock;
		blocks.x = (totalElements + threadsPerBlock -1) / threadsPerBlock;
	}
	cudaArraySetValue<<<blocks, threads>>>(totalElements, arrData, value);
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
  int inputSize = 1, outputSize = 1;
	for(int i = 0; i < input->ndim; i++){
		inputSize *= input->shape[i];
	}
	for(int i = 0; i < output->ndim; i++){
		outputSize *= output->shape[i];
	}
	const float *inPtr = (const float*)input->data; 
	float *outPtr = (float*)output->data;
	dim3 threads, blocks;
	int threadsPerBlock = 1024;
	if (inputSize <= threadsPerBlock){
		threads.x = inputSize;
		blocks.x = 1;
	}else{
		threads.x = threadsPerBlock;
		blocks.x = (outputSize + threadsPerBlock -1) / threadsPerBlock;
	}
	cudaBuildOutputArray<<<blocks, threads>>>(inputSize, outputSize, inPtr, outPtr);
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
  int totalElements = 1;
	for (int i = 1; i < input->ndim; i++){
		totalElements *= input->shape[i];
	}
	dim3 threads, blocks;
	float *outPtr = (float *)output->data;
	const float *inPtr = (const float *)input->data;
	int threadsPerBlock = 1024;
	if (totalElements <= threadsPerBlock){
		threads.x = totalElements;
		blocks.x = 1;
	}else{
		threads.x = threadsPerBlock;
		blocks.x = (totalElements + threadsPerBlock - 1) / threadsPerBlock;
	}
	CudaReduceSumAxisZero<<<blocks, threads>>>(inPtr, outPtr, input->shape[0], totalElements);
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
	int totalElements = 1;
	for (int i = 0; i < matA->ndim; i++){
		totalElements *= matA->shape[i];
	}
	const float *matAPtr = (const float *)matA->data;
	const float *matBPtr = (const float *)matB->data;
	float *outputPtr = (float *)output->data;
	dim3 threads, blocks;
	int threadsPerBlock = 1024;
	if (totalElements <= threadsPerBlock){
		threads.x = totalElements;
		blocks.x = 1;
	}else{
		threads.x = threadsPerBlock;
		blocks.x = (totalElements + threadsPerBlock - 1) / threadsPerBlock;
	}
	CudaMatrixElementwiseAdd<<<blocks, threads>>>(totalElements, matAPtr, matBPtr, outputPtr);
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
	int totalElements = 1;
	dim3 threads, blocks;
	for (int i = 0; i < input->ndim; i++){
		totalElements = totalElements * input->shape[i];
	}
	const float *inPtr = (const float *)input->data;
	float *outPtr = (float *)output->data;
	int threadsPerBlock = 1024;
	if (totalElements <= threadsPerBlock){
		threads.x = totalElements;
		blocks.x = 1;
	}else{
		threads.x = threadsPerBlock;
		blocks.x = (totalElements + threadsPerBlock - 1)/threadsPerBlock;
	}
	CudaMatrixElementwiseAddByConst<<<blocks, threads>>>(totalElements, inPtr, val, outPtr);
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
	int totalElements = 1;
	dim3 threads, blocks;
	for (int i = 0; i < matA->ndim; i++){
		totalElements = totalElements * matA->shape[i];
	}
	const float *matAPtr = (const float *)matA->data;
	const float *matBPtr = (const float *)matB->data;
	float *outData = (float *)output->data;
	int threadsPerBlock = 1024;
	if (totalElements <= threadsPerBlock){
		threads.x = totalElements;
		blocks.x = 1;
	}else{
		threads.x = threadsPerBlock;
		blocks.x = (totalElements + threadsPerBlock - 1)/threadsPerBlock;
	}
	CudaMatrixElementwiseMultiply<<<blocks, threads>>>(totalElements, matAPtr, matBPtr, outData);
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
  	int totalElements = 1;
	dim3 threads, blocks;
	for (int i = 0; i < input->ndim; i++){
		totalElements = totalElements * input->shape[i];
	}
	const float *inPtr = (const float *)input->data;
	float *outPtr = (float *)output->data;
	int threadsPerBlock = 1024;
	if (totalElements <= threadsPerBlock){
		threads.x = totalElements;
		blocks.x = 1;
	}else{
		threads.x = threadsPerBlock;
		blocks.x = (totalElements + threadsPerBlock - 1)/threadsPerBlock;
	}
	CudaMatrixElementwiseMultiplyConst<<<blocks, threads>>>(totalElements, inPtr, val, outPtr);
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
  int totalElements = 1;
	for (int i = 0; i < input->ndim; i++){
		totalElements *= input->shape[i];
	}
	const float *inPtr = (const float *)input->data;
	float *outPtr = (float *)output->data;
	dim3 blockSize, gridSize;
	int threadsPerBlock = 1024;
	if (totalElements <= threadsPerBlock){
		blockSize.x = totalElements;
		gridSize.x = 1;
	}else{
		blockSize.x = threadsPerBlock;
		gridSize.x = (totalElements + threadsPerBlock - 1) / threadsPerBlock;
	}
	CudaRelu<<<gridSize, blockSize>>>(totalElements, inPtr, outPtr);
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
  int totalElements = 1;
	for (int i = 0; i < input->ndim; i++){
		totalElements *= input->shape[i];
	}
	const float *inPtr = (const float *)input->data;
	const float *inGradPtr = (const float *)in_grad->data;
	float *outputPtr = (float *)output->data;
	dim3 blocks, threads;
	int threadsPerBlock = 1024;
	if (totalElements <= threadsPerBlock){
		threads.x = totalElements;
		blocks.x = 1;
	}else{
		threads.x = threadsPerBlock;
		blocks.x = (totalElements + threadsPerBlock - 1) / threadsPerBlock;
	}
	CudaReluGradient<<<blocks, threads>>>(totalElements, inPtr, inGradPtr, outputPtr);
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
	int threadsPerBlock = 1024;
	rows = input->shape[0];
	cols = input->shape[1];
	
	const float *inPtr = (const float *)input->data;
	float *outPtr = (float *)output->data;
	
	if (rows <= threadsPerBlock) {
		threads.x = rows;
	}else{
		threads.x = threadsPerBlock;
		threads.y = (rows + threadsPerBlock - 1) / threadsPerBlock;
	}

	CudaSoftmax<<<1, threads, rows * sizeof(float)>>>(inPtr, outPtr, rows, cols);
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
