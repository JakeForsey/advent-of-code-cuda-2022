#include "cuda.h"
#include "ops.h"

__global__ void scatter_add_kernel(int *d_input, int *d_index, int n, int *d_out) {
    for (int i = 0; i < n; i++) {
        d_out[d_index[i]] += d_input[i];
    }
}

int *scatter_add(int *d_input, int *d_index, int n, int n_out) {
    int *d_out = empty(n_out);
    scatter_add_kernel<<<1, 1>>>(d_input, d_index, n, d_out);
    return d_out;
}

__device__ int sum_op(int a, int b) {
    return a + b;
}

__device__ int max_op(int a, int b) {
    return a > b ? a : b;
}

__global__ void reduce_par(int op, int *d_input, int n, int *d_out) {
    extern __shared__ int block_memory[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_idx = threadIdx.x;
    
    block_memory[thread_idx] = d_input[i];
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (thread_idx % (2 * stride) == 0) {
            if (op == 0) {
                block_memory[thread_idx] = sum_op(block_memory[thread_idx], block_memory[thread_idx + stride]);
            } else if (op == 1) {
                block_memory[thread_idx] = max_op(block_memory[thread_idx], block_memory[thread_idx + stride]);
            }
        }
        __syncthreads();
    }

    if (thread_idx == 0) {
        d_out[blockIdx.x] = block_memory[0];
    }
}

__global__ void reduce_seq(int op, int *d_input, int n, int *d_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        for (int j = 0; j < n; j++) {
            if (op == 0) {
                d_out[0] = sum_op(d_out[0], d_input[j]);
            } else if (op == 1) {
                d_out[0] = max_op(d_out[0], d_input[j]);
            }
        }
    }
}

int *reduce(int op, int *d_input, int n) {
    int threads = 512 > n ? n + 1 : 512;
    int blocks = ceil((double) (n + 1) / (double) threads);
    int *d_out = empty(blocks);
    reduce_par<<<blocks, threads, threads * sizeof(int)>>>(op, d_input, n, d_out);
    if (blocks > 1) {
        int *d_out2 = empty(1);
        reduce_seq<<<1, 1>>>(op, d_out, blocks, d_out2);
        return d_out2;
    } else {
        return d_out;
    }
}

int *sum(int *d_input, int n) {
    return reduce(0, d_input, n);
}

int *max(int *d_input, int n) {
    return reduce(1, d_input, n);
}

__global__ void sort(int *d_input, int n, int *d_out) {
    // Bubblesort
    for (int i = 0; i < n; i++) {
        d_out[i] = d_input[i];
    }
    while (true) {
        int swapped = 0;
        for (int i = 1; i < n; i++) {
            if (d_out[i - 1] > d_out[i]) {
                int l = d_out[i - 1];
                int r = d_out[i];
                d_out[i] = l;
                d_out[i - 1] = r;
                swapped = 1;
            }
        }
        if (swapped == 0) {
            break;
        }
    }
}

int *sort(int *d_input, int n) {
    int *d_sorted = empty(n);
    sort<<<1, 1>>>(d_input, n, d_sorted);
    return d_sorted;
}

__global__ void top_3(int *d_input, int n, int *d_out) {
    for (int i = 0; i < 3; i++) {
        d_out[i] = d_input[n - i - 1];
    }
}

int *top_3(int *d_input, int n) {
    int *d_out = empty(3);
    top_3<<<1, 1>>>(d_input, n, d_out);
    return d_out;
}

__global__ void add(int *d_input, int n, int other, int *d_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        d_out[i] = d_input[i] + other;
    }
}

int *add(int *d_input, int n, int other) {
    int *d_out = empty(n);
    add<<<threads(n), blocks(n)>>>(d_input, n, other, d_out);
    return d_out;
}
