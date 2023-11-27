#ifndef OPS_H_
#define OPS_H_

int *scatter_add(int *d_input, int *d_index, int n, int n_out);

int *sum(int *d_input, int n);

int *max(int *d_input, int n);

int *sort(int *d_input, int n);

int *top_3(int *d_input, int n);

#endif
