#include <stdio.h>

#include "../../lib/aoc.h"
#include "../../lib/cuda.h"
#include "../../lib/ops.h"

__global__ void result(int *d_moves, int *d_opp_moves, int n, int *d_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int diff = d_moves[i] - d_opp_moves[i];
        if (diff == 0) {
            d_out[i] = 3;
        } else if (diff == -1 || diff == 2) {
            d_out[i] = 0;
        } else if (diff == 1 || diff == -2) {
            d_out[i] = 6;
        }
    }
}

__global__ void pick_move(int *d_targets, int *d_opp_moves, int n, int *d_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int target = d_targets[i];
        int opp_move = d_opp_moves[i];
        if (target == 0) {
            // lose
            int move = opp_move - 1;
            if (move == -1) {
                move = 2;
            }
            d_out[i] = move;
        } else if (target == 1) {
            // draw
            d_out[i] = opp_move;
        } else if (target == 2) {
            int move = opp_move + 1;
            if (move == 3) {
                move = 0;
            }
            d_out[i] = move;
        }
    }
}

int score(int *d_opp_moves, int *d_moves, int n_rows) {
    int *d_total_move_scores = sum(add(d_moves, n_rows, 1), n_rows);

    int *d_result_scores = empty(n_rows);
    result<<<blocks(n_rows), threads(n_rows)>>>(d_moves, d_opp_moves, n_rows, d_result_scores);
    int *d_total_result_scores = sum(d_result_scores, n_rows);

    int total_move_scores = from_device(d_total_move_scores, 1)[0];
    int total_result_score = from_device(d_total_result_scores, 1)[0];
    return total_move_scores + total_result_score;
}

void part1(int *d_opp_moves, int *d_targets, int n_rows) {
    printf("part1: %d\n", score(d_opp_moves, d_targets, n_rows));
}

void part2(int *d_opp_moves, int *d_targets, int n_rows) {
    int *d_moves = empty(n_rows);
    pick_move<<<blocks(n_rows), threads(n_rows)>>>(d_targets, d_opp_moves, n_rows, d_moves);
    printf("part2: %d\n", score(d_opp_moves, d_moves, n_rows));
}

int main() {
    char *text = read_file((char*) "days/day02/input");
    int max_size = 10000;
    int opp_moves[max_size];
    int targets[max_size];
    int row_idx = 0;
    for (int i = 0; i < strlen(text); i += 4) {
        opp_moves[row_idx] = (int) text[i] - 65;
        targets[row_idx] = (int) text[i + 2] - 88;
        row_idx += 1;
    }
    int n_rows = row_idx;
    /*
    opp_moves:
    A: 0, rock
    B: 1, paper
    C: 2, scissors

    target:
    X: 0, rock,     lose
    Y: 1, paper,    draw
    Z: 2, scissors, win
    */
    int *d_opp_moves = to_device(opp_moves, n_rows);
    int *d_targets = to_device(targets, n_rows);

    part1(d_opp_moves, d_targets, n_rows);
    part2(d_opp_moves, d_targets, n_rows);

    return 0;
}
