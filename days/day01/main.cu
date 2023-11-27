#include <stdio.h>

#include "../../lib/aoc.h"
#include "../../lib/cuda.h"
#include "../../lib/ops.h"

int main() {
    char *text = read_file((char*) "days/day01/input");
    int max_size = 100000;
    int counts[max_size];
    int groups[max_size];
    int group = 0;
    int row_idx = 0;
    int start = 0;
    for (int end = 0; end < strlen(text); end++) {
        if (text[end] == 10 && text[end - 1] != 10) {
            char row[(end - start) + 1];
            memcpy(row, &text[start], (1 + end - start) * sizeof(char));
            counts[row_idx] = (int) strtol(row, NULL, 10);
            groups[row_idx] = group;
            row_idx += 1;
            start = end + 1;
        } else if (text[end] == 10 && text[end - 1] == 10) {
            group += 1;
            start = end + 1;
        }
    }
    int n_groups = group;
    int n_rows = row_idx;
    int *d_counts = to_device(counts, n_rows);
    int *d_groups = to_device(groups, n_rows);

    // Shared
    int *d_group_totals = scatter_add(d_counts, d_groups, n_rows, n_groups);

    // Part 1:
    int *d_max = max(d_group_totals, n_groups);
    printf("part 1: %i\n", from_device(d_max, 1)[0]);

    // Part 2:
    int *d_sorted = sort(d_group_totals, n_groups);
    int *d_top = top_3(d_sorted, n_groups);
    int *d_sum = sum(d_top, 3);
    printf("part 2: %i\n", from_device(d_sum, 1)[0]);

    return 0;
}
