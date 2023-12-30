#pragma once

template <typename index_t, typename offset_t, typename mat_value_t,
          typename vec_x_value_t, typename vec_y_value_t>
void SpMV_cpu_navie(index_t n_rows, index_t n_cols, offset_t nnz,
    const offset_t *Ap, const index_t *Aj, const mat_value_t *Ax, 
    const vec_x_value_t *x, vec_y_value_t *y) {
    
    for (index_t row = 0; row < n_rows; ++row) {
        vec_y_value_t sum = vec_x_value_t(0);

        for (index_t col_off = Ap[row]; col_off < Ap[row+1]; ++col_off) {
            sum += Ax[col_off] * x[Aj[col_off]];
        }
        y[row] = sum;
    }
}
