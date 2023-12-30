#pragma once

#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#define MM_MAX_LINE_LENGTH 1025
#define MatrixMarketBanner "%%MatrixMarket"
#define MM_MAX_TOKEN_LENGTH 64

using MM_typecode = char[4];

/********************* MM_typecode query fucntions ***************************/

#define mm_is_matrix(typecode) ((typecode)[0] == 'M')

#define mm_is_sparse(typecode) ((typecode)[1] == 'C')
#define mm_is_coordinate(typecode) ((typecode)[1] == 'C')
#define mm_is_dense(typecode) ((typecode)[1] == 'A')
#define mm_is_array(typecode) ((typecode)[1] == 'A')

#define mm_is_complex(typecode) ((typecode)[2] == 'C')
#define mm_is_real(typecode) ((typecode)[2] == 'R')
#define mm_is_pattern(typecode) ((typecode)[2] == 'P')
#define mm_is_integer(typecode) ((typecode)[2] == 'I')

#define mm_is_symmetric(typecode) ((typecode)[3] == 'S')
#define mm_is_general(typecode) ((typecode)[3] == 'G')
#define mm_is_skew(typecode) ((typecode)[3] == 'K')
#define mm_is_hermitian(typecode) ((typecode)[3] == 'H')

/********************* MM_typecode modify fucntions ***************************/

#define mm_set_matrix(typecode) ((*typecode)[0] = 'M')
#define mm_set_coordinate(typecode) ((*typecode)[1] = 'C')
#define mm_set_array(typecode) ((*typecode)[1] = 'A')
#define mm_set_dense(typecode) mm_set_array(typecode)
#define mm_set_sparse(typecode) mm_set_coordinate(typecode)

#define mm_set_complex(typecode) ((*typecode)[2] = 'C')
#define mm_set_real(typecode) ((*typecode)[2] = 'R')
#define mm_set_pattern(typecode) ((*typecode)[2] = 'P')
#define mm_set_integer(typecode) ((*typecode)[2] = 'I')

#define mm_set_symmetric(typecode) ((*typecode)[3] = 'S')
#define mm_set_general(typecode) ((*typecode)[3] = 'G')
#define mm_set_skew(typecode) ((*typecode)[3] = 'K')
#define mm_set_hermitian(typecode) ((*typecode)[3] = 'H')

#define mm_clear_typecode(typecode)                                            \
    ((*typecode)[0] = (*typecode)[1] = (*typecode)[2] = ' ',                   \
     (*typecode)[3] = 'G')

#define mm_initialize_typecode(typecode) mm_clear_typecode(typecode)

/********************* Matrix Market error codes ***************************/

#define MM_COULD_NOT_READ_FILE 11
#define MM_PREMATURE_EOF 12
#define MM_NOT_MTX 13
#define MM_NO_HEADER 14
#define MM_UNSUPPORTED_TYPE 15
#define MM_LINE_TOO_LONG 16
#define MM_COULD_NOT_WRITE_FILE 17

/************** Matrix Market internal definitions **************************/
/**
 *                  MM_matrix_typecode: 4-character sequence
 *
 * |                 | ojbect   | sparse/dense | data type | storage scheme |
 * |-----------------|----------|--------------|-----------|----------------|
 * | string position | [0]      | [1]          | [2]       | [3]            |
 * | Matrix typecode | M(atrix) | C(oord)      | R(eal)    | G(eneral)      |
 * |                 |          | A(rray)      | C(omplex) | H(ermitian)    |
 * |                 |          |              | P(attern) | S(ymmetric)    |
 * |                 |          |              | I(nteger) | K(kew)         |
 *
 */
#define MM_MTX_STR "matrix"
#define MM_ARRAY_STR "array"
#define MM_DENSE_STR "array"
#define MM_COORDINATE_STR "coordinate"
#define MM_SPARSE_STR "coordinate"
#define MM_COMPLEX_STR "complex"
#define MM_REAL_STR "real"
#define MM_INT_STR "integer"
#define MM_GENERAL_STR "general"
#define MM_SYMM_STR "symmetric"
#define MM_HERM_STR "hermitian"
#define MM_SKEW_STR "skew-symmetric"
#define MM_PATTERN_STR "pattern"

/**
 * @brief Matrix Market format supports two kind of formats, a sparse coordinate
 * format and a dense array format.
 *
 */
enum matrix_market_format_t { coordinate, array };

/**
 * @brief Data type defines the type of data presented in the file, things like,
 * are they real numbers, complex (real and imaginary), pattern (do not have
 * weights/nonzero-values), etc.
 *
 */
enum matrix_market_data_t { real, complex, pattern, integer };

/**
 * @brief Storage scheme defines the storage structure, symmetric matrix for
 * example will be symmetric over the diagonal. Skew is skew symmetric. Etc.
 *
 */
enum matrix_market_storage_scheme_t { general, hermitian, symmetric, skew };

/****************************** Exception ***********************************/

struct exception_t : std::exception {
    std::string report;

    exception_t(std::string _message = "") { report = _message; }
    virtual const char *what() const noexcept { return report.c_str(); }
};

inline void throw_if_exception(bool is_exception, std::string message = "") {
    if (is_exception)
        throw exception_t(message);
}

template <typename index_t, typename offset_t, typename value_t> 
struct coo_t {
    coo_t(index_t n_rows, index_t n_cols, offset_t nnz)
        : number_of_rows(n_rows), number_of_columns(n_cols),
          number_of_nonzeros(nnz), row_indices(nnz), column_indices(nnz),
          nonzero_values(nnz) {}

    index_t number_of_rows;
    index_t number_of_columns;
    offset_t number_of_nonzeros;
    std::vector<index_t> row_indices;
    std::vector<index_t> column_indices;
    std::vector<value_t> nonzero_values;
};

template <typename index_t,
          typename offset_t,
          typename value_t>
struct csr_t {
  using index_type = index_t;
  using offset_type = offset_t;
  using value_type = value_t;

  index_t number_of_rows;
  index_t number_of_columns;
  offset_t number_of_nonzeros;

  std::vector<offset_t> row_offsets;    // Ap
  std::vector<index_t> column_indices;  // Aj
  std::vector<value_t> nonzero_values;  // Ax
};

inline int mm_read_banner(FILE *f, MM_typecode *matcode) {
    char line[MM_MAX_LINE_LENGTH];
    char banner[MM_MAX_TOKEN_LENGTH];
    char mtx[MM_MAX_TOKEN_LENGTH];
    char crd[MM_MAX_TOKEN_LENGTH];
    char data_type[MM_MAX_TOKEN_LENGTH];
    char storage_scheme[MM_MAX_TOKEN_LENGTH];
    char *p;

    mm_clear_typecode(matcode);

    if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL)
        return MM_PREMATURE_EOF;

    if (sscanf(line, "%s %s %s %s %s", banner, mtx, crd, data_type,
               storage_scheme) != 5)
        return MM_PREMATURE_EOF;

    for (p = mtx; *p != '\0'; *p = tolower(*p), ++p)
        ; /* convert to lower case */
    for (p = crd; *p != '\0'; *p = tolower(*p), ++p)
        ;
    for (p = data_type; *p != '\0'; *p = tolower(*p), ++p)
        ;
    for (p = storage_scheme; *p != '\0'; *p = tolower(*p), ++p)
        ;

    /* check for banner */
    if (strncmp(banner, MatrixMarketBanner, strlen(MatrixMarketBanner)) != 0)
        return MM_NO_HEADER;

    /* first field should be "mtx" */
    if (strcmp(mtx, MM_MTX_STR) != 0)
        return MM_UNSUPPORTED_TYPE;
    mm_set_matrix(matcode);

    /* second field describes whether this is a sparse matrix (in coordinate
            storgae) or a dense array */

    if (strcmp(crd, MM_SPARSE_STR) == 0)
        mm_set_sparse(matcode);
    else if (strcmp(crd, MM_DENSE_STR) == 0)
        mm_set_dense(matcode);
    else
        return MM_UNSUPPORTED_TYPE;

    /* third field */

    if (strcmp(data_type, MM_REAL_STR) == 0)
        mm_set_real(matcode);
    else if (strcmp(data_type, MM_COMPLEX_STR) == 0)
        mm_set_complex(matcode);
    else if (strcmp(data_type, MM_PATTERN_STR) == 0)
        mm_set_pattern(matcode);
    else if (strcmp(data_type, MM_INT_STR) == 0)
        mm_set_integer(matcode);
    else
        return MM_UNSUPPORTED_TYPE;

    /* fourth field */

    if (strcmp(storage_scheme, MM_GENERAL_STR) == 0)
        mm_set_general(matcode);
    else if (strcmp(storage_scheme, MM_SYMM_STR) == 0)
        mm_set_symmetric(matcode);
    else if (strcmp(storage_scheme, MM_HERM_STR) == 0)
        mm_set_hermitian(matcode);
    else if (strcmp(storage_scheme, MM_SKEW_STR) == 0)
        mm_set_skew(matcode);
    else
        return MM_UNSUPPORTED_TYPE;

    return 0;
}

int mm_read_mtx_crd_size(FILE* f,
                         std::size_t* M,
                         std::size_t* N,
                         std::size_t* nz) {
  char line[MM_MAX_LINE_LENGTH];
  int num_items_read;

  /* set return null parameter values, in case we exit with errors */
  *M = *N = *nz = 0;

  /* now continue scanning until you reach the end-of-comments */
  do {
    if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL)
      return MM_PREMATURE_EOF;
  } while (line[0] == '%');

  /* line[] is either blank or has M,N, nz */
  if (sscanf(line, "%zu %zu %zu", M, N, nz) == 3)
    return 0;

  else
    do {
      num_items_read = fscanf(f, "%zu %zu %zu", M, N, nz);
      if (num_items_read == EOF)
        return MM_PREMATURE_EOF;
    } while (num_items_read != 3);

  return 0;
}

template <typename index_t, typename offset_t, typename value_t>
coo_t<index_t, offset_t, value_t> LoadCoo(std::string filename) {
    MM_typecode code;
    // matrix_market_format_t format;         // Sparse coordinate or dense array
    // matrix_market_data_t data;             // Data type
    // matrix_market_storage_scheme_t scheme; // Storage scheme

    FILE *file;

    // Load MTX information
    if ((file = fopen(filename.c_str(), "r")) == NULL) {
        std::cerr << "File could not be opened: " << filename << std::endl;
        exit(1);
    }

    if (mm_read_banner(file, &code) != 0) {
        std::cerr << "Could not process Matrix Market banner" << std::endl;
        exit(1);
    }

    // Make sure we're actually reading a matrix, and not an array
    if (mm_is_array(code)) {
        std::cerr << "File is not a sparse matrix" << std::endl;
        exit(1);
    }

    std::size_t num_rows, num_columns, num_nonzeros;

    if ((mm_read_mtx_crd_size(file, &num_rows, &num_columns, &num_nonzeros)) !=
        0) {
        std::cerr << "Could not read file info (M, N, NNZ)" << std::endl;
        exit(1);
    }

    throw_if_exception(num_rows >= std::numeric_limits<index_t>::max() ||
                           num_columns >= std::numeric_limits<index_t>::max(),
                       "vertex_t overflow");
    throw_if_exception(num_nonzeros >= std::numeric_limits<offset_t>::max(),
                       "edge_t overflow");

    // mtx are generally written as coordinate format

    coo_t<index_t, offset_t, value_t> coo(num_rows, num_columns, num_nonzeros);

    // if (mm_is_coordinate(code))
    //     format = matrix_market_format_t::coordinate;
    // else
    //     format = matrix_market_format_t::array;

    if (mm_is_pattern(code)) {
        // data = matrix_market_data_t::pattern;

        // pattern matrix defines sparsity pattern, but not values
        for (index_t i = 0; i < num_nonzeros; ++i) {
            std::size_t row_index{0}, col_index{0};
            auto num_assigned =
                fscanf(file, " %zu %zu \n", &row_index, &col_index);
            throw_if_exception(num_assigned != 2,
                               "Could not read edge from market file");
            throw_if_exception(row_index == 0, "Market file is zero-indexed");
            throw_if_exception(col_index == 0, "Market file is zero-indexed");
            // set and adjust from 1-based to 0-based indexing
            coo.row_indices[i] = (index_t)row_index - 1;
            coo.column_indices[i] = (index_t)col_index - 1;
            coo.nonzero_values[i] =
                (value_t)1.0; // use value 1.0 for all nonzero entries
        }
    } else if (mm_is_real(code) || mm_is_integer(code)) {
        // if (mm_is_real(code))
        //     data = matrix_market_data_t::real;
        // else
        //     data = matrix_market_data_t::integer;

        for (index_t i = 0; i < coo.number_of_nonzeros; ++i) {
            std::size_t row_index{0}, col_index{0};
            double weight{0.0};

            auto num_assigned = fscanf(file, " %zu %zu %lf \n", &row_index,
                                       &col_index, &weight);

            throw_if_exception(num_assigned != 3,
                               "Could not read weighted edge from market file");
            throw_if_exception(row_index == 0, "Market file is zero-indexed");
            throw_if_exception(col_index == 0, "Market file is zero-indexed");

            coo.row_indices[i] = (index_t)row_index - 1;
            coo.column_indices[i] = (index_t)col_index - 1;
            coo.nonzero_values[i] = (value_t)weight;
        }
    } else {
        std::cerr << "Unrecognized matrix market format type" << std::endl;
        exit(1);
    }

    if (mm_is_symmetric(code)) { // duplicate off diagonal entries
        // scheme = matrix_market_storage_scheme_t::symmetric;
        index_t off_diagonals = 0;
        for (index_t i = 0; i < coo.number_of_nonzeros; ++i) {
            if (coo.row_indices[i] != coo.column_indices[i])
                ++off_diagonals;
        }

        index_t _nonzeros =
            2 * off_diagonals + (coo.number_of_nonzeros - off_diagonals);

        std::vector<index_t> new_I(_nonzeros);
        std::vector<index_t> new_J(_nonzeros);
        std::vector<value_t> new_V(_nonzeros);

        index_t *_I = new_I.data();
        index_t *_J = new_J.data();
        value_t *_V = new_V.data();

        index_t ptr = 0;
        for (index_t i = 0; i < coo.number_of_nonzeros; ++i) {
            if (coo.row_indices[i] != coo.column_indices[i]) {
                _I[ptr] = coo.row_indices[i];
                _J[ptr] = coo.column_indices[i];
                _V[ptr] = coo.nonzero_values[i];
                ++ptr;
                _J[ptr] = coo.row_indices[i];
                _I[ptr] = coo.column_indices[i];
                _V[ptr] = coo.nonzero_values[i];
                ++ptr;
            } else {
                _I[ptr] = coo.row_indices[i];
                _J[ptr] = coo.column_indices[i];
                _V[ptr] = coo.nonzero_values[i];
                ++ptr;
            }
        }
        coo.row_indices = new_I;
        coo.column_indices = new_J;
        coo.nonzero_values = new_V;
        coo.number_of_nonzeros = _nonzeros;
    } // end symmetric case

    fclose(file);

    return coo;
}

/**
 * @brief Convert a Coordinate Sparse Format into Compressed Sparse Row
 * Format.
 *
 * @tparam index_t
 * @tparam offset_t
 * @tparam value_t
 * @param coo
 * @return csr_t<space, index_t, offset_t, value_t>&
 */
template <typename index_t, typename offset_t, typename value_t>
csr_t<index_t, offset_t, value_t>
ToCsr(const coo_t<index_t, offset_t, value_t> &coo) {
    csr_t<index_t, offset_t, value_t> csr;

    csr.number_of_rows = coo.number_of_rows;
    csr.number_of_columns = coo.number_of_columns;
    csr.number_of_nonzeros = coo.number_of_nonzeros;

    offset_t* Ap;
    index_t* Aj;
    value_t* Ax;

    // If returning csr_t on host, use it's internal memory to build from
    // COO.
    csr.row_offsets.resize(csr.number_of_rows + 1);
    csr.column_indices.resize(csr.number_of_nonzeros);
    csr.nonzero_values.resize(csr.number_of_nonzeros);
    Ap = csr.row_offsets.data();
    Aj = csr.column_indices.data();
    Ax = csr.nonzero_values.data();

    // compute number of non-zero entries per row of A.
    for (offset_t n = 0; n < csr.number_of_nonzeros; ++n) {
        ++Ap[coo.row_indices[n]];
    }

    // cumulative sum the nnz per row to get row_offsets[].
    for (index_t i = 0, sum = 0; i < csr.number_of_rows; ++i) {
        index_t temp = Ap[i];
        Ap[i] = sum;
        sum += temp;
    }
    Ap[csr.number_of_rows] = csr.number_of_nonzeros;

    // write coordinate column indices and nonzero values into CSR's
    // column indices and nonzero values.
    for (offset_t n = 0; n < csr.number_of_nonzeros; ++n) {
        index_t row = coo.row_indices[n];
        index_t dest = Ap[row];

        Aj[dest] = coo.column_indices[n];
        Ax[dest] = coo.nonzero_values[n];

        ++Ap[row];
    }

    for (index_t i = 0, last = 0; i <= csr.number_of_rows; ++i) {
        index_t temp = Ap[i];
        Ap[i] = last;
        last = temp;
    }

    return csr; // CSR representation (with possible duplicates)
}
