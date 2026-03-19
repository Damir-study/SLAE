#include "classMatrix.h"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <random>

void swap(double *n1, double *n2) {
    double temp = *n1;
    *n1 = *n2;
    *n2 = temp;
}

// Конструктор
Matrix::Matrix(int dimension) : dim(dimension) {
    data = new double[dim * dim]();
}

// Конструктор копирования
Matrix::Matrix(const Matrix& other) : dim(other.dim) {
    data = new double[dim * dim];
    for (int i = 0; i < dim * dim; ++i) data[i] = other.data[i];
}

// Деструктор
Matrix::~Matrix() {
    delete[] data;
}

// Оператор присваивания
Matrix& Matrix::operator=(const Matrix& other) {
    if (this == &other) return *this;

    if (dim != other.dim) {
        delete[] data;
        dim = other.dim;
        data = new double[dim * dim];
    }

    for (int i = 0; i < dim * dim; ++i) {
        data[i] = other.data[i];
    }
    return *this;
}



// Вывод
void Matrix::print() const{
    //фиксированно 5 знаков после запятой
    std::cout << std::fixed << std::setprecision(5);
    for (int i = 0; i < dim; ++i) {
        for (int j=0;j<dim;++j) {
            std::cout  << (*this)(i,j) << "    ";
        }
        std::cout << std::endl;
    }
    std::cout<<std::endl;
}

// Ввод данных
void Matrix::input_from_console() {
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            std::cout << "Enter element [" << i + 1 << "][" << j + 1 << "]: ";
            std::cin >> (*this)(i, j);
        }
    }
}



// Вспомогательный метод округления
double Matrix::round_value(double value, int places) {
    const double multiplier = std::pow(10.0, places);
    return std::round(value * multiplier) / multiplier;
}

// Вспомогательный: выбор ведущего элемента
int Matrix::choice_leading(int row, int& swaps, Matrix* second_matrix) {
    int idx = row;
    int mx_row = idx;

    for (int i = idx + 1; i < dim; ++i) {
        if (std::abs((*this)(i, idx)) > std::abs((*this)(mx_row, idx))) {
            mx_row = i;
        }
    }

    if (std::abs((*this)(mx_row, idx)) < 1e-12) return 0;

    if (mx_row != idx) {
        // Переставляем в текущей матрице
        for (int j = 0; j < dim; ++j) {
            swap(&(*this)(idx, j), &(*this)(mx_row, j));
        }
        
        if (second_matrix != nullptr) {
            for (int j = 0; j < second_matrix->dim; ++j) {
                swap(&(*second_matrix)(idx, j), &(*second_matrix)(mx_row, j));
            }
        }
        swaps++;
    }
    return 1;
}

// Вспомогательный: выбор ведущего элемента для gauss_choice_solving
void Matrix::choice_leading(int row, std::vector<double>& b) {
    int mx_row = row;

    for (int i = row + 1; i < dim; ++i) {
        if (std::abs((*this)(i, row)) > std::abs((*this)(mx_row, row))) {
            mx_row = i;
        }
    }

    // if (std::abs((*this)(mx_row, row)) < 1e-12) throw std::runtime_error("Matrix is singular");

    if (mx_row != row) {
        // Переставляем в текущей матрице
        for (int j = 0; j < dim; ++j) {
            swap(&(*this)(row, j), &(*this)(mx_row, j));
        }
        double temp = b[row];
        b[row] = b[mx_row];
        b[mx_row] = temp;
    }
}


// Доступ к элементам (строка, столбец)
double& Matrix::operator()(int row, int col) {
    return data[row * dim + col];
}

double Matrix::operator()(int row, int col) const {
    return data[row * dim + col];
}


// Приведение к верхнетреугольному виду
Matrix Matrix::upper_triangular(int& swaps) const {
    Matrix tr_m(*this);
    swaps = 0;

    for (int i = 0; i < dim - 1; ++i) {
        if (!tr_m.choice_leading(i, swaps)) {
            return tr_m;
        }

        for (int k = i + 1; k < dim; ++k) {
            double coeff = tr_m(k, i) / tr_m(i, i);
            for (int j = i; j < dim; ++j) {
                tr_m(k, j) -= tr_m(i, j) * coeff;
            }
        }
    }
    
    return tr_m;
}


// Определитель
double Matrix::determinant() const {
    int swaps = 0;
    Matrix tr_m = this->upper_triangular(swaps);
    double d = 1.0;
    for (int i = 0; i < dim; ++i) {
        d *= tr_m(i, i);
    }

    d *= (swaps % 2 == 0) ? 1.0 : -1.0;
    return d;
}

Matrix Matrix::inverse() const {
    Matrix augmented(dim); 
    Matrix left(*this);


    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            augmented(i, j) = (i == j) ? 1.0 : 0.0;
        }
    }

    int swaps = 0;
    for (int i = 0; i < dim; ++i) {
        // Выбор ведущего элемента
        if (!left.choice_leading(i, swaps, &augmented)) {
            throw std::runtime_error("Matrix is singular (det = 0)");
        }

        // Нормализация ведущей строки
        double divisor = left(i, i);
        for (int j = 0; j < dim; ++j) {
            left(i, j) /= divisor;
            augmented(i, j) /= divisor;
        }

        // Исключение элементов выше и ниже ведущего
        for (int k = 0; k < dim; ++k) {
            if (k != i) {
                double factor = left(k, i);
                for (int j = 0; j < dim; ++j) {
                    left(k, j) -= factor * left(i, j);
                    augmented(k, j) -= factor * augmented(i, j);
                }
            }
        }
    }

    return augmented;
}


// Транспонирование
Matrix Matrix::transpose() const {
    Matrix result(dim);
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

// Умножение матриц
Matrix Matrix::operator*(const Matrix& other) const {
    if (dim != other.dim) throw std::invalid_argument("Dimensions must match");
    
    Matrix result(dim);
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            double sum = 0;
            for (int k = 0; k < dim; ++k) {
                sum += (*this)(i, k) * other(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}

// умножение матрицы на вектор столбец
std::vector<double> Matrix::operator*(const std::vector<double> &vec) const {
    std::vector<double> res(dim);
    if (vec.size() != dim) throw std::invalid_argument("Dimensions must match"); 
    
    double sum;

    for (int i=0;i<dim;++i) {
        sum = 0;
        for (int j=0;j<dim;++j) {
        sum += vec[j] * (*this)(i,j);
        }
        res[i] = sum;
    }

    return res;
}


void Matrix::LU_decomposition(Matrix& L, Matrix& U) const{
    U = *this;
    L = Matrix(dim);

    for (int i = 0; i < dim - 1; ++i) {
        L(i,i) = 1;
        // if (std::abs(U(i, i)) < 1e-12) {
        //     throw std::runtime_error("Zero on diagonal");
        // }
        for (int k = i + 1; k < dim; ++k) {
            double coeff = U(k, i) / U(i, i);
            for (int j = i; j < dim; ++j) {
                U(k, j) -= U(i, j) * coeff;
            }
            U(k, i) = 0.0;
            L(k,i) = coeff;
        }
    }
    L(dim-1,dim-1) = 1;
}



Matrix Matrix::fill_Hilbert(int dim) {
    if (dim<=0) throw std::runtime_error("Inappropriate input");
    Matrix Hilbert(dim);
    for (int i=0;i<dim;++i) {
        for (int j=0;j<=i;++j) {
            Hilbert(i,j) = 1.0/(i+j+1);
            Hilbert(j,i) = 1.0/(i+j+1);
        }
    }
    return Hilbert;
}

Matrix Matrix::fill_random(int dim, int seed) {
    if (dim<=0) throw std::runtime_error("Inappropriate input");
    if (seed < 0) {
        std::random_device rd;
        seed = rd();
    }

    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    Matrix random(dim);
    for (int i=0;i<dim;++i) {
        for (int j=0;j<dim;++j) {
            random(i,j) = dist(gen);
        }
    }

    return random;
}

std::vector<double> fill_vector_random(int size, int seed) {
    if (size<0) throw std::runtime_error("Inappropriate input");
    if (seed < 0) {
        std::random_device rd;
        seed = rd();
    }

    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    std::vector<double> random(size);
    for (int i=0;i<size;++i) {
        random[i] = dist(gen);
    }

    return random; 
}


std::vector<double> LU_solving(const Matrix& L, const Matrix& U, const std::vector<double>& b) {
    int dim = L.get_dim();
    std::vector<double> x(dim);
    std::vector<double> y(dim);

    if (dim != U.get_dim() || dim != (int)b.size()) throw std::runtime_error("Inappropriate input");
    
    for (int i=0;i<dim;++i) {
        double y_coeff = b[i];
        for (int j = 0; j<i;++j) {
            y_coeff -= L(i,j)*y[j];
        }
        y[i] = y_coeff;
    }

    for (int i=dim-1;i>=0;--i) {
        double x_coeff = y[i];
        for (int j=dim-1;j>i;--j) {
            x_coeff -= U(i,j)*x[j];
        }
        // if (std::abs(U(i, i)) < 1e-12) {
        //     throw std::runtime_error("Zero on diagonal");
        // }
        x[i] = x_coeff / U(i, i);
    }

    return x;
} 

std::vector<double> Gauss_solving(const Matrix& A, const std::vector<double>& b) {
    int dim = A.get_dim();
    if (dim != b.size()) throw std::runtime_error("Inappropriate input");

    Matrix A_copy = A;
    std::vector<double> b_copy = b;

    for (int i = 0; i < dim - 1; ++i) {
        // if (std::abs(A_copy(i, i)) < 1e-12) {
        //     throw std::runtime_error("Zero on diagonal");
        // }
        for (int k = i + 1; k < dim; ++k) {
            double coeff = A_copy(k, i) / A_copy(i, i);
            b_copy[k] -= coeff*b_copy[i];
            for (int j = i; j < dim; ++j) {
                A_copy(k, j) -= A_copy(i, j) * coeff;
            }
            A_copy(k, i) = 0.0;
        }
    }

    std::vector<double> x(dim);

    for (int i=dim-1;i>=0;--i) {
        double x_coeff = b_copy[i];
        for (int j=dim-1;j>i;--j) {
            x_coeff -= A_copy(i,j)*x[j];
        }
        // if (std::abs(A_copy(i, i)) < 1e-12) {
        //     throw std::runtime_error("Zero on diagonal");
        // }
        x[i] = x_coeff / A_copy(i, i);
    }

    return x;
}

std::vector<double> Gauss_choice_solving(const Matrix& A, const std::vector<double>& b) {
    int dim = A.get_dim();
    if (dim != b.size()) throw std::runtime_error("Inappropriate input");

    Matrix A_copy = A;
    std::vector<double> b_copy = b;

    for (int i = 0; i < dim - 1; ++i) {
        A_copy.choice_leading(i, b_copy);
        for (int k = i + 1; k < dim; ++k) {
            double coeff = A_copy(k, i) / A_copy(i, i);
            b_copy[k] -= coeff*b_copy[i];
            for (int j = i; j < dim; ++j) {
                A_copy(k, j) -= A_copy(i, j) * coeff;
            }
            A_copy(k, i) = 0.0;
        }
    }

    std::vector<double> x(dim);

    for (int i=dim-1;i>=0;--i) {
        double x_coeff = b_copy[i];
        for (int j=dim-1;j>i;--j) {
            x_coeff -= A_copy(i,j)*x[j];
        }
        // if (std::abs(A_copy(i, i)) < 1e-12) {
        //     throw std::runtime_error("Zero on diagonal");
        // }
        x[i] = x_coeff / A_copy(i, i);
    }

    return x;
}

std::chrono::microseconds measure_Gauss_solving(const Matrix& A, const std::vector<double>& b) {
    auto start = std::chrono::steady_clock::now();
    Gauss_solving(A,b);
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start);
}

std::chrono::microseconds measure_Gauss_choice_solving(const Matrix& A, const std::vector<double>& b) {
    auto start = std::chrono::steady_clock::now();
    Gauss_choice_solving(A,b);
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start);
}

std::chrono::microseconds measure_LU_solving(const Matrix& L, const Matrix& U, const std::vector<double>& b) {
    auto start = std::chrono::steady_clock::now();
    LU_solving(L,U,b);
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start);
}

std::chrono::microseconds measure_LU_decomposition(Matrix& A, Matrix& L, Matrix& U){
    auto start = std::chrono::steady_clock::now();
    A.LU_decomposition(L,U);
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start);
}



double length(const std::vector<double>& v) {
    double sq_sum = 0;
    for (int i=0;i<v.size();++i) {
        sq_sum += v[i]*v[i];
    }
    return std::sqrt(sq_sum);
}

double fractional_error(const std::vector<double>& x_ideal, const std::vector<double>& x_answer) {
    std::vector<double> diff(x_ideal.size());
    for (int i = 0; i < x_ideal.size(); ++i) {
        diff[i] = x_answer[i] - x_ideal[i];
    }

    return length(diff) / length(x_ideal);
}

double residual(const Matrix& A, const std::vector<double>& x, const std::vector<double>& b) {
    std::vector<double> diff(x.size());
    std::vector<double> prediction = A*x;
    for (int i=0;i<x.size();++i) {
        diff[i] = prediction[i] - b[i];
    }
    return length(diff);
}