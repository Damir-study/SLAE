#ifndef CLASS_MATRIX_H
#define CLASS_MATRIX_H

#include <vector>
#include <chrono>

class Matrix {
private:
    int dim;
    double* data;

    int choice_leading(int row, int& swaps, Matrix* second_matrix = nullptr); 
    static double round_value(double value, int places = 10);

public:
    // Конструкторы и деструктор
    Matrix(int dimension);
    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);  
    ~Matrix();                                

    // Ввод / Вывод
    void input_from_console();
    void print() const;

    // Математические операции
    double determinant() const;
    Matrix transpose() const;
    Matrix operator*(const Matrix& other) const;
    std::vector<double> operator*(const std::vector<double> &vec) const;
    Matrix inverse() const;
    Matrix upper_triangular(int& swaps) const;
    void choice_leading(int row, std::vector<double>& b); 

    void LU_decomposition(Matrix& L, Matrix& U) const;
    
    // Геттер
    int get_dim() const { return dim; }
    
    // Доступ к элементам
    double& operator()(int row, int col);
    double operator()(int row, int col) const;
    
    static Matrix fill_Hilbert(int dim);
    static Matrix fill_random(int dim, int seed = -1);
};

std::vector<double> fill_vector_random(int size, int seed = -1);
std::vector<double> LU_solving(const Matrix& L, const Matrix& U, const std::vector<double>& b);
std::vector<double> Gauss_solving(const Matrix& A, const std::vector<double>& b);
std::vector<double> Gauss_choice_solving(const Matrix& A, const std::vector<double>& b);
double fractional_error(const std::vector<double>& x_ideal, const std::vector<double>&  x_answer);
double residual(const Matrix& A, const std::vector<double>& x, const std::vector<double>& b);

std::chrono::microseconds measure_Gauss_solving(const Matrix& A, const std::vector<double>& b);
std::chrono::microseconds measure_Gauss_choice_solving(const Matrix& A, const std::vector<double>& b);
std::chrono::microseconds measure_LU_solving(const Matrix& L, const Matrix& U, const std::vector<double>& b);
std::chrono::microseconds measure_LU_decomposition(Matrix& A, Matrix& L, Matrix& U);

#endif //CLASS_MATRIX_H