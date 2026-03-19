#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include "classMatrix.h"

void test_single_system();
void test_multiple_rhs();
void test_ill_conditioned();

void print_menu() {
    std::cout << "1) Single system time comparison\n";
    std::cout << "2) Time saving for multiple right-hand sides\n";
    std::cout << "3) Accuracy on ill-conditioned matrices\n";
    std::cout << "0) Exit\n";
    std::cout << "Your choice: ";
}

void test_single_system() {
    std::cout << "\nSingle system time comparison\n\n";

    std::vector<int> sizes = {100, 200, 500, 1000};

    std::cout << std::setw(8) << "n"
              << std::setw(18) << "Gauss"
              << std::setw(18) << "Gauss pivot"
              << std::setw(18) << "LU total"
              << std::setw(18) << "LU decomp"
              << std::setw(18) << "LU solve"
              << "\n";

    for (int n : sizes) {
        Matrix A = Matrix::fill_random(n, 42 + n);
        std::vector<double> b = fill_vector_random(n, 100 + n);

        long long gauss_time = -1;
        long long gauss_pivot_time = -1;
        long long lu_decomp_time = -1;
        long long lu_solve_time = -1;
        long long lu_total_time = -1;

        try {
            gauss_time = measure_Gauss_solving(A, b).count();
        }
        catch (...) {
        }

        try {
            gauss_pivot_time = measure_Gauss_choice_solving(A, b).count();
        }
        catch (...) {}

        try {
            Matrix L(n), U(n);
            lu_decomp_time = measure_LU_decomposition(A, L, U).count();
            lu_solve_time = measure_LU_solving(L, U, b).count();
            lu_total_time = lu_decomp_time + lu_solve_time;
        }
        catch (...) {
        }

        std::cout << std::setw(8) << n;

        if (gauss_time >= 0) std::cout << std::setw(18) << gauss_time;
        else std::cout << std::setw(18) << "FAILED";

        if (gauss_pivot_time >= 0) std::cout << std::setw(18) << gauss_pivot_time;
        else std::cout << std::setw(18) << "FAILED";

        if (lu_total_time >= 0) std::cout << std::setw(18) << lu_total_time;
        else std::cout << std::setw(18) << "FAILED";

        if (lu_decomp_time >= 0) std::cout << std::setw(18) << lu_decomp_time;
        else std::cout << std::setw(18) << "FAILED";

        if (lu_solve_time >= 0) std::cout << std::setw(18) << lu_solve_time;
        else std::cout << std::setw(18) << "FAILED";

        std::cout << "\n";
    }

    std::cout << "\n";
}

void test_multiple_rhs() {
    std::cout << "\nTime saving for multiple right-hand sides\n\n";

    int n = 500;
    std::vector<int> ks = {1, 10, 100};

    Matrix A = Matrix::fill_random(n, 42);

    std::cout << std::setw(8) << "k"
              << std::setw(22) << "Gauss pivot"
              << std::setw(22) << "LU total"
              << std::setw(22) << "LU decomposition"
              << std::setw(22) << "LU solving"
              << "\n";

    for (int k : ks) {
        long long gauss_time = 0;
        long long lu_decomp_time = -1;
        long long lu_solve_time = 0;
        long long lu_total_time = -1;

        std::vector<std::vector<double>> bs;
        for (int i = 0; i < k; ++i) {
            bs.push_back(fill_vector_random(n, 1000 + i));
        }

        try {
            for (int i = 0; i < k; ++i) {
                gauss_time += measure_Gauss_choice_solving(A, bs[i]).count();
            }
        }
        catch (...) {
            gauss_time = -1;
        }

        try {
            Matrix L(n), U(n);
            lu_decomp_time = measure_LU_decomposition(A, L, U).count();

            for (int i = 0; i < k; ++i) {
                lu_solve_time += measure_LU_solving(L, U, bs[i]).count();
            }

            lu_total_time = lu_decomp_time + lu_solve_time;
        }
        catch (...) {
            lu_decomp_time = -1;
            lu_solve_time = -1;
            lu_total_time = -1;
        }

        std::cout << std::setw(8) << k;

        if (gauss_time >= 0) std::cout << std::setw(22) << gauss_time;
        else std::cout << std::setw(22) << "FAILED";

        if (lu_total_time >= 0) std::cout << std::setw(22) << lu_total_time;
        else std::cout << std::setw(22) << "FAILED";

        if (lu_decomp_time >= 0) std::cout << std::setw(22) << lu_decomp_time;
        else std::cout << std::setw(22) << "FAILED";

        if (lu_solve_time >= 0) std::cout << std::setw(22) << lu_solve_time;
        else std::cout << std::setw(22) << "FAILED";

        std::cout << "\n";
    }

    std::cout << "\n";
}

void test_ill_conditioned() {
    std::cout << std::scientific << std::setprecision(10);

    std::cout << "\nAccuracy on ill-conditioned matrices\n\n";

    std::vector<int> sizes = {5, 10, 15};

    std::cout << std::setw(8)  << "n"
              << std::setw(20) << "Method"
              << std::setw(22) << "Relative error"
              << std::setw(22) << "Residual"
              << "\n";

    for (int n : sizes) {
        Matrix H = Matrix::fill_Hilbert(n);
        std::vector<double> x_ideal(n, 1.0);
        std::vector<double> b = H * x_ideal;

        try {
            std::vector<double> x = Gauss_solving(H, b);

            std::cout << std::setw(8)  << n
                      << std::setw(20) << "Gauss"
                      << std::setw(22) << fractional_error(x_ideal, x)
                      << std::setw(22) << residual(H, x, b)
                      << "\n";
        }
        catch (...) {
            std::cout << std::setw(8)  << n
                      << std::setw(20) << "Gauss"
                      << std::setw(22) << "FAILED"
                      << std::setw(22) << "FAILED"
                      << "\n";
        }

        try {
            std::vector<double> x = Gauss_choice_solving(H, b);

            std::cout << std::setw(8)  << ""
                      << std::setw(20) << "Gauss pivot"
                      << std::setw(22) << fractional_error(x_ideal, x)
                      << std::setw(22) << residual(H, x, b)
                      << "\n";
        }
        catch (...) {
            std::cout << std::setw(8)  << ""
                      << std::setw(20) << "Gauss pivot"
                      << std::setw(22) << "FAILED"
                      << std::setw(22) << "FAILED"
                      << "\n";
        }

        // LU
        try {
            Matrix L(n), U(n);
            H.LU_decomposition(L, U);
            std::vector<double> x = LU_solving(L, U, b);

            std::cout << std::setw(8)  << ""
                      << std::setw(20) << "LU"
                      << std::setw(22) << fractional_error(x_ideal, x)
                      << std::setw(22) << residual(H, x, b)
                      << "\n";
        }
        catch (...) {
            std::cout << std::setw(8)  << ""
                      << std::setw(20) << "LU"
                      << std::setw(22) << "FAILED"
                      << std::setw(22) << "FAILED"
                      << "\n";
        }

        std::cout << "\n";
    }
}

int main() {
    int choice;

    while (true) {
        print_menu();
        std::cin >> choice;

        if (choice == 0) {
            std::cout << "Exit.\n";
            break;
        }
        else if (choice == 1) {
            test_single_system();
        }
        else if (choice == 2) {
            test_multiple_rhs();
        }
        else if (choice == 3) {
            test_ill_conditioned();
        }
        else {
            std::cout << "Wrong choice.\n\n";
        }
    }

    return 0;
}