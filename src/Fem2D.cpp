#include "Fem2D.h"
#include "Mesh2D.h"

Matrix3 element_stiffness(const Mesh2D &M, size_t e, double k){
    Triangle t = M.triangle(e);
    double area = M.area(e);

    std::array<double, 3> x{}, y{}, b{}, c{};

    for( size_t i {0}; i < 3; ++i){
        x[i] = M.x(t[i]);
        y[i] = M.y(t[i]);
    }

    for( size_t i {0}; i < 3; ++i){
        size_t p = (i + 1) % 3;
        size_t q = (i + 2) % 3;

        b[i] = y[p] - y[q];
        c[i] = x[q] - x[p];
    }

    Matrix3 K{};

    for( size_t i {0}; i < 3; ++i){
        for( size_t j {0}; j < 3; ++j){
            K[i][j] = k / (4.0 * area) * (b[i] * b[j] + c[i] * c[j]);}}

    return K;
}

Matrix3 element_mass(const Mesh2D &M, size_t e, double rho_c){
    double area = M.area(e);

    Matrix3 matrix{};

    for( size_t i {0}; i < 3; ++i){
        for( size_t j {0}; j < 3; ++j){
            matrix[i][j] = (i == j) ? (rho_c * area / 6.0) : (rho_c * area / 12.0);}}

    return matrix;
}

Vector3 element_load(const Mesh2D &M, size_t e, double q){
    double area = M.area(e);

    Vector3 vect{};
    vect.fill(q * area / 3.0);

    return vect;
}

Vector3 element_load(const Mesh2D &M, std::size_t e, const std::function<double(double, double)> &q){
    Triangle t = M.triangle(e);
    double area = M.area(e);

    std::array<double, 3> q_mid{};
    Vector3 f{};

    for(std::size_t i {0}; i < 3; ++i){
        std::size_t a = t[i];
        std::size_t b = t[(i + 1) % 3];

        double x = (M.x(a) + M.x(b)) * 0.5;
        double y = (M.y(a) + M.y(b)) * 0.5;

        q_mid[i] = q(x, y);}

    for(std::size_t i {0}; i < 3; ++i){
        f[i] = area / 6.0 * (q_mid[i] + q_mid[(i + 2) % 3]);}

    return f;
}