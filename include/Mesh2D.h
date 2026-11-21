#ifndef MESH2D_H
#define MESH2D_H

#include <array>
#include <cstddef>
#include <vector>

// A triangle is the triple of the numbers of its 3 nodes (positions in the list of nodes).
using Triangle = std::array<std::size_t, 3>;

// 2D mesh of triangles: the nodes, each with its coordinates (x, y), and the triangles.
//
// Unlike ThermoMechanicalProblem, here the data must obey rules that ALWAYS hold. That is why this
// is a class: the data are private, only the constructor sets them, and it does so after checking
// the rules. Whoever uses the mesh (assembly, boundary conditions...) can rely on it without
// checking again.
//   1. xs and ys have the same number of values: node i is at (xs[i], ys[i]);
//   2. there is at least one triangle;
//   3. the 3 nodes of every triangle exist (numbers smaller than the number of nodes);
//   4. every triangle is counter-clockwise and not degenerate, i.e. its signed area is > 0:
//          area = 1/2 * [ (x1 - x0) (y2 - y0) - (x2 - x0) (y1 - y0) ]
//      The sign is + if nodes 0, 1, 2 follow each other turning left, - if they turn right,
//      and the area is 0 if the three nodes are aligned (or if two of them coincide).
class Mesh2D {
private:
    std::vector<double> x_;
    std::vector<double> y_;
    std::vector<Triangle> triangles_;
    std::vector<double> areas_;   // areas_[e] = area of triangle e, computed once in the constructor
    
public:
    Mesh2D(const std::vector<double>& xs, const std::vector<double>& ys,
           const std::vector<Triangle>& triangles);

    std::size_t num_nodes() const;
    std::size_t num_triangles() const;

    // Access with index check: they throw std::out_of_range if node (or e) does not exist.
    double x(std::size_t node) const;                 // x coordinate of the node
    double y(std::size_t node) const;                 // y coordinate of the node
    const Triangle& triangle(std::size_t e) const;    // the 3 nodes of triangle e
    double area(std::size_t e) const;                 // area of triangle e (always > 0)

};

Mesh2D rectangle_mesh(double xl, double yl, size_t x, size_t y);

std::vector<size_t> boundary_nodes(const Mesh2D &M);

#endif
