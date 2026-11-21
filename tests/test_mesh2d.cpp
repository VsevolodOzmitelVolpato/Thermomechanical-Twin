#include <gtest/gtest.h>
#include "Mesh2D.h"

#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <map>
#include <string>
#include <utility>

#include "Mesh2D.h"

// ---- The 2D mesh: nodes, triangles, areas ----------------------------------------------

// EXAMPLE 1. The unit square split into two triangles by the diagonal 0-2:
//
//     3 ---------- 2        node 0 = (0, 0)     node 1 = (1, 0)
//     |          /  |        node 2 = (1, 1)     node 3 = (0, 1)
//     |   T1   /    |
//     |      /  T0  |        T0 = nodes (0, 1, 2)      T1 = nodes (0, 2, 3)
//     |    /        |
//     |  /          |
//     0 ---------- 1
//
// Area of T0: the edges leaving node 0 are (1, 0) and (1, 1), so
//     area = 1/2 * (1*1 - 1*0) = 1/2   (positive: 0 -> 1 -> 2 turns left).
// Area of T1: the edges are (1, 1) and (0, 1), so area = 1/2 * (1*1 - 0*1) = 1/2.
// Together they cover the square, which has area 1.
static Mesh2D unit_square() {
    return Mesh2D({0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}, {{0, 1, 2}, {0, 2, 3}});
}

TEST(Mesh2d, SquareInTwoTrianglesMatchesHandComputation) {
    Mesh2D m = unit_square();

    EXPECT_EQ(m.num_nodes(), 4u);
    EXPECT_EQ(m.num_triangles(), 2u);

    const double ex[4] = {0.0, 1.0, 1.0, 0.0};
    const double ey[4] = {0.0, 0.0, 1.0, 1.0};
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(m.x(i), ex[i]) << "node " << i;
        EXPECT_DOUBLE_EQ(m.y(i), ey[i]) << "node " << i;
    }

    EXPECT_EQ(m.triangle(0), (Triangle{0, 1, 2}));
    EXPECT_EQ(m.triangle(1), (Triangle{0, 2, 3}));
    EXPECT_NEAR(m.area(0), 0.5, 1e-15);
    EXPECT_NEAR(m.area(1), 0.5, 1e-15);
}

// EXAMPLE 2. An irregular quadrilateral, with the four nodes all different and the two triangles of different
// area (so it is noticed if an area is attributed to the wrong triangle):
//     node 0 = (0, 0)   node 1 = (4, 0)   node 2 = (3, 2)   node 3 = (0, 3)
//     T0 = (0, 1, 2): edges from 0 = (4, 0) e (3, 2)  ->  area = 1/2 * (4*2 - 3*0) = 4
//     T1 = (0, 2, 3): edges from 0 = (3, 2) e (0, 3)  ->  area = 1/2 * (3*3 - 0*2) = 4.5
// The quadrilateral has area 8.5: the shoelace formula, 1/2 * (0*0-4*0 + 4*2-3*0 + 3*3-0*2 + 0*0-0*3) = 17/2.
TEST(Mesh2d, IrregularQuadrilateralHasUnequalAreas) {
    Mesh2D m({0.0, 4.0, 3.0, 0.0}, {0.0, 0.0, 2.0, 3.0}, {{0, 1, 2}, {0, 2, 3}});

    EXPECT_DOUBLE_EQ(m.x(1), 4.0);
    EXPECT_DOUBLE_EQ(m.y(1), 0.0);
    EXPECT_DOUBLE_EQ(m.x(2), 3.0);
    EXPECT_DOUBLE_EQ(m.y(2), 2.0);
    EXPECT_DOUBLE_EQ(m.x(3), 0.0);
    EXPECT_DOUBLE_EQ(m.y(3), 3.0);

    EXPECT_NEAR(m.area(0), 4.0, 1e-14);
    EXPECT_NEAR(m.area(1), 4.5, 1e-14);
    EXPECT_NEAR(m.area(0) + m.area(1), 8.5, 1e-14);
}

// EXAMPLE 3. An arbitrary triangle, with nodes (1, 1), (4, 2), (2, 5): edges from 0 = (3, 1) and (1, 4),
// area = 1/2 * (3*4 - 1*1) = 5.5. The same triangle, listed starting from another vertex
// (1, 2, 0) or (2, 0, 1), still turns left and has the same area.
TEST(Mesh2d, AreaDoesNotDependOnWhichCornerComesFirst) {
    const std::vector<double> xs = {1.0, 4.0, 2.0};
    const std::vector<double> ys = {1.0, 2.0, 5.0};

    for (const Triangle& t : {Triangle{0, 1, 2}, Triangle{1, 2, 0}, Triangle{2, 0, 1}}) {
        Mesh2D m(xs, ys, std::vector<Triangle>{t});
        EXPECT_NEAR(m.area(0), 5.5, 1e-14) << "triangle (" << t[0] << ", " << t[1] << ", " << t[2] << ")";
    }
}

// The constructor rules. We start from a valid triangle, (0, 0) (1, 0) (0, 1), and break one
// rule at a time.
TEST(Mesh2d, WrongInputsThrow) {
    const std::vector<double> xs = {0.0, 1.0, 0.0};
    const std::vector<double> ys = {0.0, 0.0, 1.0};
    const std::vector<Triangle> good = {{0, 1, 2}};
    EXPECT_NO_THROW(Mesh2D(xs, ys, good));

    // rule 1: xs and ys with different numbers of values. The triangle {0, 1, 2} only uses nodes that
    // exist in both vectors and has a good area: the only broken rule is number 1.
    EXPECT_THROW(Mesh2D(xs, std::vector<double>{0.0, 0.0, 1.0, 5.0}, good), std::invalid_argument);
    EXPECT_THROW(Mesh2D(std::vector<double>{0.0, 1.0, 0.0, 5.0}, ys, good), std::invalid_argument);

    // rule 2: no triangle
    EXPECT_THROW(Mesh2D(xs, ys, std::vector<Triangle>{}), std::invalid_argument);

    // rule 3: a node that does not exist. There are 3 nodes (0, 1, 2): number 3 is already outside.
    // It is tried in each of the three positions of the triangle.
    EXPECT_THROW(Mesh2D(xs, ys, std::vector<Triangle>{{3, 1, 2}}), std::invalid_argument);
    EXPECT_THROW(Mesh2D(xs, ys, std::vector<Triangle>{{0, 3, 2}}), std::invalid_argument);
    EXPECT_THROW(Mesh2D(xs, ys, std::vector<Triangle>{{0, 1, 3}}), std::invalid_argument);
    EXPECT_THROW(Mesh2D(xs, ys, std::vector<Triangle>{{0, 1, 99}}), std::invalid_argument);

    // rule 4: clockwise triangle. (0, 2, 1) = (0, 0) (0, 1) (1, 0): 1/2 * (0*0 - 1*1) = -1/2.
    EXPECT_THROW(Mesh2D(xs, ys, std::vector<Triangle>{{0, 2, 1}}), std::invalid_argument);
    // rule 4: repeated node (area 0)
    EXPECT_THROW(Mesh2D(xs, ys, std::vector<Triangle>{{0, 0, 1}}), std::invalid_argument);
    // rule 4: three collinear nodes, (0, 0) (1, 1) (2, 2): area 0
    EXPECT_THROW(Mesh2D(std::vector<double>{0.0, 1.0, 2.0}, std::vector<double>{0.0, 1.0, 2.0}, good),
                 std::invalid_argument);
    // rule 4: a NaN coordinate (it is neither > 0 nor <= 0: it must be rejected all the same)
    EXPECT_THROW(Mesh2D(std::vector<double>{0.0, std::nan(""), 0.0}, ys, good), std::invalid_argument);

    // ALL the triangles must be checked, not only the first: on the unit square
    // the second triangle (0, 3, 2) turns right.
    EXPECT_NO_THROW(Mesh2D({0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}, {{0, 1, 2}, {0, 2, 3}}));
    EXPECT_THROW(Mesh2D({0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}, {{0, 1, 2}, {0, 3, 2}}),
                 std::invalid_argument);
    // and the second triangle with a node that does not exist
    EXPECT_THROW(Mesh2D({0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}, {{0, 1, 2}, {0, 2, 4}}),
                 std::invalid_argument);
}

// With at() a wrong index does not silently read out of bounds: it throws std::out_of_range.
// (It is the kind of error that made the tests crash in element_stress.)
TEST(Mesh2d, AccessorsCheckTheirBounds) {
    Mesh2D m = unit_square();   // 4 nodes, 2 triangles

    EXPECT_NO_THROW(m.x(3));
    EXPECT_NO_THROW(m.y(3));
    EXPECT_NO_THROW(m.triangle(1));
    EXPECT_NO_THROW(m.area(1));

    EXPECT_THROW(m.x(4), std::out_of_range);
    EXPECT_THROW(m.y(4), std::out_of_range);
    EXPECT_THROW(m.triangle(2), std::out_of_range);
    EXPECT_THROW(m.area(2), std::out_of_range);
}

// ---- Structured rectangular mesh: rectangle_mesh ----------------------------------------------
//
// rectangle_mesh(Lx, Ly, nx, ny) divides the rectangle [0, Lx] x [0, Ly] into nx x ny equal cells, and
// each cell into 2 triangles, with the diagonal from the bottom-left corner to the top-right
// one. The numbering rules, for any nx and ny:
//   - nodes: row by row, from bottom to top; in each row from left to right. The node in
//     column i (from 0 to nx) and row j (from 0 to ny) has number  j*(nx+1) + i  and coordinates
//     (i*Lx/nx, j*Ly/ny);
//   - cells: in the same order (row by row, from left to right). The corners of a cell are
//     a = bottom left, b = bottom right, c = top right, d = top left;
//   - triangles: cell number k gives triangles number 2k = (a, b, c) and number 2k+1 = (a, c, d).

// EXAMPLE 4. Two cells horizontally, one vertically: Lx = 2, Ly = 1, nx = 2, ny = 1.
//
//     3 --------- 4 --------- 5
//     |        /  |        /  |
//     |  T1  /    |  T3  /    |
//     |    /  T0  |    /  T2  |
//     |  /        |  /        |
//     0 --------- 1 --------- 2
//
// Nodes: 0 = (0, 0)  1 = (1, 0)  2 = (2, 0)  3 = (0, 1)  4 = (1, 1)  5 = (2, 1).
// Cell 0: (a, b, c, d) = (0, 1, 4, 3), so T0 = (0, 1, 4) and T1 = (0, 4, 3).
// Cell 1: (a, b, c, d) = (1, 2, 5, 4), so T2 = (1, 2, 5) and T3 = (1, 5, 4).
// Each triangle is half of a square of side 1, so it has area 1/2; in total 4 * 1/2 = 2 = Lx*Ly.
TEST(Mesh2d, RectangleTwoByOneMatchesHandComputation) {
    Mesh2D m = rectangle_mesh(2.0, 1.0, 2, 1);

    EXPECT_EQ(m.num_nodes(), 6u);
    EXPECT_EQ(m.num_triangles(), 4u);

    const double ex[6] = {0.0, 1.0, 2.0, 0.0, 1.0, 2.0};
    const double ey[6] = {0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    for (std::size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(m.x(i), ex[i], 1e-14) << "node " << i;
        EXPECT_NEAR(m.y(i), ey[i], 1e-14) << "node " << i;
    }

    EXPECT_EQ(m.triangle(0), (Triangle{0, 1, 4}));
    EXPECT_EQ(m.triangle(1), (Triangle{0, 4, 3}));
    EXPECT_EQ(m.triangle(2), (Triangle{1, 2, 5}));
    EXPECT_EQ(m.triangle(3), (Triangle{1, 5, 4}));

    for (std::size_t e = 0; e < 4; ++e) {
        EXPECT_NEAR(m.area(e), 0.5, 1e-14) << "triangle " << e;
    }
}

// EXAMPLE 5. Non-square cells and nx different from ny: Lx = 3, Ly = 1, nx = 3, ny = 2.
// The grid steps are dx = 3/3 = 1 and dy = 1/2 = 0.5. Each row has nx+1 = 4 nodes, so the node
// in column i and row j is number 4j + i:
//
//     8 --------- 9 -------- 10 -------- 11        row j = 2   y = 1
//     |        /  |        /  |        /  |
//     |  T7  /    |  T9  /    | T11  /    |
//     |    /  T6  |    /  T8  |    /  T10 |
//     |  /        |  /        |  /        |
//     4 --------- 5 --------- 6 --------- 7        row j = 1   y = 0.5
//     |        /  |        /  |        /  |
//     |  T1  /    |  T3  /    |  T5  /    |
//     |    /  T0  |    /  T2  |    /  T4  |
//     |  /        |  /        |  /        |
//     0 --------- 1 --------- 2 --------- 3        row j = 0   y = 0
//
// There are 6 cells, row by row. Cell 3 (column 0, row 1) has (a, b, c, d) = (4, 5, 9, 8),
// so T6 = (4, 5, 9) and T7 = (4, 9, 8). The last one, cell 5, has (6, 7, 11, 10), so
// T10 = (6, 7, 11) and T11 = (6, 11, 10). Each triangle has area 1/2 * 1 * 0.5 = 0.25.
TEST(Mesh2d, RectangleWithUnequalSidesPlacesTheNodesCorrectly) {
    Mesh2D m = rectangle_mesh(3.0, 1.0, 3, 2);

    EXPECT_EQ(m.num_nodes(), 12u);
    EXPECT_EQ(m.num_triangles(), 12u);

    const double ex[12] = {0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0};
    const double ey[12] = {0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0};
    for (std::size_t i = 0; i < 12; ++i) {
        EXPECT_NEAR(m.x(i), ex[i], 1e-14) << "node " << i;
        EXPECT_NEAR(m.y(i), ey[i], 1e-14) << "node " << i;
    }

    EXPECT_EQ(m.triangle(0), (Triangle{0, 1, 5}));
    EXPECT_EQ(m.triangle(1), (Triangle{0, 5, 4}));
    EXPECT_EQ(m.triangle(6), (Triangle{4, 5, 9}));
    EXPECT_EQ(m.triangle(7), (Triangle{4, 9, 8}));
    EXPECT_EQ(m.triangle(10), (Triangle{6, 7, 11}));
    EXPECT_EQ(m.triangle(11), (Triangle{6, 11, 10}));

    for (std::size_t e = 0; e < 12; ++e) {
        EXPECT_NEAR(m.area(e), 0.25, 1e-14) << "triangle " << e;
    }
}

// Count, for each side (the pair of its two nodes, in increasing order), in how many triangles it appears.
static std::map<std::pair<std::size_t, std::size_t>, int> edge_counts(const Mesh2D& m) {
    std::map<std::pair<std::size_t, std::size_t>, int> counts;
    for (std::size_t e = 0; e < m.num_triangles(); ++e) {
        const Triangle& t = m.triangle(e);
        for (std::size_t k = 0; k < 3; ++k) {
            std::size_t a = t[k];
            std::size_t b = t[(k + 1) % 3];
            if (a > b) {
                std::swap(a, b);
            }
            ++counts[{a, b}];
        }
    }
    return counts;
}

// EXAMPLE 6. Things that must hold for ANY rectangle and ANY number of cells, even where
// not everything can be computed by hand. With dx = Lx/nx and dy = Ly/ny:
//   - the nodes are (nx+1)(ny+1) and the triangles 2 nx ny;
//   - each triangle is half a cell: area 1/2 dx dy. So together they cover the rectangle
//     with no holes and no overlaps: sum of the areas = Lx Ly;
//   - the nodes lie inside the rectangle and touch all four of its sides;
//   - the mesh is "conforming": each side belongs to 2 triangles (interior side) or to 1 (side on the
//     boundary), never to 3 or more. The boundary sides are 2 (nx + ny); the sides in total are
//     nx (ny+1) horizontal + ny (nx+1) vertical + nx ny diagonal = 3 nx ny + nx + ny.
TEST(Mesh2d, RectangleCoversTheDomainExactlyOnce) {
    struct Case {
        double lx;
        double ly;
        std::size_t nx;
        std::size_t ny;
    };
    const Case cases[] = {{1.0, 1.0, 1, 1}, {1.0, 1.0, 4, 4}, {2.5, 0.5, 5, 3}, {0.1, 7.0, 1, 9}, {3.0, 2.0, 7, 1}};

    for (const Case& c : cases) {
        SCOPED_TRACE("nx = " + std::to_string(c.nx) + ", ny = " + std::to_string(c.ny));
        Mesh2D m = rectangle_mesh(c.lx, c.ly, c.nx, c.ny);
        const double dx = c.lx / static_cast<double>(c.nx);
        const double dy = c.ly / static_cast<double>(c.ny);

        EXPECT_EQ(m.num_nodes(), (c.nx + 1) * (c.ny + 1));
        EXPECT_EQ(m.num_triangles(), 2 * c.nx * c.ny);

        double total_area = 0.0;
        for (std::size_t e = 0; e < m.num_triangles(); ++e) {
            EXPECT_NEAR(m.area(e), 0.5 * dx * dy, 1e-12 * dx * dy) << "triangle " << e;
            total_area += m.area(e);
        }
        EXPECT_NEAR(total_area, c.lx * c.ly, 1e-12 * c.lx * c.ly);

        double xmin = m.x(0);
        double xmax = m.x(0);
        double ymin = m.y(0);
        double ymax = m.y(0);
        for (std::size_t n = 1; n < m.num_nodes(); ++n) {
            xmin = std::min(xmin, m.x(n));
            xmax = std::max(xmax, m.x(n));
            ymin = std::min(ymin, m.y(n));
            ymax = std::max(ymax, m.y(n));
        }
        EXPECT_NEAR(xmin, 0.0, 1e-12 * c.lx);
        EXPECT_NEAR(xmax, c.lx, 1e-12 * c.lx);
        EXPECT_NEAR(ymin, 0.0, 1e-12 * c.ly);
        EXPECT_NEAR(ymax, c.ly, 1e-12 * c.ly);

        const auto counts = edge_counts(m);
        std::size_t boundary_edges = 0;
        for (const auto& entry : counts) {
            const int times = entry.second;
            EXPECT_TRUE(times == 1 || times == 2) << "the side (" << entry.first.first << ", "
                                                   << entry.first.second << ") is in " << times << " triangles";
            if (times == 1) {
                ++boundary_edges;
            }
        }
        EXPECT_EQ(boundary_edges, 2 * (c.nx + c.ny));
        EXPECT_EQ(counts.size(), 3 * c.nx * c.ny + c.nx + c.ny);
    }
}

// Wrong inputs: non-positive sides or NaN, no cells.
TEST(Mesh2d, RectangleWrongInputsThrow) {
    EXPECT_NO_THROW(rectangle_mesh(1.0, 1.0, 1, 1));

    EXPECT_THROW(rectangle_mesh(0.0, 1.0, 2, 2), std::invalid_argument);
    EXPECT_THROW(rectangle_mesh(1.0, 0.0, 2, 2), std::invalid_argument);
    EXPECT_THROW(rectangle_mesh(-1.0, 1.0, 2, 2), std::invalid_argument);
    EXPECT_THROW(rectangle_mesh(1.0, -1.0, 2, 2), std::invalid_argument);
    // Two negative sides together would give a "valid" mesh (in the third quadrant), with all the
    // triangles counter-clockwise. But a rectangle with negative sides makes no sense: it must be rejected.
    EXPECT_THROW(rectangle_mesh(-1.0, -1.0, 2, 2), std::invalid_argument);
    EXPECT_THROW(rectangle_mesh(std::nan(""), 1.0, 2, 2), std::invalid_argument);
    EXPECT_THROW(rectangle_mesh(1.0, std::nan(""), 2, 2), std::invalid_argument);

    EXPECT_THROW(rectangle_mesh(1.0, 1.0, 0, 2), std::invalid_argument);
    EXPECT_THROW(rectangle_mesh(1.0, 1.0, 2, 0), std::invalid_argument);
}

// ---- Boundary nodes: boundary_nodes ------------------------------------------------------------
//
// boundary_nodes(mesh) returns the numbers of the nodes that lie on the boundary of the mesh, in
// increasing order and each one only once. A node is a boundary node if it is an endpoint of a boundary side, and a
// side (the side of a triangle) is a boundary side if it belongs to ONE SINGLE triangle: an interior side has a
// triangle on each side, so it has two. The criterion does not look at the coordinates, so it holds for
// any shape, even with holes.

// EXAMPLE 7. The unit square of EXAMPLE 1, by hand. The sides of the two triangles T0 = (0, 1, 2) and
// T1 = (0, 2, 3), and in how many triangles they appear:
//     (0, 1): 1     (1, 2): 1     (0, 2): 2  <- the diagonal is interior     (2, 3): 1     (0, 3): 1
// The boundary sides are the four with 1, and they touch nodes 0, 1, 2, 3: all of them are boundary nodes.
//
// In the rectangle meshes a node is interior if it is not on the perimeter. With 2 x 2 cells (9 nodes,
// numbered row by row) the only interior one is node 4, at the center. With 3 x 2 cells (12 nodes) the
// interior ones are 5 and 6, i.e. columns 1 and 2 of row 1.
TEST(Mesh2d, BoundaryNodesOfSmallMeshesMatchHandComputation) {
    EXPECT_EQ(boundary_nodes(unit_square()), (std::vector<std::size_t>{0, 1, 2, 3}));

    EXPECT_EQ(boundary_nodes(rectangle_mesh(1.0, 1.0, 2, 2)), (std::vector<std::size_t>{0, 1, 2, 3, 5, 6, 7, 8}));

    EXPECT_EQ(boundary_nodes(rectangle_mesh(3.0, 1.0, 3, 2)),
              (std::vector<std::size_t>{0, 1, 2, 3, 4, 7, 8, 9, 10, 11}));
}

// EXAMPLE 8. A triangle with a node inside. It is not a rectangle, so no shortcuts with the
// coordinates. Nodes: 0 = (0, 0), 1 = (3, 0), 2 = (0, 3) and 3 = (1, 1), which is inside. Three triangles
// around node 3: (0, 1, 3), (1, 2, 3), (2, 0, 3), each of area 3/2 (in total 9/2, like the
// large triangle). The sides (0, 1), (1, 2), (2, 0) are in one triangle only; the three sides leaving
// node 3 are in two. So the boundary nodes are 0, 1, 2 and node 3 is interior.
TEST(Mesh2d, BoundaryNodesOfATriangleWithAnInteriorNode) {
    Mesh2D m({0.0, 3.0, 0.0, 1.0}, {0.0, 0.0, 3.0, 1.0}, {{0, 1, 3}, {1, 2, 3}, {2, 0, 3}});

    EXPECT_NEAR(m.area(0) + m.area(1) + m.area(2), 4.5, 1e-14);   // the mesh is the right one
    EXPECT_EQ(boundary_nodes(m), (std::vector<std::size_t>{0, 1, 2}));
}

// EXAMPLE 9. A 3 x 3 square with a 1 x 1 square hole at the center: nodes 0..3 are on the outer
// square and 4..7 on the inner one, both numbered counter-clockwise.
//
//     3 ----------------- 2
//     | \               / |
//     |   7 --------- 6   |
//     |   |           |   |
//     |   |   (hole)  |   |
//     |   |           |   |
//     |   4 --------- 5   |
//     | /               \ |
//     0 ----------------- 1
//
// The ring is made of 4 trapezoids, each split into 2 triangles: (0, 1, 5) (0, 5, 4) - (1, 2, 6)
// (1, 6, 5) - (2, 3, 7) (2, 7, 6) - (3, 0, 4) (3, 4, 7). The total area is 9 - 1 = 8. The sides
// of the hole also have a single triangle (there is nothing on the other side): nodes 4, 5, 6, 7 are
// boundary nodes like the others.
TEST(Mesh2d, BoundaryNodesIncludeTheOnesAroundAHole) {
    Mesh2D m({0.0, 3.0, 3.0, 0.0, 1.0, 2.0, 2.0, 1.0}, {0.0, 0.0, 3.0, 3.0, 1.0, 1.0, 2.0, 2.0},
             {{0, 1, 5}, {0, 5, 4}, {1, 2, 6}, {1, 6, 5}, {2, 3, 7}, {2, 7, 6}, {3, 0, 4}, {3, 4, 7}});

    double total_area = 0.0;
    for (std::size_t e = 0; e < m.num_triangles(); ++e) {
        total_area += m.area(e);
    }
    EXPECT_NEAR(total_area, 8.0, 1e-14);   // the mesh is the right one

    EXPECT_EQ(boundary_nodes(m), (std::vector<std::size_t>{0, 1, 2, 3, 4, 5, 6, 7}));
}

// EXAMPLE 10. On rectangles there is a second, independent way to know which nodes are on the boundary:
// those that touch one of the four sides, i.e. with x = 0, x = Lx, y = 0 or y = Ly. The result
// of boundary_nodes, which looks only at the triangles, must coincide with this for any size.
// In addition: the boundary nodes of nx x ny cells are 2 (nx + ny), and the list is in strictly
// increasing order (hence without repetitions).
TEST(Mesh2d, BoundaryNodesMatchTheGeometryOfRectangles) {
    struct Case {
        double lx;
        double ly;
        std::size_t nx;
        std::size_t ny;
    };
    const Case cases[] = {{1.0, 1.0, 1, 1}, {1.0, 1.0, 4, 4}, {2.5, 0.5, 5, 3},
                          {0.1, 7.0, 1, 9}, {3.0, 2.0, 7, 1}, {2.0, 2.0, 6, 6}};

    for (const Case& c : cases) {
        SCOPED_TRACE("nx = " + std::to_string(c.nx) + ", ny = " + std::to_string(c.ny));
        Mesh2D m = rectangle_mesh(c.lx, c.ly, c.nx, c.ny);
        const std::vector<std::size_t> found = boundary_nodes(m);

        const double tol_x = 1e-9 * c.lx;
        const double tol_y = 1e-9 * c.ly;
        std::vector<std::size_t> expected;
        for (std::size_t n = 0; n < m.num_nodes(); ++n) {
            const bool on_a_vertical_side = std::fabs(m.x(n)) < tol_x || std::fabs(m.x(n) - c.lx) < tol_x;
            const bool on_a_horizontal_side = std::fabs(m.y(n)) < tol_y || std::fabs(m.y(n) - c.ly) < tol_y;
            if (on_a_vertical_side || on_a_horizontal_side) {
                expected.push_back(n);
            }
        }

        EXPECT_EQ(found, expected);
        EXPECT_EQ(found.size(), 2 * (c.nx + c.ny));
        for (std::size_t k = 1; k < found.size(); ++k) {
            EXPECT_LT(found[k - 1], found[k]) << "position " << k;
        }
    }
}
