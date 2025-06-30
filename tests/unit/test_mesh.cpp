// tests/unit/test_mesh.cpp
#include <gtest/gtest.h>
#include "cfd/core/Mesh.hpp"
#include "cfd/core/Cell.hpp"
#include "cfd/core/Face.hpp"
#include <memory>

using namespace cfd;

class MeshTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple 2x2 structured mesh
        mesh = std::make_unique<Mesh>();
        
        // Add vertices
        mesh->addVertex(Vector3d(0.0, 0.0, 0.0)); // 0
        mesh->addVertex(Vector3d(1.0, 0.0, 0.0)); // 1
        mesh->addVertex(Vector3d(2.0, 0.0, 0.0)); // 2
        mesh->addVertex(Vector3d(0.0, 1.0, 0.0)); // 3
        mesh->addVertex(Vector3d(1.0, 1.0, 0.0)); // 4
        mesh->addVertex(Vector3d(2.0, 1.0, 0.0)); // 5
        mesh->addVertex(Vector3d(0.0, 2.0, 0.0)); // 6
        mesh->addVertex(Vector3d(1.0, 2.0, 0.0)); // 7
        mesh->addVertex(Vector3d(2.0, 2.0, 0.0)); // 8
    }

    std::unique_ptr<Mesh> mesh;
};

TEST_F(MeshTest, VertexCount) {
    EXPECT_EQ(mesh->getNumVertices(), 9);
}

TEST_F(MeshTest, VertexCoordinates) {
    auto v0 = mesh->getVertex(0);
    EXPECT_DOUBLE_EQ(v0[0], 0.0);
    EXPECT_DOUBLE_EQ(v0[1], 0.0);
    EXPECT_DOUBLE_EQ(v0[2], 0.0);
    
    auto v8 = mesh->getVertex(8);
    EXPECT_DOUBLE_EQ(v8[0], 2.0);
    EXPECT_DOUBLE_EQ(v8[1], 2.0);
    EXPECT_DOUBLE_EQ(v8[2], 0.0);
}

TEST_F(MeshTest, CellCreation) {
    // Create cells (quadrilaterals)
    std::vector<Index> cell0 = {0, 1, 4, 3};
    std::vector<Index> cell1 = {1, 2, 5, 4};
    std::vector<Index> cell2 = {3, 4, 7, 6};
    std::vector<Index> cell3 = {4, 5, 8, 7};
    
    mesh->addCell(cell0);
    mesh->addCell(cell1);
    mesh->addCell(cell2);
    mesh->addCell(cell3);
    
    EXPECT_EQ(mesh->getNumCells(), 4);
}

TEST_F(MeshTest, BoundingBox) {
    auto bbox = mesh->getBoundingBox();
    EXPECT_DOUBLE_EQ(bbox.min()[0], 0.0);
    EXPECT_DOUBLE_EQ(bbox.min()[1], 0.0);
    EXPECT_DOUBLE_EQ(bbox.max()[0], 2.0);
    EXPECT_DOUBLE_EQ(bbox.max()[1], 2.0);
}
