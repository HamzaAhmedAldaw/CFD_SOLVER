// tests/unit/test_field.cpp
#include <gtest/gtest.h>
#include "cfd/core/Field.hpp"
#include "cfd/core/Mesh.hpp"
#include <memory>

using namespace cfd;

class FieldTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple mesh with 4 cells
        mesh = std::make_unique<Mesh>();
        
        // Add vertices for 2x2 mesh
        for (int j = 0; j <= 2; ++j) {
            for (int i = 0; i <= 2; ++i) {
                mesh->addVertex(Vector3d(i, j, 0));
            }
        }
        
        // Add 4 cells
        mesh->addCell({0, 1, 4, 3});
        mesh->addCell({1, 2, 5, 4});
        mesh->addCell({3, 4, 7, 6});
        mesh->addCell({4, 5, 8, 7});
        
        // Create scalar and vector fields
        scalarField = std::make_unique<ScalarField>("pressure", mesh.get());
        vectorField = std::make_unique<VectorField>("velocity", mesh.get());
    }

    std::unique_ptr<Mesh> mesh;
    std::unique_ptr<ScalarField> scalarField;
    std::unique_ptr<VectorField> vectorField;
};

TEST_F(FieldTest, FieldInitialization) {
    EXPECT_EQ(scalarField->getName(), "pressure");
    EXPECT_EQ(scalarField->size(), mesh->getNumCells());
    
    EXPECT_EQ(vectorField->getName(), "velocity");
    EXPECT_EQ(vectorField->size(), mesh->getNumCells());
}

TEST_F(FieldTest, ScalarFieldOperations) {
    // Set uniform value
    scalarField->setValue(101325.0);
    
    for (Index i = 0; i < scalarField->size(); ++i) {
        EXPECT_DOUBLE_EQ((*scalarField)[i], 101325.0);
    }
    
    // Set individual values
    (*scalarField)[0] = 100000.0;
    (*scalarField)[1] = 101000.0;
    
    EXPECT_DOUBLE_EQ((*scalarField)[0], 100000.0);
    EXPECT_DOUBLE_EQ((*scalarField)[1], 101000.0);
}

TEST_F(FieldTest, VectorFieldOperations) {
    // Set uniform vector
    Vector3d uniformVel(1.0, 0.0, 0.0);
    vectorField->setValue(uniformVel);
    
    for (Index i = 0; i < vectorField->size(); ++i) {
        EXPECT_DOUBLE_EQ((*vectorField)[i][0], 1.0);
        EXPECT_DOUBLE_EQ((*vectorField)[i][1], 0.0);
        EXPECT_DOUBLE_EQ((*vectorField)[i][2], 0.0);
    }
    
    // Set individual vectors
    (*vectorField)[0] = Vector3d(2.0, 1.0, 0.0);
    EXPECT_DOUBLE_EQ((*vectorField)[0][0], 2.0);
    EXPECT_DOUBLE_EQ((*vectorField)[0][1], 1.0);
}

TEST_F(FieldTest, FieldArithmetic) {
    // Initialize fields
    scalarField->setValue(100.0);
    auto field2 = std::make_unique<ScalarField>("temperature", mesh.get());
    field2->setValue(300.0);
    
    // Addition
    auto sum = *scalarField + *field2;
    EXPECT_DOUBLE_EQ(sum[0], 400.0);
    
    // Scalar multiplication
    auto scaled = *scalarField * 2.0;
    EXPECT_DOUBLE_EQ(scaled[0], 200.0);
}
