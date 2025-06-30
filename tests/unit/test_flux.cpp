// tests/unit/test_flux.cpp
#include <gtest/gtest.h>
#include "cfd/numerics/FluxScheme.hpp"
#include "cfd/core/Face.hpp"
#include "cfd/core/Types.hpp"

using namespace cfd;

class FluxTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test face
        face = std::make_unique<Face>();
        face->setArea(1.0);
        face->setNormal(Vector3d(1.0, 0.0, 0.0));
        
        // Create flux schemes
        roeFlux = FluxScheme::create("Roe");
        ausmFlux = FluxScheme::create("AUSM");
        hllcFlux = FluxScheme::create("HLLC");
    }
    
    ConservativeVariables createState(Real rho, Real u, Real v, Real p) {
        ConservativeVariables state;
        Real E = p / (1.4 - 1.0) + 0.5 * rho * (u*u + v*v);
        state.rho = rho;
        state.rhoU = Vector3d(rho * u, rho * v, 0.0);
        state.rhoE = E;
        return state;
    }

    std::unique_ptr<Face> face;
    std::unique_ptr<FluxScheme> roeFlux;
    std::unique_ptr<FluxScheme> ausmFlux;
    std::unique_ptr<FluxScheme> hllcFlux;
};

TEST_F(FluxTest, RoeFluxSodProblem) {
    // Sod shock tube problem
    auto stateL = createState(1.0, 0.0, 0.0, 1.0);
    auto stateR = createState(0.125, 0.0, 0.0, 0.1);
    
    auto flux = roeFlux->computeFlux(stateL, stateR, *face);
    
    // Check mass flux
    EXPECT_GT(flux.rho, 0.0);  // Flow from left to right
    EXPECT_LT(flux.rho, 0.5);  // But not too large
}

TEST_F(FluxTest, FluxConsistency) {
    // Test that flux is consistent when states are equal
    auto state = createState(1.0, 1.0, 0.0, 101325.0);
    
    auto fluxRoe = roeFlux->computeFlux(state, state, *face);
    auto fluxAUSM = ausmFlux->computeFlux(state, state, *face);
    auto fluxHLLC = hllcFlux->computeFlux(state, state, *face);
    
    // All schemes should give same result for uniform flow
    EXPECT_NEAR(fluxRoe.rho, fluxAUSM.rho, 1e-10);
    EXPECT_NEAR(fluxRoe.rho, fluxHLLC.rho, 1e-10);
}

TEST_F(FluxTest, FluxSymmetry) {
    // Test that reversing states reverses flux
    auto stateL = createState(1.0, 1.0, 0.0, 101325.0);
    auto stateR = createState(0.5, 0.5, 0.0, 50000.0);
    
    auto fluxLR = roeFlux->computeFlux(stateL, stateR, *face);
    
    // Reverse face normal
    face->setNormal(Vector3d(-1.0, 0.0, 0.0));
    auto fluxRL = roeFlux->computeFlux(stateR, stateL, *face);
    
    EXPECT_NEAR(fluxLR.rho, -fluxRL.rho, 1e-10);
}
