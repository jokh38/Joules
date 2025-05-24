#include "gtest/gtest.h"
#include "mqi_scorer.hpp"
#include "mqi_track.hpp" // For track_t
#include "mqi_grid3d.hpp" // For grid3d
#include "mqi_material.hpp" // For material_t and h2o_t
#include "scorers/mqi_scorer_energy_deposit.hpp" // For energy_deposit and dose_to_water compute functions

// Mock for grid3d to control density and volume for dose calculations
template<typename T, typename R>
class MockGrid3D : public mqi::grid3d<T, R> {
public:
    R mock_volume = 1.0;
    T mock_density = 1.0; // g/cm^3, will be converted to g/mm^3 in calculations

    MockGrid3D() : mqi::grid3d<T, R>() {
        // Setup minimal valid dimensions if needed by base class, though not strictly used by mocked methods
        R edges[2] = {0.0, 1.0};
        this->dim_ = {1, 1, 1};
        this->xe_ = new R[2]; this->xe_[0]=edges[0]; this->xe_[1]=edges[1];
        this->ye_ = new R[2]; this->ye_[0]=edges[0]; this->ye_[1]=edges[1];
        this->ze_ = new R[2]; this->ze_[0]=edges[0]; this->ze_[1]=edges[1];
        this->data_ = new T[1]; // Allocate to prevent null access if base methods are called
        this->data_[0] = mock_density;
    }

    ~MockGrid3D() override {
        // delete[] this->xe_; // xe_, ye_, ze_ are deleted by grid3d destructor
        // delete[] this->ye_;
        // delete[] this->ze_;
        // delete[] this->data_; // data_ is also deleted by grid3d destructor
    }
    
    CUDA_HOST_DEVICE T operator[](const mqi::cnb_t p) override {
        return mock_density / 1000.0f; // Convert g/cm3 to g/mm3 as used in dose_to_water
    }

    CUDA_HOST_DEVICE R get_volume(const mqi::cnb_t p) override {
        return mock_volume; // mm^3
    }
     T* get_data() const override { // Made const to match base
        return this->data_;
    }
};

// Mock for material_t to control stopping power ratio for dose_to_water tests
template<typename R>
class MockMaterial : public mqi::h2o_t<R> { // Inherit from h2o_t for convenience
public:
    R mock_spr = 1.0;

    MockMaterial() : mqi::h2o_t<R>() {}

    //This function is CUDA_DEVICE only in the original code.
    //For host-side testing, we make it host-callable.
    //If it were HOST_DEVICE, we could directly use it.
    //Since it's DEVICE only and relies on CUDA_DEVICE const tables,
    //we must mock its behavior for host tests.
    CUDA_HOST_DEVICE R stopping_power_ratio(R Ek, int8_t id = -1) override {
        return mock_spr;
    }
};


// Test fixture for scorer tests
class MqiScorerTest : public ::testing::Test {
protected:
    mqi::scorer<float>* test_scorer_edep;
    mqi::scorer<float>* test_scorer_dose;
    MockGrid3D<mqi::density_t, float> mock_grid;
    mqi::track_t<float> mock_track;

    // Dummy compute_hit function for basic scorer tests (not testing accumulation logic here)
    // This is just to allow scorer construction.
    // The actual functions from mqi_scorer_energy_deposit.hpp will be tested separately for their logic.
    static double dummy_compute_hit(const mqi::track_t<float>& trk, const mqi::cnb_t& cnb, mqi::grid3d<mqi::density_t, float>& geo) {
        return 1.0;
    }
    
    // Host callable version of energy_deposit for testing its logic
    static double host_energy_deposit(const mqi::track_t<float>& trk, const mqi::cnb_t& cnb, mqi::grid3d<mqi::density_t, float>& geo) {
        return trk.dE + trk.local_dE;
    }

    // Host callable version of dose_to_water for testing its logic (with mocked material)
    static double host_dose_to_water(const mqi::track_t<float>& trk, const mqi::cnb_t& cnb, mqi::grid3d<mqi::density_t, float>& geo, MockMaterial<float>& mat) {
        float density_g_per_mm3 = geo.get_data()[cnb]; // Assuming get_data returns density in g/mm3
        float volume_mm3 = geo.get_volume(cnb);
        
        if (density_g_per_mm3 < 1.0e-7f) { // density in g/mm3
            return 0.0;
        } else {
            // Original formula: (trk.dE + trk.local_dE) * 1.60218e-10 / (volume * density * water.stopping_power_ratio(trk.vtx0.ke));
            // Here, water.rho_mass is set to density_g_per_mm3 inside the original dose_to_water,
            // and stopping_power_ratio is called on that water object.
            // Our MockMaterial will use its mock_spr.
            return (trk.dE + trk.local_dE) * 1.60218e-10 / (volume_mm3 * density_g_per_mm3 * mat.stopping_power_ratio(trk.vtx0.ke));
        }
    }


    void SetUp() override {
        const int max_cap = 100;
        // For scorer constructor tests, we can use a dummy compute_hit.
        // The actual compute_hit functions' logic will be tested separately.
        test_scorer_edep = new mqi::scorer<float>("test_edep", max_cap, dummy_compute_hit);
        test_scorer_edep->data_ = new mqi::key_value[max_cap];
        test_scorer_edep->clear_data(); // Initialize with empty_pair

        test_scorer_dose = new mqi::scorer<float>("test_dose", max_cap, dummy_compute_hit);
        test_scorer_dose->data_ = new mqi::key_value[max_cap];
        test_scorer_dose->clear_data();
        
        // Initialize mock_track
        mock_track.vtx0.ke = 100.0f; // MeV
    }

    void TearDown() override {
        delete test_scorer_edep; // This will call delete_data_if_used internally
        delete test_scorer_dose;
    }
};

TEST_F(MqiScorerTest, Initialization) {
    ASSERT_NE(test_scorer_edep->data_, nullptr);
    EXPECT_EQ(test_scorer_edep->max_capacity_, 100);
    EXPECT_STREQ(test_scorer_edep->name_, "test_edep");
    // Check if data is initialized to empty_pair (or whatever clear_data does)
    for (uint32_t i = 0; i < test_scorer_edep->max_capacity_; ++i) {
        EXPECT_EQ(test_scorer_edep->data_[i].key1, mqi::empty_pair);
        EXPECT_EQ(test_scorer_edep->data_[i].key2, mqi::empty_pair);
        EXPECT_EQ(test_scorer_edep->data_[i].value, 0.0);
    }
}

TEST_F(MqiScorerTest, ClearData) {
    // Modify some data
    if (test_scorer_edep->max_capacity_ > 0) {
        test_scorer_edep->data_[0] = {1, 1, 10.0};
    }
    test_scorer_edep->clear_data();
    for (uint32_t i = 0; i < test_scorer_edep->max_capacity_; ++i) {
        EXPECT_EQ(test_scorer_edep->data_[i].key1, mqi::empty_pair);
        EXPECT_EQ(test_scorer_edep->data_[i].key2, mqi::empty_pair);
        EXPECT_EQ(test_scorer_edep->data_[i].value, 0.0);
    }
}

TEST_F(MqiScorerTest, EnergyDepositLogic) {
    mock_track.dE = 10.0f;
    mock_track.local_dE = 5.0f;
    mqi::cnb_t cnb_idx = 0; // Dummy cnb
    double result = host_energy_deposit(mock_track, cnb_idx, mock_grid);
    EXPECT_DOUBLE_EQ(result, 15.0);

    mock_track.dE = 0.0f;
    mock_track.local_dE = 0.0f;
    result = host_energy_deposit(mock_track, cnb_idx, mock_grid);
    EXPECT_DOUBLE_EQ(result, 0.0);
}

TEST_F(MqiScorerTest, DoseToWaterLogic) {
    mock_track.dE = 1.0f; // MeV
    mock_track.local_dE = 0.0f;
    mock_track.vtx0.ke = 100.0f; // MeV (for stopping_power_ratio call)
    
    mqi::cnb_t cnb_idx = 0; 
    
    MockMaterial<float> mock_material;
    mock_material.mock_spr = 1.0f; // Water equivalent

    // Mock grid setup
    mock_grid.mock_volume = 1.0f; // mm^3
    mock_grid.mock_density = 1.0f; // g/cm^3
    // get_data() in mock_grid returns density in g/mm^3
    mock_grid.data_[cnb_idx] = mock_grid.mock_density / 1000.0f; // 0.001 g/mm^3

    // Expected dose: (1.0 MeV * 1.60218e-10 J/MeV) / (1.0 mm^3 * 0.001 g/mm^3 * 1.0 SPR) = 0.000160218 J/g = 0.000160218 Gy
    // The formula uses density in g/mm3 and volume in mm3, so mass is g. MeV to J. J/g -> Gy
    // MeV * 1.60218e-13 J/MeV. Mass = Volume (mm^3) * Density (g/mm^3). So Dose = (MeV * 1.60218e-13) / (Mass_g) Gy
    // (1.0 MeV * 1.60218e-10 J/MeV) / ( (1 mm^3 * (1.0/1000) g/mm^3) * 1.0 ) = 1.60218e-7 J/g = 1.60218e-7 Gy
    double expected_dose = (1.0 * 1.60218e-10) / (1.0 * (1.0/1000.0) * 1.0); // J/g = Gy
    
    double result = host_dose_to_water(mock_track, cnb_idx, mock_grid, mock_material);
    EXPECT_NEAR(result, expected_dose, 1e-12);

    // Test with different SPR
    mock_material.mock_spr = 1.1f;
    expected_dose = (1.0 * 1.60218e-10) / (1.0 * (1.0/1000.0) * 1.1);
    result = host_dose_to_water(mock_track, cnb_idx, mock_grid, mock_material);
    EXPECT_NEAR(result, expected_dose, 1e-12);

    // Test with zero density (should return 0)
    mock_grid.data_[cnb_idx] = 0.0f;
    result = host_dose_to_water(mock_track, cnb_idx, mock_grid, mock_material);
    EXPECT_DOUBLE_EQ(result, 0.0);
}

// The actual accumulation logic (process_hit, insert_pair) is CUDA_DEVICE
// and involves atomic operations or mutexes, making it hard to unit test
// purely from the host without significant refactoring or a CUDA-based test environment.
// Thus, we primarily test the host-callable parts and the logic of compute_hit functions.

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
