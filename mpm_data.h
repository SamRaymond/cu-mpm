#ifndef MPM_DATA_H
#define MPM_DATA_H

#include "mpm_common.h"

//------------------------------------------------------------
// Data Structures
//------------------------------------------------------------

struct MpmParticleData
{
    float* pos_x; 
    float* pos_y;
    float* vel_x;
    float* vel_y;
    float* mass;
    float* mom_x;
    float* mom_y;

    float* halfSizeX;   // particle half-size in x (GIMP top-hat half-width)
    float* halfSizeY;   // particle half-size in y

    // Stress/strain etc. stored but not deeply used here
    float* stress_xx;
    float* stress_yy;
    float* stress_xy;
    float* strain_xx;
    float* strain_yy;
    float* strain_xy;
    float* grad_vx_x;
    float* grad_vx_y;
    float* grad_vy_x;
    float* grad_vy_y;
};

struct MpmGridNodeData
{
    float* mass;
    float* mom_x;
    float* mom_y;
    float* vel_x;
    float* vel_y;
};

//------------------------------------------------------------
// Function Prototypes
//------------------------------------------------------------
MpmParticleData allocateParticleDataOnDevice(int numParticles);
void freeParticleData(MpmParticleData p);

MpmGridNodeData allocateGridNodeDataOnDevice(int numNodes);
void freeGridNodeData(MpmGridNodeData g);

#endif // MPM_DATA_H
