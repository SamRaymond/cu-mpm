#ifndef MPM_IO_H
#define MPM_IO_H

#include "mpm_common.h"

// Write Particle Positions + Velocities to VTP
void SaveParticleDataToVTP(const char* filename,
                           const float* x, const float* y,
                           const float* vx, const float* vy,
                           int n);

#endif // MPM_IO_H
