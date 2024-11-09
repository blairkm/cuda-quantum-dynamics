#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <cufft.h>
#include <stdio.h>

// Function prototypes for the split-step Fourier method
__global__ void potential_propagation_kernel(cufftDoubleComplex* psi, const double* potential, const double dt, const int NX);
__global__ void kinetic_propagation_kernel(cufftDoubleComplex* psi, const double* kinetic_operator, const int NX);

void initialize_wave_function(cufftDoubleComplex* h_psi, double X0, double S0, double E0, int NX, double dx);
void initialize_potential(double* h_potential, int NX, double LX, double BH, double BW, double EH, double dx);
void initialize_kinetic_operator(double* h_kinetic_operator, int NX, double dx, double DT);
void init_param(double* X0, double* S0, double* E0, double* BH, double* BW, double* EH, FILE** fp, int argc, char* argv[]);
void output_results(FILE* fp, cufftDoubleComplex* h_psi, double* h_potential, int step, int NX, double DT, double LX, double X0, double S0, double E0, double BH, double BW, double EH);

#endif // SIMULATOR_H //