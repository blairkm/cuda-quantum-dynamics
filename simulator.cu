#define _USE_MATH_DEFINES  // Ensure M_PI is defined
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include "simulator.cuh"
#include "device_launch_parameters.h"

// CUDA kernel to propagate wave function in potential space for half-step
__global__ void potential_propagation_kernel(cufftDoubleComplex* psi, const double* potential, const double dt, const int NX) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NX) {
        double wr = cos(-0.5 * dt * potential[idx]) * psi[idx].x - sin(-0.5 * dt * potential[idx]) * psi[idx].y;
        double wi = cos(-0.5 * dt * potential[idx]) * psi[idx].y + sin(-0.5 * dt * potential[idx]) * psi[idx].x;
        psi[idx].x = wr;
        psi[idx].y = wi;
    }
}

// CUDA kernel for kinetic propagation in momentum space
__global__ void kinetic_propagation_kernel(cufftDoubleComplex* psi, const double* kinetic_operator, const int NX) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NX) {
        psi[idx].x *= kinetic_operator[idx];
        psi[idx].y *= kinetic_operator[idx];
    }
}

void initialize_wave_function(cufftDoubleComplex* h_psi, double X0, double S0, double E0, int NX, double dx) {
    double norm = 0.0;
    for (int i = 0; i < NX; ++i) {
        double x = dx * i - X0;
        double gauss = exp(-0.25 * x * x / (S0 * S0));
        h_psi[i].x = gauss * cos(sqrt(2.0 * E0) * x);
        h_psi[i].y = gauss * sin(sqrt(2.0 * E0) * x);
        norm += h_psi[i].x * h_psi[i].x + h_psi[i].y * h_psi[i].y;
    }

    // Normalize the wave function
    norm = sqrt(norm * dx);
    for (int i = 0; i < NX; ++i) {
        h_psi[i].x /= norm;
        h_psi[i].y /= norm;
    }
}

void initialize_potential(double* h_potential, int NX, double LX, double BH, double BW, double EH, double dx) {
    for (int i = 0; i < NX; ++i) {
        double x = dx * i;
        if (i == 0 || i == NX - 1) {
            h_potential[i] = EH;
        }
        else if (fabs(x - 0.5 * LX) < 0.5 * BW) {
            h_potential[i] = BH;
        }
        else {
            h_potential[i] = 0.0;
        }
    }
}

void initialize_kinetic_operator(double* h_kinetic_operator, int NX, double dx, double DT) {
    for (int i = 0; i < NX; ++i) {
        double k = (i < NX / 2) ? i * (2 * M_PI / (NX * dx)) : (i - NX) * (2 * M_PI / (NX * dx));
        h_kinetic_operator[i] = exp(-0.5 * DT * k * k);
    }
}

void init_param(double* X0, double* S0, double* E0, double* BH, double* BW, double* EH, FILE** fp, int argc, char* argv[]) {
    if (argc >= 10) {
        *X0 = atof(argv[1]);
        *S0 = atof(argv[2]);
        *E0 = atof(argv[3]);
        *fp = fopen(argv[4], "w+");
    }
    else {
        *X0 = atof(argv[1]);
        *S0 = atof(argv[2]);
        *E0 = atof(argv[3]);
        *BH = atof(argv[4]);
        *BW = atof(argv[5]);
        *EH = atof(argv[6]);
        *fp = fopen(argv[7], "w+");
    }
}

void output_results(FILE* fp, cufftDoubleComplex* h_psi, double* h_potential, int step, int NX, double DT, double LX, double X0, double S0, double E0, double BH, double BW, double EH) {
    // Output results in a format similar to the previous implementation
    fprintf(fp, "timestamp: %.5f\n", step * DT);
    fprintf(fp, "params: %d %lf %lf %lf %lf %lf %lf %lf %lf\n", NX, LX, DT, X0, S0, E0, BH, BW, EH);
    fprintf(fp, "psi_re: ");
    for (int i = 0; i < NX; i++) {
        fprintf(fp, "%le ", h_psi[i].x);
    }
    fprintf(fp, "\n");
    fprintf(fp, "psi_im: ");
    for (int i = 0; i < NX; i++) {
        fprintf(fp, "%le ", h_psi[i].y);
    }
    fprintf(fp, "\n");
    fprintf(fp, "pot: ");
    for (int i = 0; i < NX; i++) {
        fprintf(fp, "%le ", h_potential[i]);
    }
    fprintf(fp, "\n\n");
}

int main(int argc, char* argv[]) {
    // Initialize parameters (e.g., from command line arguments or file input)
    const int NX = 1024;         // Number of mesh points
    const double LX = 100.0;     // Length of the simulation box
    const double DT = 0.0005;    // Time discretization unit
    const int BLOCK_SIZE = 256;  // CUDA block size

    double X0, S0, E0, BH, BW, EH;
    FILE* fp;
    init_param(&X0, &S0, &E0, &BH, &BW, &EH, &fp, argc, argv);

    if (fp == NULL) {
        fprintf(stderr, "Error opening output file\n");
        return 1;
    }

    // Calculate spatial step size
    double dx = LX / NX;

    // Host arrays
    cufftDoubleComplex* h_psi = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex) * NX);
    double* h_potential = (double*)malloc(sizeof(double) * NX);
    double* h_kinetic_operator = (double*)malloc(sizeof(double) * NX);

    // Initialize wave function, potential, and kinetic operator
    initialize_wave_function(h_psi, X0, S0, E0, NX, dx);
    initialize_potential(h_potential, NX, LX, BH, BW, EH, dx);
    initialize_kinetic_operator(h_kinetic_operator, NX, dx, DT);

    // Device arrays
    cufftDoubleComplex* d_psi;
    double* d_potential, * d_kinetic_operator;
    cudaMalloc((void**)&d_psi, sizeof(cufftDoubleComplex) * NX);
    cudaMalloc((void**)&d_potential, sizeof(double) * NX);
    cudaMalloc((void**)&d_kinetic_operator, sizeof(double) * NX);

    // Copy data to device
    cudaMemcpy(d_psi, h_psi, sizeof(cufftDoubleComplex) * NX, cudaMemcpyHostToDevice);
    cudaMemcpy(d_potential, h_potential, sizeof(double) * NX, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kinetic_operator, h_kinetic_operator, sizeof(double) * NX, cudaMemcpyHostToDevice);

    // Create cuFFT plan
    cufftHandle plan;
    cufftResult result = cufftPlan1d(&plan, NX, CUFFT_Z2Z, 1);
    if (result != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: Plan creation failed\n");
        return 1;
    }

    // Run simulation
    int numBlocks = (NX + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int step = 0; step < 100000; ++step) {
        // Step 1: Half-step potential propagation in real space
        potential_propagation_kernel <<<numBlocks, BLOCK_SIZE >>> (d_psi, d_potential, DT, NX);

        // Step 2: Fourier Transform to momentum space
        cufftExecZ2Z(plan, d_psi, d_psi, CUFFT_FORWARD);

        // Step 3: Kinetic propagation in momentum space
        kinetic_propagation_kernel <<<numBlocks, BLOCK_SIZE >>> (d_psi, d_kinetic_operator, NX);

        // Step 4: Inverse Fourier Transform to real space
        cufftExecZ2Z(plan, d_psi, d_psi, CUFFT_INVERSE);

        // Step 5: Half-step potential propagation in real space again
        potential_propagation_kernel <<<numBlocks, BLOCK_SIZE >>> (d_psi, d_potential, DT, NX);

        // Output results every 200 steps
        if (step % 200 == 0) {
            cudaMemcpy(h_psi, d_psi, sizeof(cufftDoubleComplex) * NX, cudaMemcpyDeviceToHost);
            output_results(fp, h_psi, h_potential, step, NX, DT, LX, X0, S0, E0, BH, BW, EH);
        }
    }

    // Cleanup
    fclose(fp);
    cufftDestroy(plan);
    cudaFree(d_psi);
    cudaFree(d_potential);
    cudaFree(d_kinetic_operator);
    free(h_psi);
    free(h_potential);
    free(h_kinetic_operator);

    return 0;
}





