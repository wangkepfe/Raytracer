#pragma once

struct Lock {
    int *mutex;
    
    Lock( void ) {
        cudaMalloc( (void**)&mutex, sizeof(int) );
        cudaMemset( mutex, 0, sizeof(int) );
    }

    ~Lock( void ) {
        cudaFree( mutex );
    }

    __device__ void lock( void ) {
        while( atomicCAS( mutex, 0, 1 ) != 0 );
    }

    __device__ void unlock( void ) {
        atomicExch( mutex, 0 );
    }
};