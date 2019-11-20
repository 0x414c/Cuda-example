#include <assert.h>  // assert
#include <float.h>  // DECIMAL_DIG
#include <stdio.h>  // fprintf, printf, stderr
#include <stddef.h>  // size_t, NULL
#include <stdlib.h>  // free, malloc, EXIT_SUCCESS
#include <math.h>  // cos, fabs, isfinite, sin

#include "cuda_runtime.h"  // __global__, __restrict__, cuda*


#ifndef DECIMAL_DIG
#define DECIMAL_DIG (21)
#endif  // DECIMAL_DIG


typedef double real_type;


__global__ void
vectorAddition (
  const real_type * __restrict__ a, const real_type * __restrict__ b, real_type * __restrict__ c, size_t count
)
{
  const unsigned thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  if (thread_id < count)
  {
    c [thread_id] = a [thread_id] + b [thread_id];
  }
}


int
isClose (real_type x, real_type y, real_type rel_tol, real_type abs_tol)
{
  assert (rel_tol >= 0.0);
  assert (abs_tol >= 0.0);

  if (isfinite (x) && isfinite (y))
  {
    if (x == y)
    {
      return 1;
    }

    const real_type abs_diff (fabs (x - y));

    return (abs_diff <= abs_tol)
        || (abs_diff <= (rel_tol * fabs (x)))
        || (abs_diff <= (rel_tol * fabs (y)));
  }

  return x == y;
}


int
main (int argc, char * argv [])
{
  const size_t count = 65536;
  const size_t bytes = count * sizeof (real_type);
  real_type * __restrict__ const host_a = (real_type *) malloc (bytes);
  if (host_a == NULL)
  {
    fprintf (stderr, "Error: couldn't allocate memory for host vector `a' (%zu bytes).\n", bytes);

    return EXIT_FAILURE;
  }

  real_type * __restrict__ const host_b = (real_type *) malloc (bytes);
  if (host_b == NULL)
  {
    fprintf (stderr, "Error: couldn't allocate memory for host vector `b' (%zu bytes).\n", bytes);

    return EXIT_FAILURE;
  }

  real_type * __restrict__ const host_c = (real_type *) malloc (bytes);
  if (host_c == NULL)
  {
    fprintf (stderr, "Error: couldn't allocate memory for host vector `c' (%zu bytes).\n", bytes);

    return EXIT_FAILURE;
  }

  real_type * __restrict__ const device_a = NULL;
  cudaError_t allocated = cudaMalloc ((void **) & device_a, bytes);
  if (allocated != cudaSuccess)
  {
    fprintf (stderr, "Error: couldn't allocate memory for device vector `a' (%zu bytes): %s.\n", bytes, cudaGetErrorString (allocated));

    return EXIT_FAILURE;
  }

  real_type * __restrict__ const device_b = NULL;
  allocated = cudaMalloc ((void **) & device_b, bytes);
  if (allocated != cudaSuccess)
  {
    fprintf (stderr, "Error: couldn't allocate memory for device vector `b' (%zu bytes): %s.\n", bytes, cudaGetErrorString (allocated));

    return EXIT_FAILURE;
  }

  real_type * __restrict__ const device_c = NULL;
  allocated = cudaMalloc ((void **) & device_c, bytes);
  if (allocated != cudaSuccess)
  {
    fprintf (stderr, "Error: couldn't allocate memory for device vector `c' (%zu bytes): %s.\n", bytes, cudaGetErrorString (allocated));

    return EXIT_FAILURE;
  }

  for (size_t i = 0; i < count; ++ i)
  {
/*    host_a [i] = i;
    host_b [i] = count - i - 1;*/
    host_a [i] = sin (i) * sin (i);
    host_b [i] = cos (i) * cos (i);
  }

  cudaError_t copied = cudaMemcpy (device_a, host_a, bytes, cudaMemcpyHostToDevice);
  if (copied != cudaSuccess)
  {
    fprintf (stderr, "Error: couldn't copy host vector `a' to device: %s.\n", cudaGetErrorString (copied));

    return EXIT_FAILURE;
  }

  copied = cudaMemcpy (device_b, host_b, bytes, cudaMemcpyHostToDevice);
  if (copied != cudaSuccess)
  {
    fprintf (stderr, "Error: couldn't copy host vector `b' to device: %s.\n", cudaGetErrorString (copied));

    return EXIT_FAILURE;
  }

  const unsigned threads_per_block = 256;
  const unsigned blocks_per_grid = (count + threads_per_block - 1) / threads_per_block;
  cudaGetLastError ();
  vectorAddition <<<blocks_per_grid, threads_per_block>>> (device_a, device_b, device_c, count);
  cudaDeviceSynchronize ();
  const cudaError_t added = cudaGetLastError ();
  if (added != cudaSuccess)
  {
    fprintf (stderr, "Error: couldn't launch kernel: %s.\n", cudaGetErrorString (added));

    return EXIT_FAILURE;
  }

  copied = cudaMemcpy (host_c, device_c, bytes, cudaMemcpyDeviceToHost);
  if (copied != cudaSuccess)
  {
    fprintf (stderr, "Error: couldn't copy vector `c' to host: %s.\n", cudaGetErrorString (copied));

    return EXIT_FAILURE;
  }

  for (size_t i = 0; i < count; ++ i)
  {
    const real_type expected = host_a [i] + host_b [i];
    const real_type actual = host_c [i];
    if (! isClose (expected, actual, 1e-8, 1e-16))
    {
      fprintf (
        stderr, "Test failed at element %zu: expected=%.*f;actual=%.*f;.\n",
        i, DECIMAL_DIG, expected, DECIMAL_DIG, actual
      );

      return EXIT_FAILURE;
    }
  }

  cudaError_t freed = cudaFree (device_a);
  if (freed != cudaSuccess)
  {
    fprintf (stderr, "Error: couldn't free device vector `a': %s.\n", cudaGetErrorString (freed));

    return EXIT_FAILURE;
  }

  freed = cudaFree (device_b);
  if (freed != cudaSuccess)
  {
    fprintf (stderr, "Error: couldn't free device vector `b': %s.\n", cudaGetErrorString (freed));

    return EXIT_FAILURE;
  }

  freed = cudaFree (device_c);
  if (freed != cudaSuccess)
  {
    fprintf (stderr, "Error: couldn't free device vector `c': %s.\n", cudaGetErrorString (freed));

    return EXIT_FAILURE;
  }

  free (host_a);
  free (host_b);
  free (host_c);

  printf ("Done.\n");

  return EXIT_SUCCESS;
}
