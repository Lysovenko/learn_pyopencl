// Heterogeneous Computing with OpenCL 2.0
// http://dx.doi.org/10.1016/B978-0-12-801414-1.00004-1


kernel void
convolution (read_only image2d_t src, write_only image2d_t dest,
	     constant float *filter, int filter_width, sampler_t sampler)
{
  int column = get_global_id (0), row = get_global_id (1),
    half_width = (int) (filter_width / 2);
  float4 sum = { 0.0f, 0.0f, 0.0f, 0.0f };
  int filter_id = 0;
  int2 coords;			// Coordinates for accesing the image
/* Iterate the filter rows */
  for (int i = -half_width; i <= half_width; i++)
    {
      coords.y = row + i;
      for (int j = -half_width; j <= half_width; j++)
	{
	  coords.x = column + j;
	  float4 pixel = read_imagef (src, sampler, coords);
	  sum += pixel * filter[filter_id++];
	}
    }
  write_imagef (dest, (int2) (column, row), sum);
}
