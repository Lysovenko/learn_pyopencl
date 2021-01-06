// Heterogeneous Computing with OpenCL 2.0
// http://dx.doi.org/10.1016/B978-0-12-801414-1.00004-1


constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
  CLK_FILTER_LINEAR | CLK_ADDRESS_CLAMP;

kernel void
rotate_ (read_only image2d_t src,
	 write_only image2d_t dest, int width, int height, float theta)
{
  /* Output coordinates */
  int x = get_global_id (0), y = get_global_id (1);
  /* compute image center */
  float x0 = (float) width / 2.0, y0 = (float) height / 2.0,
    /* Work-item location relative to the image center */
    xprime = x - x0, yprime = y - y0,
    /*sine and cosine */
    sin_theta = sin (theta), cos_theta = cos (theta);
  /* input location */
  float2 read_coord;
  read_coord.x = xprime * cos_theta - yprime * sin_theta + x0;
  read_coord.y = xprime * sin_theta + yprime * cos_theta + y0;
  /* Read the input image */
  float4 value = read_imagef (src, sampler, read_coord);
  /* write the output image */
  write_imagef (dest, (int2) (x, y), value);
}
