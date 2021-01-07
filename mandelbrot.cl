
kernel void
mandelbrot (write_only image2d_t dest, int width, int height,
	 float x0, float y0, float dx, float dy)
{
  /* Output coordinates */
  int x = get_global_id (0), y = get_global_id (1), steps;
  float real = x0 + x * dx, imag = y0 - y * dy, zr = 0., zi = 0.;
  for (steps = 0; steps < 64; steps++)
    {
      zr = zr * zr - zi * zi + real;
      zi = 2. * zr * zi + imag;
      if ((zr * zr + zi * zi) > 4.)
	break;
    }
  int r = (steps & 1) | ((steps >> 3) & 1) << 1 | ((steps >> 6) & 1) << 2,
    g =
    ((steps >> 1) & 1) | ((steps >> 4) & 1) << 1 | ((steps >> 7) & 1) << 2,
    b =
    ((steps >> 2) & 1) | ((steps >> 5) & 1) << 1 | ((steps >> 8) & 1) << 2;
  if (steps == 64)
    {r =  g = b = 0;}
  else
    {
      r = 8 - r;
      g = 8 - g;
      b = 8 - b;
    }
  float4 value;
  value.x = (float) r / 8.;
  value.y = (float) g / 8.;
  value.z = (float) b / 8.;
  value.w = 1.;
  /* write the output image */
  write_imagef (dest, (int2) (x, y), value);
}
