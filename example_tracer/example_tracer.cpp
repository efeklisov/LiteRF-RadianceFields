#include <vector>
#include <chrono>
#include <string>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "Image2d.h"

#include "example_tracer.h"

int enzyme_dup;
int enzyme_out;
int enzyme_const;
int enzyme_dupnoneed;

float __enzyme_autodiff(...);

float2 RayBoxIntersection(float3 ray_pos, float3 ray_dir, float3 boxMin, float3 boxMax)
{
  ray_dir.x = 1.0f/ray_dir.x; // may precompute if intersect many boxes
  ray_dir.y = 1.0f/ray_dir.y; // may precompute if intersect many boxes
  ray_dir.z = 1.0f/ray_dir.z; // may precompute if intersect many boxes

  float lo = ray_dir.x*(boxMin.x - ray_pos.x);
  float hi = ray_dir.x*(boxMax.x - ray_pos.x);
  
  float tmin = std::min(lo, hi);
  float tmax = std::max(lo, hi);

  float lo1 = ray_dir.y*(boxMin.y - ray_pos.y);
  float hi1 = ray_dir.y*(boxMax.y - ray_pos.y);

  tmin = std::max(tmin, std::min(lo1, hi1));
  tmax = std::min(tmax, std::max(lo1, hi1));

  float lo2 = ray_dir.z*(boxMin.z - ray_pos.z);
  float hi2 = ray_dir.z*(boxMax.z - ray_pos.z);

  tmin = std::max(tmin, std::min(lo2, hi2));
  tmax = std::min(tmax, std::max(lo2, hi2));
  
  return float2(tmin, tmax);
}

bool RaySphereIntersection(float3 ray_pos, float3 ray_dir, float step)
{
  float t_min = 1.0;
  float t_max = 8.0;

  float t = t_min;
  while (t < t_max) {
    float3 p = ray_pos + t * ray_dir;
    if (p[0] * p[0] + p[1] * p[1] + p[2] * p[2] < 0.25)
      return true;
    t += step;
  }

  return false;
}

size_t indexGrid(size_t x, size_t y, size_t z, size_t gridSize) {
    return x + y * gridSize + z * gridSize * gridSize;
}

float sigmoid(float x) {
  return 1 / (1 + exp(-x));
}

inline Cell lerpCell(const Cell v0, const Cell v1, const float t) {
  Cell ret;
  ret.density = lerp(v0.density, v1.density, t);
  for (size_t i = 0; i < 9; i++) {
    ret.sh_r[i] = lerp(v0.sh_r[i], v1.sh_r[i], t);  
    ret.sh_g[i] = lerp(v0.sh_g[i], v1.sh_g[i], t);  
    ret.sh_b[i] = lerp(v0.sh_b[i], v1.sh_b[i], t);
  }
  return ret;
}

inline float eval_sh(float sh_coeffs[9], float3 dir) {
  return C0 * sh_coeffs[0] - C1 * dir[1] * sh_coeffs[1] + C1 * dir[2] * sh_coeffs[2] - C1 * dir[0] * sh_coeffs[3]
    + C2[0] * dir[0] * dir[1] * sh_coeffs[4] + C2[1] * dir[1] * dir[2] * sh_coeffs[5] 
    + C2[2] * (2.0 * dir[2] * dir[2] - dir[0] * dir[0] - dir[1] * dir[1]) * sh_coeffs[6]
    + C2[3] * dir[0] * dir[2] * sh_coeffs[7] + C2[4] * (dir[0] * dir[0] - dir[1] * dir[1]) * sh_coeffs[8];
}


float3 RayGridIntersection(bool mode, float3 L, float3 ray_pos, float3 ray_dir, double step, float2 boxFarNear, BoundingBox bb, Cell* grid, size_t gridSize)
{
  float t_min = 1.0;
  float t_max = 8.0;

  if (t_min > boxFarNear.y)
    return L;

  float throughput = 1.0;
  float3 colour = float3(0.0);

  float t = max(t_min, boxFarNear.x);
  while ((t < min(t_max, boxFarNear.y)) && (throughput > 0.01)) {
    float3 p = ray_pos + t * ray_dir;

    float3 coords01 = (p - bb.min) / (bb.max - bb.min);
    float3 coords = coords01 * (float)(gridSize - 1);

    float3 nearCoords = coords - ray_dir * 0.5;
    nearCoords[0] = round(nearCoords[0]);
    nearCoords[1] = round(nearCoords[1]);
    nearCoords[2] = round(nearCoords[2]);

    float3 farCoords = coords + ray_dir * 0.5;
    farCoords[0] = round(farCoords[0]);
    farCoords[1] = round(farCoords[1]);
    farCoords[2] = round(farCoords[2]);

    float3 lerpFactors = coords - nearCoords;

    Cell xy00 = lerpCell(grid[indexGrid(nearCoords[0], nearCoords[1], nearCoords[2], gridSize)], grid[indexGrid(farCoords[0], nearCoords[1], nearCoords[2], gridSize)], lerpFactors.x);
    Cell xy10 = lerpCell(grid[indexGrid(nearCoords[0], farCoords[1], nearCoords[2], gridSize)], grid[indexGrid(farCoords[0], farCoords[1], nearCoords[2], gridSize)], lerpFactors.x);
    Cell xy01 = lerpCell(grid[indexGrid(nearCoords[0], nearCoords[1], farCoords[2], gridSize)], grid[indexGrid(farCoords[0], nearCoords[1], farCoords[2], gridSize)], lerpFactors.x);
    Cell xy11 = lerpCell(grid[indexGrid(nearCoords[0], farCoords[1], farCoords[2], gridSize)], grid[indexGrid(farCoords[0], farCoords[1], farCoords[2], gridSize)], lerpFactors.x);

    Cell xyz0 = lerpCell(xy00, xy10, lerpFactors.y);
    Cell xyz1 = lerpCell(xy01, xy11, lerpFactors.y);

    Cell gridVal = lerpCell(xyz0, xyz1, lerpFactors.z);

    // relu
    if (gridVal.density < 0.0)
      gridVal.density = 0.0;

    float tr = exp(-gridVal.density * step);

    float3 RGB = float3(1.0, 1.0, 1.0);
    //float3 RGB = float3(eval_sh(gridVal.sh_r, ray_dir), eval_sh(gridVal.sh_g, ray_dir), eval_sh(gridVal.sh_b, ray_dir));
    colour = colour + (mode ? 1 : -1) * throughput * (1 - tr) * RGB;
    
    throughput *= tr;

    t += step;
  }

  return colour;
}

float RayGridLoss(float3 &ref, float3 &L, float3 &rayPos, float3 &rayDir, double &step, float2 &boxFarNear, BoundingBox &bb, Cell* &grid, size_t &gridSize) {
  float3 color = RayGridIntersection(true, float3(0.0f), rayPos, rayDir, step, boxFarNear, bb, grid, gridSize);

  return std::abs(color[0] - ref[0]) + std::abs(color[1] - ref[1]) + std::abs(color[2] - ref[2]);
}

float RayGridLossGrad(float3 &ref, float3 &L, float3 &rayPos, float3 &rayDir, double &step, float2 &boxFarNear, BoundingBox &bb, Cell* &grid, Cell* &grid_d, size_t &gridSize) {
  return __enzyme_autodiff((void*)RayGridLoss, 
    enzyme_const, &ref, enzyme_const, &L, enzyme_const, &rayPos, 
    enzyme_const, &rayDir, enzyme_const, &step, enzyme_const, &boxFarNear, 
    enzyme_const, &bb, enzyme_dup, &grid, &grid_d, enzyme_const, &gridSize);
}

void RayMarcherExample::LoadModel(std::string densities, std::string sh) {
  std::cout << "Loading model\n";
  {
    json densities_data;
    {
      std::ifstream f(densities);
      densities_data = json::parse(f);
    }

    std::cout << "Loaded densities from disk\n";
  
    #pragma openmp parallel for
    for (size_t z = 0; z < gridSize; z++)
      for (size_t y = 0; y < gridSize; y++)
        for (size_t x = 0; x < gridSize; x++) {
          size_t gridIndex = indexGrid(x, y, z, gridSize);
          grid[gridIndex].density = densities_data[z][y][x][0];
        }

    std::cout << "Assigned densities\n";
  }
  
  // {
  //   json sh_data;
  //   {
  //     std::ifstream f(sh);
  //     sh_data = json::parse(f);
  //   }

  //   std::cout << "Loaded SH coefficients from disk\n";

  //   #pragma openmp parallel for
  //   for (size_t z = 0; z < gridSize; z++)
  //     for (size_t y = 0; y < gridSize; y++)
  //       for (size_t x = 0; x < gridSize; x++) {
  //         size_t gridIndex = indexGrid(x, y, z, gridSize);

  //         for (size_t i = 0; i < 9; i++) {
  //           grid[gridIndex].sh_r[i] = sh_data[z][y][x][i * 3];
  //           grid[gridIndex].sh_g[i] = sh_data[z][y][x][1 + i * 3];
  //           grid[gridIndex].sh_b[i] = sh_data[z][y][x][2 + i * 3];
  //         }
  //       }

  //   std::cout << "Assigned SH coefficients\n";
  // }
  std::cout << "Done loading\n";
}

static inline float3 EyeRayDir(float x, float y, float4x4 a_mViewProjInv)
{
  float4 pos = float4(2.0f*x - 1.0f, 2.0f*y - 1.0f, 0.0f, 1.0f );
  pos = a_mViewProjInv * pos;
  pos /= pos.w;
  return normalize(to_float3(pos));
}

static inline void transform_ray3f(float4x4 a_mWorldViewInv, float3* ray_pos, float3* ray_dir) 
{
  float4 rayPosTransformed = a_mWorldViewInv*to_float4(*ray_pos, 1.0f);
  float4 rayDirTransformed = a_mWorldViewInv*to_float4(*ray_dir, 0.0f);
  
  (*ray_pos) = to_float3(rayPosTransformed);
  (*ray_dir) = to_float3(normalize(rayDirTransformed));
}

float4 RayMarchConstantFog(float tmin, float tmax, float& alpha)
{
  float dt = 0.05f;
	float t  = tmin;
	
	alpha = 1.0f;
	float4 color = float4(0.0f);
	
	while(t < tmax && alpha > 0.01f)
	{
	  float a = 0.025f;
	  color += a*alpha*float4(1.0f,1.0f,0.0f,0.0f);
	  alpha *= (1.0f-a);
	  t += dt;
	}
	
	return color;
}

static inline uint32_t RealColorToUint32(float4 real_color)
{
  float  r = real_color[0]*255.0f;
  float  g = real_color[1]*255.0f;
  float  b = real_color[2]*255.0f;
  float  a = real_color[3]*255.0f;

  uint32_t red   = (uint32_t)r;
  uint32_t green = (uint32_t)g;
  uint32_t blue  = (uint32_t)b;
  uint32_t alpha = (uint32_t)a;

  return red | (green << 8) | (blue << 16) | (alpha << 24);
}

static inline float4 Uint32ToRealColor(uint color) {
  float r = (color & 0x000F) / 255.0f;
  float g = ((color >> 8) & 0x000F) / 255.0f;
  float b = ((color >> 16) & 0x000F) / 255.0f;
  float a = ((color >> 24) & 0x000F) / 255.0f;

  return float4(r, g, b, a);
}

// void L1LossGrad(float* loss, float* loss_d, uint* ref, uint* gen, int width, int height, RayMarcherExample* pImpl, RayMarcherExample* pImpl_d) {

//   __enzyme_autodiff(L1Loss, enzyme_dup, loss, loss_d, enzyme_const, ref, enzyme_const, gen, enzyme_const, width, enzyme_const, height, enzyme_dup, pImpl, pImpl_d);
// }

void L1Loss(float* loss, uint* ref, uint* gen, int width, int height, RayMarcherExample* pImpl, RayMarcherExample* pImpl_d, const char* fileName) {
  *loss = 0.0f;

  std::vector<float4> L(width * height);
  std::vector<float4> ref_float(width * height);

  pImpl->kernel2D_RayMarch(gen, width, height, L.data());

  LiteImage::SaveBMP(fileName, gen, width, height);

  #pragma openmp parallel for
  for (int y = 0; y < height; y++)
    for (int x = 0; x < width; x++) {
      float4 ref_pixel = Uint32ToRealColor(ref[y * width + x]);
      float4 gen_pixel = L[y * width + x];

      *loss += abs(ref_pixel[0] - gen_pixel[0]) + abs(ref_pixel[1] - gen_pixel[1]) + abs(ref_pixel[2] - gen_pixel[2]);
      ref_float[y*width + x] = ref_pixel;
    }

  pImpl->kernel2D_RayMarchGrad(width, height, ref_float.data(), L.data(), pImpl_d);
  
}

void RayMarcherExample::kernel2D_RayMarch(uint32_t* out_color, uint32_t width, uint32_t height, float4* L) 
{
  #pragma openmp parallel for
  for(uint32_t y=0;y<height;y++) 
  {
    for(uint32_t x=0;x<width;x++) 
    {
      float3 rayDir = EyeRayDir((float(x) + 0.5f) / float(width), (float(y) + 0.5f) / float(height), m_worldViewProjInv); 
      float3 rayPos = float3(0.0f, 0.0f, 0.0f);

      transform_ray3f(m_worldViewInv, &rayPos, &rayDir);
      
      float2 tNearAndFar = RayBoxIntersection(rayPos, rayDir, bb.min, bb.max);
      
      float4 resColor(0.0f);
      if(tNearAndFar.x < tNearAndFar.y)
      //if (RaySphereIntersection(rayPos, rayDir, 0.05))
      {
        float step = (bb.max[0] - bb.min[0]) / gridSize;
        float3 colour = RayGridIntersection(true, float3(0.0f), rayPos, rayDir, step, tNearAndFar, bb, grid.data(), gridSize);
        resColor = float4(colour[0], colour[1], colour[2], 1.0);

        // float alpha = 1.0f;
        // resColor = float4(1.0f);
	      // resColor = RayMarchConstantFog(tNearAndFar.x, tNearAndFar.y, alpha);
      }
      
      L[y * width + x] = resColor;
      out_color[y*width+x] = RealColorToUint32(resColor);
    }
  }
}

void RayMarcherExample::kernel2D_RayMarchGrad(uint32_t width, uint32_t height, float4* res, float4* L, RayMarcherExample* pImpl_d) 
{
  RayMarcherExample pImpl_in;
  pImpl_in.InitGrid(pImpl_d->gridSize);

  for(uint32_t y=0;y<height;y++) 
  {
    for(uint32_t x=0;x<width;x++) 
    {
      float3 rayDir = EyeRayDir((float(x) + 0.5f) / float(width), (float(y) + 0.5f) / float(height), m_worldViewProjInv); 
      float3 rayPos = float3(0.0f, 0.0f, 0.0f);

      transform_ray3f(m_worldViewInv, &rayPos, &rayDir);
      
      float2 tNearAndFar = RayBoxIntersection(rayPos, rayDir, bb.min, bb.max);
      
      if(tNearAndFar.x < tNearAndFar.y)
      //if (RaySphereIntersection(rayPos, rayDir, 0.05))
      {
        double step = (bb.max[0] - bb.min[0]) / gridSize;

        float4 res_pixel = res[y*width+x];
        float4 L_pixel = L[y*width+x];

        float3 res_pixel3;
        res_pixel3[0] = res_pixel[0];
        res_pixel3[1] = res_pixel[1];
        res_pixel3[2] = res_pixel[2];

        float3 L_pixel3;
        L_pixel3[0] = L_pixel[0];
        L_pixel3[1] = L_pixel[1];
        L_pixel3[2] = L_pixel[2];

        Cell* grid_data = grid.data();
        Cell* grid_in_data = pImpl_in.grid.data();

        pImpl_in.zeroGrad();
        float dL1 = RayGridLossGrad(res_pixel3, L_pixel3, rayPos, rayDir, step, tNearAndFar, bb, grid_data, grid_in_data, gridSize);
        pImpl_d->optimizerStep(&pImpl_in, 1.0f);
      }
    }
  }
}

void RayMarcherExample::RayMarch(uint32_t* out_color, uint32_t width, uint32_t height)
{ 
  // auto start = std::chrono::high_resolution_clock::now();
  // kernel2D_RayMarch(out_color, width, height);
  // rayMarchTime = float(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count())/1000.f;
}  

void RayMarcherExample::GetExecutionTime(const char* a_funcName, float a_out[4])
{
  if(std::string(a_funcName) == "RayMarch")
    a_out[0] =  rayMarchTime;
}
