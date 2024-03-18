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
  ray_dir.x = 1.0f / ray_dir.x; // may precompute if intersect many boxes
  ray_dir.y = 1.0f / ray_dir.y; // may precompute if intersect many boxes
  ray_dir.z = 1.0f / ray_dir.z; // may precompute if intersect many boxes

  float lo = ray_dir.x * (boxMin.x - ray_pos.x);
  float hi = ray_dir.x * (boxMax.x - ray_pos.x);

  float tmin = std::min(lo, hi);
  float tmax = std::max(lo, hi);

  float lo1 = ray_dir.y * (boxMin.y - ray_pos.y);
  float hi1 = ray_dir.y * (boxMax.y - ray_pos.y);

  tmin = std::max(tmin, std::min(lo1, hi1));
  tmax = std::min(tmax, std::max(lo1, hi1));

  float lo2 = ray_dir.z * (boxMin.z - ray_pos.z);
  float hi2 = ray_dir.z * (boxMax.z - ray_pos.z);

  tmin = std::max(tmin, std::min(lo2, hi2));
  tmax = std::min(tmax, std::max(lo2, hi2));

  return float2(tmin, tmax);
}

bool RaySphereIntersection(float3 ray_pos, float3 ray_dir, float step)
{
  float t_min = 1.0;
  float t_max = 8.0;

  float t = t_min;
  while (t < t_max)
  {
    float3 p = ray_pos + t * ray_dir;
    if (p[0] * p[0] + p[1] * p[1] + p[2] * p[2] < 0.25)
      return true;
    t += step;
  }

  return false;
}

size_t indexGrid(size_t x, size_t y, size_t z, size_t gridSize)
{
  return x + y * gridSize + z * gridSize * gridSize;
}

float sigmoid(float x)
{
  return 1 / (1 + exp(-x));
}

inline Cell lerpCell(const Cell v0, const Cell v1, const float t)
{
  Cell ret = {};

  for (size_t i = 0; i < CellSize; i++)
    ((float*)&ret)[i] = LiteMath::lerp(((float*)&v0)[i], ((float*)&v1)[i], t);

  return ret;
}

// From Mitsuba 3
void sh_eval_2(const float3 &d, float *out)
{
  float x = d.x, y = d.y, z = d.z, z2 = z * z;
  float c0, c1, s0, s1, tmp_a, tmp_b, tmp_c;

  out[0] = 0.28209479177387814;
  out[2] = z * 0.488602511902919923;
  out[6] = z2 * 0.94617469575756008 + -0.315391565252520045;
  c0 = x;
  s0 = y;

  tmp_a = -0.488602511902919978;
  out[3] = tmp_a * c0;
  out[1] = tmp_a * s0;
  tmp_b = z * -1.09254843059207896;
  out[7] = tmp_b * c0;
  out[5] = tmp_b * s0;
  c1 = x * c0 - y * s0;
  s1 = x * s0 + y * c0;

  tmp_c = 0.546274215296039478;
  out[8] = tmp_c * c1;
  out[4] = tmp_c * s1;
}

float eval_sh(float *sh, float3 rayDir)
{
  float sh_coeffs[SH_WIDTH];
  sh_eval_2(rayDir, sh_coeffs);

  float sum = 0.0f;
  for (int i = 0; i < SH_WIDTH; i++)
    sum += sh[i] * sh_coeffs[i];

  return sum;
}

float3 RayGridIntersection(float3 rayPos, float3 rayDir, double step, 
  const std::vector<size_t> &intersects, const std::vector<OrthoTree::BoundingBox3D> &boxes, 
  Cell *grid, size_t gridSize, float &transmittance, float &cauchy, bool greyscale = false)
{
  float throughput = 1.0;
  float3 colour = float3(0.0);

  cauchy = 0;
  float tPrev = -1;
  for (const size_t i : intersects)
  {
    if (throughput < 0.1)
      break;

    float3 bbMin = float3(boxes[i].Min[0], boxes[i].Min[1], boxes[i].Min[2]);
    float3 bbMax = float3(boxes[i].Max[0], boxes[i].Max[1], boxes[i].Max[2]);

    float2 tNearAndFar = RayBoxIntersection(rayPos, rayDir, bbMin, bbMax);

    if (tPrev < 0.0f)
      tPrev = tNearAndFar.x;

    float t = (tNearAndFar.x + tNearAndFar.y) / 2.0f;

    float actualStep = (t - tPrev);
    if (actualStep > sqrt(3) * step)
      actualStep -= ((int)(actualStep / step) - 1) * step;

    float3 p = rayPos + t * rayDir;

    float3 lerpFactors = (p - bbMin) / (bbMax - bbMin);

    int3 nearCoords = clamp((int3)(bbMin * gridSize), 0, gridSize - 1);
    int3 farCoords = clamp(nearCoords + 1, 0, gridSize - 1);

    Cell xy00 = lerpCell(grid[indexGrid(nearCoords[0], nearCoords[1], nearCoords[2], gridSize)],
      grid[indexGrid(farCoords[0], nearCoords[1], nearCoords[2], gridSize)], lerpFactors.x);
    Cell xy10 = lerpCell(grid[indexGrid(nearCoords[0], farCoords[1], nearCoords[2], gridSize)],
      grid[indexGrid(farCoords[0], farCoords[1], nearCoords[2], gridSize)], lerpFactors.x);
    Cell xy01 = lerpCell(grid[indexGrid(nearCoords[0], nearCoords[1], farCoords[2], gridSize)],
      grid[indexGrid(farCoords[0], nearCoords[1], farCoords[2], gridSize)], lerpFactors.x);
    Cell xy11 = lerpCell(grid[indexGrid(nearCoords[0], farCoords[1], farCoords[2], gridSize)],
      grid[indexGrid(farCoords[0], farCoords[1], farCoords[2], gridSize)], lerpFactors.x);

    Cell xyz0 = lerpCell(xy00, xy10, lerpFactors.y);
    Cell xyz1 = lerpCell(xy01, xy11, lerpFactors.y);

    Cell gridVal = lerpCell(xyz0, xyz1, lerpFactors.z);

    // relu
    if (gridVal.density < 0.0)
      gridVal.density = 0.0;

    cauchy += log(1 + 2 * gridVal.density * gridVal.density);

    float tr = exp(-gridVal.density * actualStep);

    float3 RGB;
    if (greyscale)
      RGB = float3(1.0f);
    else {
      float sh = clamp(eval_sh(gridVal.sh, rayDir), 0.0f, 1.0f);
      RGB = float3(clamp(gridVal.RGB[0] * sh, 0.0f, 1.0f),
        clamp(gridVal.RGB[1] * sh, 0.0f, 1.0f),
        clamp(gridVal.RGB[2] * sh, 0.0f, 1.0f));
    }
      
    colour = colour + throughput * (1 - tr) * RGB;

    throughput *= tr;
    tPrev = t;
  }
  transmittance = throughput;
  return clamp(colour, 0.0, 1.0);
}

float TVRegularisation(size_t &x, size_t &y, size_t &z, Cell *&grid, size_t &gridSize)
{
  float ret = 0.0f;

  float lambda_density = 0.01f;
  float lambda_SH = 0.01f;

  auto getVoxel = [&](size_t x, size_t y, size_t z)
  {
    float3 coords = float3(x, y, z) + 0.5f;

    int3 nearCoords = clamp((int3)coords, int3(0), int3(gridSize - 1));
    int3 farCoords = clamp((int3)coords + int3(1), int3(0), int3(gridSize - 1));

    float3 lerpFactors = coords - (float3)nearCoords;

    Cell xy00 = lerpCell(grid[indexGrid(nearCoords[0], nearCoords[1], nearCoords[2], gridSize)], 
      grid[indexGrid(farCoords[0], nearCoords[1], nearCoords[2], gridSize)], lerpFactors.x);
    Cell xy10 = lerpCell(grid[indexGrid(nearCoords[0], farCoords[1], nearCoords[2], gridSize)], 
      grid[indexGrid(farCoords[0], farCoords[1], nearCoords[2], gridSize)], lerpFactors.x);
    Cell xy01 = lerpCell(grid[indexGrid(nearCoords[0], nearCoords[1], farCoords[2], gridSize)], 
      grid[indexGrid(farCoords[0], nearCoords[1], farCoords[2], gridSize)], lerpFactors.x);
    Cell xy11 = lerpCell(grid[indexGrid(nearCoords[0], farCoords[1], farCoords[2], gridSize)], 
      grid[indexGrid(farCoords[0], farCoords[1], farCoords[2], gridSize)], lerpFactors.x);

    Cell xyz0 = lerpCell(xy00, xy10, lerpFactors.y);
    Cell xyz1 = lerpCell(xy01, xy11, lerpFactors.y);

    Cell gridVal = lerpCell(xyz0, xyz1, lerpFactors.z);

    if (gridVal.density < 0.0)
      gridVal.density = 0.0;

    return gridVal;
  };

  Cell voxel = getVoxel(x, y, z);
  Cell voxelX = getVoxel(x + 1, y, z);
  Cell voxelY = getVoxel(x, y + 1, z);
  Cell voxelZ = getVoxel(x, y, z + 1);

  ret += lambda_density * sqrt((voxel.density - voxelX.density) * (voxel.density - voxelX.density) +
    (voxel.density - voxelY.density) * (voxel.density - voxelY.density) +
    (voxel.density - voxelZ.density) * (voxel.density - voxelZ.density));

  for (size_t j = offsetof(Cell, sh) / sizeof(Cell::density); j < CellSize; j++)
    ((float*)&ret)[j] = lambda_SH * sqrt(
      (((float*)&voxel)[j] - ((float*)&voxelX)[j]) * (((float*)&voxel)[j] - ((float*)&voxelX)[j]) +
      (((float*)&voxel)[j] - ((float*)&voxelY)[j]) * (((float*)&voxel)[j] - ((float*)&voxelY)[j]) +
      (((float*)&voxel)[j] - ((float*)&voxelZ)[j]) * (((float*)&voxel)[j] - ((float*)&voxelZ)[j]));

  return ret;
}

float TVRegularisationGrad(size_t &x, size_t &y, size_t &z, Cell *&grid, Cell *&grid_d,
  size_t &gridSize)
{
  return __enzyme_autodiff((void *)TVRegularisation,
                           enzyme_const, &x, enzyme_const, &y, enzyme_const, &z,
                           enzyme_dup, &grid, &grid_d, enzyme_const, &gridSize);
}

float RayGridLoss(float4 &ref, float3 &rayPos, float3 &rayDir, double &step,
  const std::vector<size_t> &intersects, const std::vector<OrthoTree::BoundingBox3D> &boxes,
  Cell *&grid, size_t &gridSize, float &width, float &height, bool &greyscale)
{
  float transmittance, cauchy;
  float3 color = RayGridIntersection(rayPos, rayDir, step, intersects, boxes, grid, gridSize,
    transmittance, cauchy, greyscale);

  return abs(color[0] - ref[0]) + abs(color[1] - ref[1]) + abs(color[2] - ref[2]) +
    0.0005f * cauchy + 0.0001f * (log(transmittance + 1e-3) + log(1 - transmittance + 1e-3));
}

float RayGridLossGrad(float4 &ref, float3 &rayPos, float3 &rayDir, double &step, 
  const std::vector<size_t> &intersects, const std::vector<OrthoTree::BoundingBox3D> &boxes,
  Cell *&grid, Cell *&grid_d, size_t &gridSize, float &width, float &height, bool &greyscale)
{
  return __enzyme_autodiff((void *)RayGridLoss,
    enzyme_const, &ref, enzyme_const, &rayPos,
    enzyme_const, &rayDir, enzyme_const, &step, enzyme_const, &intersects,
    enzyme_const, &boxes, enzyme_dup, &grid, &grid_d, enzyme_const, &gridSize,
    enzyme_const, &width, enzyme_const, &height, enzyme_const, &greyscale);
}

void RayMarcherExample::UpsampleGrid()
{
  size_t oldGSize = gridSize;
  gridSize *= 2;
  float step = 1.0f / (float)gridSize;

  std::vector<Cell> newGrid(gridSize * gridSize * gridSize);

#pragma omp parallel for
  for (size_t z = 0; z < gridSize; z++)
    for (size_t y = 0; y < gridSize; y++)
      for (size_t x = 0; x < gridSize; x++)
      {
        float3 coords = (float3(0.0f) + step * (float3(x, y, z) + 0.5f)) * oldGSize;
        // std::cout << coords[0] << ' ' << coords[1] << ' ' << coords[2] << std::endl;

        int3 nearCoords = clamp((int3)coords, int3(0), int3(oldGSize - 1));
        int3 farCoords = clamp((int3)coords + int3(1), int3(0), int3(oldGSize - 1));

        float3 lerpFactors = coords - (float3)nearCoords;

        Cell xy00 = lerpCell(grid[indexGrid(nearCoords[0], nearCoords[1], nearCoords[2], oldGSize)],
          grid[indexGrid(farCoords[0], nearCoords[1], nearCoords[2], oldGSize)], lerpFactors.x);
        Cell xy10 = lerpCell(grid[indexGrid(nearCoords[0], farCoords[1], nearCoords[2], oldGSize)],
          grid[indexGrid(farCoords[0], farCoords[1], nearCoords[2], oldGSize)], lerpFactors.x);
        Cell xy01 = lerpCell(grid[indexGrid(nearCoords[0], nearCoords[1], farCoords[2], oldGSize)],
          grid[indexGrid(farCoords[0], nearCoords[1], farCoords[2], oldGSize)], lerpFactors.x);
        Cell xy11 = lerpCell(grid[indexGrid(nearCoords[0], farCoords[1], farCoords[2], oldGSize)],
          grid[indexGrid(farCoords[0], farCoords[1], farCoords[2], oldGSize)], lerpFactors.x);

        Cell xyz0 = lerpCell(xy00, xy10, lerpFactors.y);
        Cell xyz1 = lerpCell(xy01, xy11, lerpFactors.y);

        Cell gridVal = lerpCell(xyz0, xyz1, lerpFactors.z);

        if (gridVal.density < 0.0)
          gridVal.density = 0.0;

        newGrid[indexGrid(x, y, z, gridSize)] = gridVal;
      }

  grid = newGrid;
}

void RayMarcherExample::UpsampleOctree()
{
  std::cout << "Old octree size = " << boxes.size() << std::endl;
  std::vector<OrthoTree::BoundingBox3D> newBoxes;
  newBoxes.reserve((gridSize - 1) * (gridSize - 1) * (gridSize - 1));

  auto getDensity = [this](OrthoTree::BoundingBox3D box)
  {
    float3 bbMin = float3(box.Min[0], box.Min[1], box.Min[2]);
    float3 bbMax = float3(box.Max[0], box.Max[1], box.Max[2]);

    float3 p = (bbMin + bbMax) / 2.0f;

    float3 lerpFactors = (p - bbMin) / (bbMax - bbMin);

    int3 nearCoords = clamp((int3)(bbMin * gridSize), 0, gridSize - 1);
    int3 farCoords = clamp(nearCoords + 1, 0, gridSize - 1);

    Cell xy00 = lerpCell(grid[indexGrid(nearCoords[0], nearCoords[1], nearCoords[2], gridSize)],
      grid[indexGrid(farCoords[0], nearCoords[1], nearCoords[2], gridSize)], lerpFactors.x);
    Cell xy10 = lerpCell(grid[indexGrid(nearCoords[0], farCoords[1], nearCoords[2], gridSize)],
      grid[indexGrid(farCoords[0], farCoords[1], nearCoords[2], gridSize)], lerpFactors.x);
    Cell xy01 = lerpCell(grid[indexGrid(nearCoords[0], nearCoords[1], farCoords[2], gridSize)],
      grid[indexGrid(farCoords[0], nearCoords[1], farCoords[2], gridSize)], lerpFactors.x);
    Cell xy11 = lerpCell(grid[indexGrid(nearCoords[0], farCoords[1], farCoords[2], gridSize)],
      grid[indexGrid(farCoords[0], farCoords[1], farCoords[2], gridSize)], lerpFactors.x);

    Cell xyz0 = lerpCell(xy00, xy10, lerpFactors.y);
    Cell xyz1 = lerpCell(xy01, xy11, lerpFactors.y);

    Cell gridVal = lerpCell(xyz0, xyz1, lerpFactors.z);
    return gridVal.density;
  };

  for (auto box : boxes)
  {
    double dims[3] = {(box.Max[0] - box.Min[0]) / 2.0f, (box.Max[1] - box.Min[1]) / 2.0f,
      (box.Max[2] - box.Min[2]) / 2.0f};
    {
      auto newBox = box;
      newBox.Max[0] -= dims[0];
      newBox.Max[1] -= dims[1];
      newBox.Max[2] -= dims[2];

      if (getDensity(newBox) > 0.0)
        newBoxes.push_back(newBox);
    }

    {
      auto newBox = box;
      newBox.Min[0] += dims[0];
      newBox.Min[1] += dims[1];
      newBox.Min[2] += dims[2];

      if (getDensity(newBox) > 0.0)
        newBoxes.push_back(newBox);
    }

    {
      auto newBox = box;
      newBox.Min[0] += dims[0];
      newBox.Max[1] -= dims[1];
      newBox.Max[2] -= dims[2];

      if (getDensity(newBox) > 0.0)
        newBoxes.push_back(newBox);
    }

    {
      auto newBox = box;
      newBox.Max[0] -= dims[0];
      newBox.Min[1] += dims[1];
      newBox.Max[2] -= dims[2];

      if (getDensity(newBox) > 0.0)
        newBoxes.push_back(newBox);
    }

    {
      auto newBox = box;
      newBox.Min[0] += dims[0];
      newBox.Min[1] += dims[1];
      newBox.Max[2] -= dims[2];

      if (getDensity(newBox) > 0.0)
        newBoxes.push_back(newBox);
    }

    {
      auto newBox = box;
      newBox.Max[0] -= dims[0];
      newBox.Max[1] -= dims[1];
      newBox.Min[2] += dims[2];

      if (getDensity(newBox) > 0.0)
        newBoxes.push_back(newBox);
    }

    {
      auto newBox = box;
      newBox.Min[0] += dims[0];
      newBox.Max[1] -= dims[1];
      newBox.Min[2] += dims[2];

      if (getDensity(newBox) > 0.0)
        newBoxes.push_back(newBox);
    }

    {
      auto newBox = box;
      newBox.Max[0] -= dims[0];
      newBox.Min[1] += dims[1];
      newBox.Min[2] += dims[2];

      if (getDensity(newBox) > 0.0)
        newBoxes.push_back(newBox);
    }
  }

  boxes = newBoxes;
  octree = OrthoTree::OctreeBoxC::Create(boxes);
  std::cout << "New octree size = " << boxes.size() << std::endl;
}

void RayMarcherExample::RebuildOctree()
{
  std::cout << "Old octree size = " << boxes.size() << std::endl;
  auto oldBoxes = boxes;

  boxes.clear();
  boxes.reserve((gridSize - 1) * (gridSize - 1) * (gridSize - 1));

  for (auto box : oldBoxes)
  {
    float3 bbMin = float3(box.Min[0], box.Min[1], box.Min[2]);
    float3 bbMax = float3(box.Max[0], box.Max[1], box.Max[2]);

    float3 p = (bbMin + bbMax) / 2.0f;

    float3 lerpFactors = (p - bbMin) / (bbMax - bbMin);

    int3 nearCoords = clamp((int3)(bbMin * gridSize), 0, gridSize - 1);
    int3 farCoords = clamp(nearCoords + 1, 0, gridSize - 1);

    Cell xy00 = lerpCell(grid[indexGrid(nearCoords[0], nearCoords[1], nearCoords[2], gridSize)],
      grid[indexGrid(farCoords[0], nearCoords[1], nearCoords[2], gridSize)], lerpFactors.x);
    Cell xy10 = lerpCell(grid[indexGrid(nearCoords[0], farCoords[1], nearCoords[2], gridSize)],
      grid[indexGrid(farCoords[0], farCoords[1], nearCoords[2], gridSize)], lerpFactors.x);
    Cell xy01 = lerpCell(grid[indexGrid(nearCoords[0], nearCoords[1], farCoords[2], gridSize)],
      grid[indexGrid(farCoords[0], nearCoords[1], farCoords[2], gridSize)], lerpFactors.x);
    Cell xy11 = lerpCell(grid[indexGrid(nearCoords[0], farCoords[1], farCoords[2], gridSize)],
      grid[indexGrid(farCoords[0], farCoords[1], farCoords[2], gridSize)], lerpFactors.x);

    Cell xyz0 = lerpCell(xy00, xy10, lerpFactors.y);
    Cell xyz1 = lerpCell(xy01, xy11, lerpFactors.y);

    Cell gridVal = lerpCell(xyz0, xyz1, lerpFactors.z);

    if (gridVal.density > 0.0)
      boxes.push_back(box);
  }

  octree = OrthoTree::OctreeBoxC::Create(boxes);
  std::cout << "New octree size = " << boxes.size() << std::endl;
}

// void RayMarcherExample::LoadModel(std::string densities, std::string sh)
// {
//   std::cout << "Loading model\n";
//   {
//     json densities_data;
//     {
//       std::ifstream f(densities);
//       densities_data = json::parse(f);
//     }

//     std::cout << "Loaded densities from disk\n";

//     for (size_t z = 0; z < gridSize; z++)
//       for (size_t y = 0; y < gridSize; y++)
//         for (size_t x = 0; x < gridSize; x++)
//         {
//           size_t gridIndex = indexGrid(x, y, z, gridSize);
//           grid[gridIndex].density = densities_data[z][y][x][0];
//         }

//     std::cout << "Assigned densities\n";
//   }

//   {
//     json sh_data;
//     {
//       std::ifstream f(sh);
//       sh_data = json::parse(f);
//     }

//     std::cout << "Loaded SH coefficients from disk\n";

//     for (size_t z = 0; z < gridSize; z++)
//       for (size_t y = 0; y < gridSize; y++)
//         for (size_t x = 0; x < gridSize; x++)
//         {
//           size_t gridIndex = indexGrid(x, y, z, gridSize);

//           for (size_t i = 0; i < SH_WIDTH; i++)
//           {
//             grid[gridIndex].sh_r[i] = sh_data[z][y][x][i * 3];
//             grid[gridIndex].sh_g[i] = sh_data[z][y][x][1 + i * 3];
//             grid[gridIndex].sh_b[i] = sh_data[z][y][x][2 + i * 3];
//           }
//         }

//     std::cout << "Assigned SH coefficients\n";
//   }
//   std::cout << "Done loading\n";
// }

static inline float3 EyeRayDir(float x, float y, float4x4 a_mViewProjInv)
{
  float4 pos = float4(2.0f * x - 1.0f, 2.0f * y - 1.0f, 0.0f, 1.0f);
  pos = a_mViewProjInv * pos;
  pos /= pos.w;
  return normalize(to_float3(pos));
}

static inline void transform_ray3f(float4x4 a_mWorldViewInv, float3 *ray_pos, float3 *ray_dir)
{
  float4 rayPosTransformed = a_mWorldViewInv * to_float4(*ray_pos, 1.0f);
  float4 rayDirTransformed = a_mWorldViewInv * to_float4(*ray_dir, 0.0f);

  (*ray_pos) = to_float3(rayPosTransformed);
  (*ray_dir) = to_float3(normalize(rayDirTransformed));
}

float4 RayMarchConstantFog(float tmin, float tmax, float &alpha)
{
  float dt = 0.05f;
  float t = tmin;

  alpha = 1.0f;
  float4 color = float4(0.0f);

  while (t < tmax && alpha > 0.01f)
  {
    float a = 0.025f;
    color += a * alpha * float4(1.0f, 1.0f, 0.0f, 0.0f);
    alpha *= (1.0f - a);
    t += dt;
  }

  return color;
}

static inline uint32_t RealColorToUint32(float4 real_color)
{
  float r = real_color[0] * 255.0f;
  float g = real_color[1] * 255.0f;
  float b = real_color[2] * 255.0f;
  float a = real_color[3] * 255.0f;

  uint32_t red = (uint32_t)r;
  uint32_t green = (uint32_t)g;
  uint32_t blue = (uint32_t)b;
  uint32_t alpha = (uint32_t)a;

  return red | (green << 8) | (blue << 16) | (alpha << 24);
}

static inline float4 Uint32ToRealColor(uint color)
{
  float r = (color & 0x000F) / 255.0f;
  float g = ((color >> 8) & 0x000F) / 255.0f;
  float b = ((color >> 16) & 0x000F) / 255.0f;
  float a = ((color >> 24) & 0x000F) / 255.0f;

  return float4(r, g, b, a);
}

void L1Loss(float *loss, float4 *ref, uint *gen, int width, int height, RayMarcherExample *pImpl,
  const char *fileName, bool greyscale)
{
  *loss = 0.0f;

  std::cout << "Running learning kernel start" << std::endl;
  pImpl->kernel2D_RayMarchGrad(width, height, ref, greyscale);
  std::cout << "Running learning kernel end" << std::endl;

  pImpl->kernel2D_RayMarch(gen, width, height, greyscale);

  LiteImage::SaveBMP(fileName, gen, width, height);

  // #pragma omp parallel for
  // for (int y = 0; y < height; y++)
  //   for (int x = 0; x < width; x++) {
  //     float4 ref_pixel = ref[y * width + x];
  //     float4 gen_pixel = Uint32ToRealColor(gen[y * width + x]);

  //     *loss += abs(ref_pixel[0] - gen_pixel[0]) + abs(ref_pixel[1] - gen_pixel[1]) +
  //        abs(ref_pixel[2] - gen_pixel[2]);
  //   }
}

void RayMarcherExample::kernel2D_RayMarch(uint32_t *out_color, uint32_t width, uint32_t height,
  bool greyscale)
{
#pragma omp parallel for
  for (uint32_t y = 0; y < height; y++)
  {
    for (uint32_t x = 0; x < width; x++)
    {
      float3 rayDir = EyeRayDir((float(x) + 0.5f) / float(width), (float(y) + 0.5f) / float(height),
        m_worldViewProjInv);
      float3 rayPos = float3(0.0f, 0.0f, 0.0f);

      transform_ray3f(m_worldViewInv, &rayPos, &rayDir);

      std::vector<size_t> intersects = octree.RayIntersectedAll(
        OrthoTree::Point3D{rayPos[0], rayPos[1], rayPos[2]}, 
        OrthoTree::Point3D{rayDir[0], rayDir[1], rayDir[2]}, 0.0);

      float4 resColor(0.0f);
      if (intersects.size())
      // if (RaySphereIntersection(rayPos, rayDir, 0.05))
      {
        float step = (bb.max[0] - bb.min[0]) / (gridSize);
        float transmittance, cauchy;
        float3 color = RayGridIntersection(rayPos, rayDir, step, intersects, boxes, grid.data(), 
          gridSize, transmittance, cauchy, greyscale);
        resColor = float4(color[0], color[1], color[2], 1.0f);

        // float alpha = 1.0f;
        // resColor = float4(1.0f);
        // resColor = RayMarchConstantFog(tNearAndFar.x, tNearAndFar.y, alpha);
      }

      out_color[y * width + x] = RealColorToUint32(resColor);
    }
  }
}

void RayMarcherExample::kernel2D_RayMarchGrad(uint32_t width, uint32_t height, float4 *res, 
  bool greyscale)
{
#pragma omp parallel for
  for (uint32_t y = 0; y < height; y++)
  {
    for (uint32_t x = 0; x < width; x++)
    {
      float3 rayDir = EyeRayDir((float(x) + 0.5f) / float(width), (float(y) + 0.5f) / float(height),
        m_worldViewProjInv);
      float3 rayPos = float3(0.0f, 0.0f, 0.0f);

      transform_ray3f(m_worldViewInv, &rayPos, &rayDir);

      std::vector<size_t> intersects = octree.RayIntersectedAll(
        OrthoTree::Point3D{rayPos[0], rayPos[1], rayPos[2]}, 
        OrthoTree::Point3D{rayDir[0], rayDir[1], rayDir[2]}, 0.0);

      // std::cout << x << ' ' << y << ' ';
      // for (auto t: intersects)
      //   std::cout << t << ' ';
      // std::cout << std::endl;

      if (intersects.size())
      // if (RaySphereIntersection(rayPos, rayDir, 0.05))
      {

        double step = (bb.max[0] - bb.min[0]) / (gridSize);

        float4 res_pixel_color = res[y * width + x];
        float4 res_pixel;
        if (greyscale)
        {
          res_pixel[0] = 0.11f * res_pixel_color[2] + 0.59f * res_pixel_color[1] +
            0.3f * res_pixel_color[0];
          res_pixel[1] = 0.11f * res_pixel_color[2] + 0.59f * res_pixel_color[1] +
            0.3f * res_pixel_color[0];
          res_pixel[2] = 0.11f * res_pixel_color[2] + 0.59f * res_pixel_color[1] +
            0.3f * res_pixel_color[0];
        }
        else
          res_pixel = res_pixel_color;

        Cell *grid_data = grid.data();
        Cell *grid_in_data = grid_d.data();

        float fwidth = width;
        float fheight = height;

        float dL1 = RayGridLossGrad(res_pixel, rayPos, rayDir, step, intersects, boxes, grid_data,
          grid_in_data, gridSize, fwidth, fheight, greyscale);
      }
    }
  }

  float avgVal = 0.0f;
  float maxVal = 0.0f;
  for (size_t i = 0; i < gridSize; i++)
  {
    Cell currCell = grid_d[i];

    for (size_t j = 0; j < CellSize; j++)
      avgVal += ((float*)&currCell)[j];

    for (size_t j = 0; j < CellSize; j++)
      maxVal += std::max(maxVal, ((float*)&currCell)[j]);
  }

  std::cout << "avgVal = " << avgVal / float(gridSize) << std::endl;
  std::cout << "maxVal = " << maxVal << std::endl;
}

void RayMarcherExample::kernel2D_TVGrad()
{
#pragma omp parallel for
  for (size_t z = 0; z < gridSize - 1; z++)
    for (size_t y = 0; y < gridSize - 1; y++)
      for (size_t x = 0; x < gridSize - 1; x++)
      {
        Cell *grid_data = grid.data();
        Cell *grid_in_data = grid_d.data();

        TVRegularisationGrad(x, y, z, grid_data, grid_in_data, gridSize);
      }
}

void RayMarcherExample::RayMarch(uint32_t *out_color, uint32_t width, uint32_t height)
{
  // auto start = std::chrono::high_resolution_clock::now();
  // kernel2D_RayMarch(out_color, width, height);
  // rayMarchTime = float(std::chrono::duration_cast<std::chrono::microseconds>(
  //   std::chrono::high_resolution_clock::now() - start).count())/1000.f;
}

void RayMarcherExample::GetExecutionTime(const char *a_funcName, float a_out[4])
{
  if (std::string(a_funcName) == "RayMarch")
    a_out[0] = rayMarchTime;
}
