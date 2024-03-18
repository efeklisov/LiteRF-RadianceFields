#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include <cstdint>

#include "LiteMath.h"
using namespace LiteMath;

#include <Octree/octree.h>

const size_t SH_WIDTH = 9;

const float C0 = 0.28209479177387814;
const float C1 = 0.4886025119029199;
const float C2[5] = {
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
};

struct Cell
{
  float density;
  float RGB[3];
  float sh[SH_WIDTH];
};

const size_t CellSize = sizeof(Cell) / sizeof(Cell::density);

struct BoundingBox
{
  float3 min;
  float3 max;
};

class RayMarcherExample // : public IRenderAPI
{
public:
  RayMarcherExample()
  {
    const float4x4 view = lookAt(float3(0, 1.5, -3), float3(0, 0, 0), float3(0, 1, 0));
    const float4x4 proj = perspectiveMatrix(90.0f, 1.0f, 0.1f, 100.0f);
    m_worldViewInv = inverse4x4(view);
    m_worldViewProjInv = inverse4x4(proj);
  }

  void InitGrid(const float _gridSize)
  {
    std::cout << "Init grid start" << std::endl;

    gridSize = _gridSize;
    grid.resize(gridSize * gridSize * gridSize);

#pragma omp parallel for
    for (size_t z = 0; z < gridSize; z++)
      for (size_t y = 0; y < gridSize; y++)
        for (size_t x = 0; x < gridSize; x++)
        {
          size_t i = x + y * gridSize + z * gridSize * gridSize;

          grid[i].density = 0.01;
          for (size_t j = 1; j < CellSize; j++)
            ((float*)&grid[i])[j] = 0.1;
        }

    boxes.resize((gridSize - 1) * (gridSize - 1) * (gridSize - 1));

#pragma omp parallel for
    for (size_t z = 0; z < gridSize - 1; z++)
      for (size_t y = 0; y < gridSize - 1; y++)
        for (size_t x = 0; x < gridSize - 1; x++)
        {
          size_t i = x + y * (gridSize - 1) + z * (gridSize - 1) * (gridSize - 1);

          boxes[i] = OrthoTree::BoundingBox3D{
            {(float)x / (float)gridSize, (float)y / (float)gridSize, (float)z / (float)gridSize},
            {(float)(x + 1) / (float)gridSize, (float)(y + 1) / (float)gridSize, 
              (float)(z + 1) / (float)gridSize}
          };
        }

    std::cout << "Creating octree start" << std::endl;
    octree = OrthoTree::OctreeBoxC::Create(boxes);
    std::cout << "Creating octree end" << std::endl;

    std::cout << "Init grid end" << std::endl;
  }

  void InitGrad()
  {
    grid_d.resize(gridSize * gridSize * gridSize);
  }

  void SetBoundingBox(const float3 boxMin, const float3 boxMax)
  {
    bb.min = boxMin;
    bb.max = boxMax;
  }

  void zeroGrad()
  {
#pragma omp parallel for
    for (size_t i = 0; i < (grid_d.size() * CellSize); i++)
      ((float*)grid_d.data())[i] = 0.0f;
  }

  void optimizerInit()
  {
    const size_t vecSize = gridSize * gridSize * gridSize * CellSize;

    // momentum.resize(vecSize);
    // m_GSquare.resize(vecSize);
    // std::fill(momentum.begin(), momentum.end(), 0.0);
    // std::fill(m_GSquare.begin(), m_GSquare.end(), 0.0);

    lr = 0.15f;
    beta_1 = 0.9f;
    beta_2 = 0.999f;
    eps = 1e-8;
    V = std::vector<float>(vecSize, 0);
    S = std::vector<float>(vecSize, 0);
  }

  void optimizerStep(int iter)
  {
    const size_t vecSize = gridSize * gridSize * gridSize * CellSize;

    float *gridPtr = (float *)grid.data();
    float *gridPtr_d = (float *)grid_d.data();

    // int factorGamma = iter/100 + 1;
    // const float alpha   = 0.5;
    // const float beta    = 0.25;
    // const float gamma   = 0.25/factorGamma;

    // for(size_t i=0;i<momentum.size();i++)
    // {
    //   auto gradVal = gridPtr_d[i];
    //   momentum [i] = momentum[i]*beta + gradVal*(float(1.0)-beta);
    //   m_GSquare[i] = float(2.0)*(m_GSquare[i]*alpha + (gradVal*gradVal)*(float(1.0)-alpha)); 
    //   // does not works without 2.0
    // }
    // std::cout << std::endl;

    // for (size_t i=0;i<momentum.size();i++)
    //   gridPtr[i] -= (gamma*momentum[i]/(std::sqrt(m_GSquare[i] + epsilon)));

    const auto b1 = std::pow(beta_1, iter + 1);
    const auto b2 = std::pow(beta_2, iter + 1);
    for (size_t i = 0; i < vecSize; i++)
    {
      auto g = gridPtr_d[i];
      V[i] = beta_1 * V[i] + (1 - beta_1) * g;
      auto Vh = V[i] / ((1) - b1);
      S[i] = beta_2 * S[i] + (1 - beta_2) * g * g;
      auto Sh = S[i] / ((1) - b2);
      gridPtr[i] -= lr * Vh / (std::sqrt(Sh) + eps);
    }
  }

  void optimizerStepDensity(int iter)
  {
    const size_t vecSize = gridSize * gridSize * gridSize * CellSize;

    float *gridPtr = (float *)grid.data();
    float *gridPtr_d = (float *)grid_d.data();

    const auto b1 = std::pow(beta_1, iter + 1);
    const auto b2 = std::pow(beta_2, iter + 1);
    for (size_t i = 0; i < vecSize; i += CellSize)
    {
      auto g = gridPtr_d[i];
      V[i] = beta_1 * V[i] + (1 - beta_1) * g;
      auto Vh = V[i] / ((1) - b1);
      S[i] = beta_2 * S[i] + (1 - beta_2) * g * g;
      auto Sh = S[i] / ((1) - b2);
      gridPtr[i] -= lr * Vh / (std::sqrt(Sh) + eps);
    }
  }

  void optimizerStepSH(int iter)
  {
    const size_t vecSize = gridSize * gridSize * gridSize * CellSize;

    float *gridPtr = (float *)grid.data();
    float *gridPtr_d = (float *)grid_d.data();

    const auto b1 = std::pow(beta_1, iter + 1);
    const auto b2 = std::pow(beta_2, iter + 1);
    for (size_t i = 0; i < vecSize; i++)
    {
      if ((i % CellSize) == 0)
        continue;

      auto g = gridPtr_d[i];
      V[i] = beta_1 * V[i] + (1 - beta_1) * g;
      auto Vh = V[i] / ((1) - b1);
      S[i] = beta_2 * S[i] + (1 - beta_2) * g * g;
      auto Sh = S[i] / ((1) - b2);
      gridPtr[i] -= lr * Vh / (std::sqrt(Sh) + eps);
    }
  }

  void UpsampleGrid();
  void UpsampleOctree();
  void RebuildOctree();

  void LoadModel(std::string densities, std::string sh);

  void SetWorldViewMatrix(const float4x4 &a_mat) { m_worldViewInv = inverse4x4(a_mat); }
  void SetWorldViewMProjatrix(const float4x4 &a_mat) { m_worldViewProjInv = inverse4x4(a_mat); }

  void kernel2D_RayMarch(uint32_t *out_color, uint32_t width, uint32_t height, 
    bool greyscale = false);
  void kernel2D_RayMarchGrad(uint32_t width, uint32_t height, float4 *res, bool greyscale = false);
  void kernel2D_TVGrad();
  void RayMarch(uint32_t *out_color [[size("width*height")]], uint32_t width, uint32_t height);

  void CommitDeviceData() {} // will be overriden in generated class
  void UpdateMembersPlainData() {} // will be overriden in generated class (optional function)
  // // will be overriden in generated class (optional function)
  // virtual void UpdateMembersVectorData() {}
  // // will be overriden in generated class (optional function)
  // virtual void UpdateMembersTexureData() {}
  // will be overriden in generated class
  void GetExecutionTime(const char *a_funcName, float a_out[4]);

  float4x4 m_worldViewProjInv;
  float4x4 m_worldViewInv;
  float rayMarchTime;

  std::vector<Cell> grid;
  std::vector<Cell> grid_d;
  size_t gridSize;

  std::vector<OrthoTree::BoundingBox3D> boxes;
  OrthoTree::OctreeBoxC octree;

  // std::vector<float> momentum;
  // std::vector<float> m_GSquare;
  // float epsilon = 1e-8;

  float lr, beta_1, beta_2, eps;
  std::vector<float> V;
  std::vector<float> S;

  BoundingBox bb;
};

void L1Loss(float *loss, float4 *ref, uint *gen, int width, int height, RayMarcherExample *pImpl, 
  const char *fileName, bool greyscale = false);