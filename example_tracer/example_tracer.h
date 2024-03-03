#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include <cstdint>

#include "LiteMath.h"
using namespace LiteMath;

const float C0 = 0.28209479177387814;
const float C1 = 0.4886025119029199;
const float C2[5] = {
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
};

struct Cell {
  float density;
  float sh_r[9];
  float sh_g[9];
  float sh_b[9];
};

struct BoundingBox {
  float3 min;
  float3 max;
};

class RayMarcherExample // : public IRenderAPI
{
public:

  RayMarcherExample()
  {
    const float4x4 view = lookAt(float3(0,1.5,-3), float3(0,0,0), float3(0,1,0)); // pos, look_at, up
    const float4x4 proj = perspectiveMatrix(90.0f, 1.0f, 0.1f, 100.0f);
    m_worldViewInv      = inverse4x4(view); 
    m_worldViewProjInv  = inverse4x4(proj); 
  }

  void InitGrid(const float _gridSize) {
    gridSize = _gridSize;
    grid.resize(gridSize * gridSize * gridSize, {0.01, {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1},
      {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}, {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}});
    grid_grad.resize(gridSize  * gridSize * gridSize);
  }
  void SetBoundingBox(const float3 boxMin, const float3 boxMax) {
    bb.min = boxMin;
    bb.max = boxMax;
  }

  void zeroGrad() {
    for (int z = 0; z < gridSize; z++)
      for (int y = 0; y < gridSize; y++)
        for (int x = 0; x < gridSize; x++) {
          Cell* cell = &grid[z * gridSize * gridSize + y * gridSize + x];
          cell->density = 0.0f;

          for (int i = 0; i < 9; i++) {
            cell->sh_r[i] = 0.0f;
            cell->sh_g[i] = 0.0f;
            cell->sh_b[i] = 0.0f;
          }
        }
  }

  void optimizerInit() {
    const size_t vecSize = gridSize * gridSize * gridSize * sizeof(Cell) / 4;

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

  void optimizerStep(RayMarcherExample* pImpl_d, int iter) {
    const size_t vecSize = gridSize * gridSize * gridSize * sizeof(Cell) / 4;

    float* gridPtr = (float*)grid.data();
    float* gridPtr_d = (float*)pImpl_d->grid.data();

    // int factorGamma = iter/100 + 1;
    // const float alpha   = 0.5;
    // const float beta    = 0.25;
    // const float gamma   = 0.25/factorGamma;
    
    // for(size_t i=0;i<momentum.size();i++)
    // {
    //   auto gradVal = gridPtr_d[i];
    //   momentum [i] = momentum[i]*beta + gradVal*(float(1.0)-beta);
    //   m_GSquare[i] = float(2.0)*(m_GSquare[i]*alpha + (gradVal*gradVal)*(float(1.0)-alpha)); // does not works without 2.0
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

  void LoadModel(std::string densities, std::string sh);

  void SetWorldViewMatrix(const float4x4& a_mat) {m_worldViewInv = inverse4x4(a_mat);}
  void SetWorldViewMProjatrix(const float4x4& a_mat) {m_worldViewProjInv = inverse4x4(a_mat);}

  void kernel2D_RayMarch(uint32_t* out_color, uint32_t width, uint32_t height, float4* L);
  void kernel2D_RayMarchGrad(uint32_t width, uint32_t height, float4* res, float4* L, RayMarcherExample* pImpl_d);
  void RayMarch(uint32_t* out_color [[size("width*height")]], uint32_t width, uint32_t height);  

  void CommitDeviceData() {}                                       // will be overriden in generated class
  void UpdateMembersPlainData() {}                                 // will be overriden in generated class (optional function)
  //virtual void UpdateMembersVectorData() {}                              // will be overriden in generated class (optional function)
  //virtual void UpdateMembersTexureData() {}                              // will be overriden in generated class (optional function)
  void GetExecutionTime(const char* a_funcName, float a_out[4]);   // will be overriden in generated class

  float4x4 m_worldViewProjInv;
  float4x4 m_worldViewInv;
  float    rayMarchTime;

  std::vector<Cell> grid;
  std::vector<Cell> grid_grad;
  size_t gridSize;

  // std::vector<float> momentum; 
  // std::vector<float> m_GSquare;
  // float epsilon = 1e-8;

  float lr, beta_1, beta_2, eps;
  std::vector<float> V;
  std::vector<float> S;

  BoundingBox bb;
};

void L1Loss(float* loss, uint* ref, uint* gen, int width, int height, RayMarcherExample* pImpl, RayMarcherExample* pImpl_d, const char* fileName);
// void L1LossGrad(float* loss, float* loss_d, uint* ref, uint* gen, int width, int height, RayMarcherExample* pImpl, RayMarcherExample* pImpl_d);