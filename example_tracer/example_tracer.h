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
    grid.resize(gridSize * gridSize * gridSize, {0.1, {1, 1, 1, 1, 1, 1, 1, 1, 1},
      {1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}});
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

  void optimizerStep(RayMarcherExample* pImpl_d, float learning_rate) {
    for (int z = 0; z < gridSize; z++)
    for (int y = 0; y < gridSize; y++)
      for (int x = 0; x < gridSize; x++) {
        Cell* cell = &grid[z * gridSize * gridSize + y * gridSize + x];
        Cell* cell_d = &pImpl_d->grid[z * gridSize * gridSize + y * gridSize + x];

        cell->density += cell_d->density * learning_rate;
        for (int i = 0; i < 9; i++) {
          cell->sh_r[i] += cell_d->sh_r[i] * learning_rate;
          cell->sh_g[i] += cell_d->sh_g[i] * learning_rate;
          cell->sh_b[i] += cell_d->sh_b[i] * learning_rate;
        }
      }
  }

  void LoadModel(std::string densities, std::string sh);

  void SetWorldViewMatrix(const float4x4& a_mat) {m_worldViewInv = inverse4x4(a_mat);}
  void SetWorldViewMProjatrix(const float4x4& a_mat) {m_worldViewProjInv = inverse4x4(a_mat);}

  void kernel2D_RayMarch(uint32_t* out_color, uint32_t width, uint32_t height, float4* L);
  void kernel2D_RayMarchGrad(uint32_t width, uint32_t height, float4* res, RayMarcherExample* pImpl_d);
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

  BoundingBox bb;
};

void L1Loss(float* loss, uint* ref, uint* gen, int width, int height, RayMarcherExample* pImpl, RayMarcherExample* pImpl_d, const char* fileName);
// void L1LossGrad(float* loss, float* loss_d, uint* ref, uint* gen, int width, int height, RayMarcherExample* pImpl, RayMarcherExample* pImpl_d);