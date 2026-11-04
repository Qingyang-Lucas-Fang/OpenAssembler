/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

#include "cutlass/gemm/threadblock/presum_detail.h"
#include "cutlass/gemm/device/strassen_decls.h"

#define MAX(x,y) (((x)<(y)) ? (y) : (x))

using namespace cutlass::gemm::threadblock;

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
//This is not called right now.
  class StrassenMiGroup_,
  int Stages,
  class TileShape_,
  class ElementA_,
  class StrideA_,
  class ElementB_,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_,
  class PresumClusterShape,
  class PresumTileShapeA_,
  class PresumGmemTiledCopyA,
  class PresumSmemLayoutAtomA,
  class PresumSmemCopyAtomA,
  class PresumTileShapeB_,
  class PresumGmemTiledCopyB,
  class PresumSmemLayoutAtomB,
  class PresumSmemCopyAtomB,
  class PresumOpt_,
  class ProblemShape_>
struct CollectiveStrassenMma<
    StrassenMiGroup_,
    MainloopSm80CpAsyncUnpredicated<Stages>,
    TileShape_,
    ElementA_,
    StrideA_,
    ElementB_,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_,
    PresumClusterShape,
    PresumTileShapeA_,
    PresumGmemTiledCopyA,
    PresumSmemLayoutAtomA,
    PresumSmemCopyAtomA,
    PresumTileShapeB_,
    PresumGmemTiledCopyB,
    PresumSmemLayoutAtomB,
    PresumSmemCopyAtomB,
    PresumOpt_,
    ProblemShape_
  >
{
  //
  // Type Aliases
  //
  using StrassenMiGroup = StrassenMiGroup_;
  using PresumOpt = PresumOpt_;
  using ProblemShape = ProblemShape_;
  using DispatchPolicy = MainloopSm80CpAsyncUnpredicated<Stages>;
  using TileShape = TileShape_;
  using ElementA = ElementA_;
  using StrideA = StrideA_;
  using ElementB = ElementB_;
  using StrideB = StrideB_;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using FragmentC = decltype(partition_fragment_C(TiledMma(), take<0,2>(TileShape{})));
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;
  static constexpr bool IsStrassenLayout = false;
  using PresumShape = GemmShape<size<0>(TileShape{}), size<1>(TileShape{}), 1>;
  using PresumShapeA = PresumShape;
  using PresumShapeB = PresumShape;
  using PresumTileShapeA = PresumTileShapeA_;
  using PresumTileShapeB = PresumTileShapeB_;
  using PresumVecTypeA = Array<ElementA, (size<0>(PresumTileShapeA{})*sizeof(ElementA))/sizeof(ElementA)>;
  using PresumVecTypeB = Array<ElementA, (size<0>(PresumTileShapeB{})*sizeof(ElementA))/sizeof(ElementA)>;
  using PresumStoreVecType = Array<ElementA, 16/sizeof(ElementA)>;
  using PresumComputeType = ElementAccumulator;
  using PresumComputeToIOTypeA = NumericArrayConverter<ElementA, PresumComputeType, PresumVecTypeA::kElements>;
  using PresumIOToComputeTypeA = NumericArrayConverter<PresumComputeType, ElementA, PresumVecTypeA::kElements>;
  using PresumComputeToIOTypeB = NumericArrayConverter<ElementA, PresumComputeType, PresumVecTypeB::kElements>;
  using PresumIOToComputeTypeB = NumericArrayConverter<PresumComputeType, ElementA, PresumVecTypeB::kElements>;
  static const int NumMMAThreads = size(TiledMma{});
  static constexpr int PresumStoreWarpSize = NumMMAThreads/4;
  static const int kPresumThreads = NumMMAThreads;///4 * (size<0>(PresumTileShapeA{})/2);//TODO: Pass this from StrassenGemmKernel
  using PresumGlobalIteratorA = PresumDetail::GlobalIterator<ElementA, kPresumThreads,  
                                                             PresumShape, PresumStoreVecType, false, false>;
  using PresumGlobalIteratorB = PresumDetail::GlobalIterator<ElementB, kPresumThreads, 
                                                             PresumShape, PresumStoreVecType, true, false>;
  static const uint kPresumComputeIterationsA = size<0>(TileShape{})/size<0>(PresumTileShapeA{});
  static const uint kPresumComputeIterationsB = size<0>(TileShape{})/size<0>(PresumTileShapeB{});

  static const bool is_fused = StrassenMiGroup::hasM0() && StrassenMiGroup::hasM1();
  static const bool is_fused_m2_m3 = StrassenMiGroup::hasM2() && StrassenMiGroup::hasM3();

  // Follow the change in TestSmall: TileShape switch to CtaShape
  // For sm80 arch, CtaShape should equal to TileShape
  using CtaShape_MNK = TileShape;
  static const int PresumStages = DispatchPolicy::Stages;

  static_assert(cute::rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(cute::rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));

  using PresumSmemLayoutA = decltype(make_layout(make_shape(shape<0>(PresumTileShapeA{}), shape<1>(PresumTileShapeA{}), Int<PresumStages>{}), Step<_1,_2,_3>{}));
  using PresumSmemLayoutB = decltype(make_layout(make_shape(shape<0>(PresumTileShapeB{}), shape<1>(PresumTileShapeB{}), Int<PresumStages>{}), Step<_1,_2,_3>{}));

  using PresumSmemShapeA__ = decltype(make_shape(shape<0>(PresumTileShapeB{}), shape<1>(PresumTileShapeB{}), Int<PresumStages>{}));
  using PresumSmemLayoutA__ = decltype(make_layout(make_shape(shape<0>(PresumTileShapeA{}), shape<1>(PresumTileShapeA{})), LayoutRight{}));
  using PresumSmemLayoutB__ = decltype(make_layout(make_shape(shape<0>(PresumTileShapeB{}), shape<1>(PresumTileShapeB{})), LayoutRight{}));

  static_assert(DispatchPolicy::Stages >= 2, "CpAsync mainloop must have at least 2 stages in the pipeline.");

  struct SharedStorage
  {
    cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutA>> smem_a;
    cute::array_aligned<ElementB, cute::cosize_v<SmemLayoutB>> smem_b;
  };

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A;
    StrideA dA;
    ElementB const* ptr_B;
    StrideB dB;
  };

  // Device side kernel params
  using Params = Arguments;

  //
  // Methods
  //

  CollectiveStrassenMma() = default;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& _, Arguments const& args, void* workspace) {
    (void) workspace;
    return args;
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  template <
    class FrgTensorD,
    class TensorA,
    class TensorB,
    class FrgTensorC,
    class KTileIterator,
    class ResidueMNK
  >
  CUTLASS_DEVICE void
  operator() (
      FrgTensorD &accum,
      TensorA gA,
      TensorB gB,
      FrgTensorC const &src_accum,
      KTileIterator k_tile_iter, int k_tile_count,
      ResidueMNK residue_mnk,
      int thread_idx,
      char *smem_buf)
  {
    using namespace cute;

    static_assert(is_rmem<FrgTensorD>::value, "D tensor must be rmem resident.");
    static_assert(is_gmem<TensorA>::value,    "A tensor must be gmem resident.");
    static_assert(is_gmem<TensorB>::value,    "B tensor must be gmem resident.");
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(cute::rank(SmemLayoutA{}) == 3,
      "MainloopSm80CpAsync must have a pipeline mode in the smem layout.");
    static_assert(cute::rank(SmemLayoutB{}) == 3,
      "MainloopSm80CpAsync must have a pipeline mode in the smem layout.");

    // Construct shared memory tiles
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

    CUTE_STATIC_ASSERT_V(size<0>(gA) == size<0>(sA));                          // BLK_M
    CUTE_STATIC_ASSERT_V(size<1>(gA) == size<1>(sA));                          // BLK_K
    CUTE_STATIC_ASSERT_V(size<0>(gB) == size<0>(sB));                          // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(gB) == size<1>(sB));                          // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(sA) == size<1>(sB));                          // BLK_K
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));        // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));        // PIPE

    // Partition the copying of A and B tiles across the threads
    GmemTiledCopyA gmem_tiled_copy_A;
    GmemTiledCopyB gmem_tiled_copy_B;
    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);

    Tensor tAgA = gmem_thr_copy_A.partition_S(gA);                             // (ACPY,ACPY_M,ACPY_K,k)
    Tensor tAsA = gmem_thr_copy_A.partition_D(sA);                             // (ACPY,ACPY_M,ACPY_K,PIPE)
    Tensor tBgB = gmem_thr_copy_B.partition_S(gB);                             // (BCPY,BCPY_N,BCPY_K,k)
    Tensor tBsB = gmem_thr_copy_B.partition_D(sB);                             // (BCPY,BCPY_N,BCPY_K,PIPE)

    //
    // PREDICATES
    //

    (void) residue_mnk;
    //assert(residue_mnk == make_tuple(0,0,0));

    //
    // PREFETCH
    //

    // Start async loads for all pipes but the last
    CUTLASS_PRAGMA_UNROLL
    for (int k_pipe = 0; k_pipe < DispatchPolicy::Stages-1; ++k_pipe) {
      copy(gmem_tiled_copy_A, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,k_pipe));
      copy(gmem_tiled_copy_B, tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,k_pipe));
      cp_async_fence();
      --k_tile_count;
      if (k_tile_count > 0) { ++k_tile_iter; }
    }

    //
    // MMA Atom partitioning
    //

    // Tile MMA compute thread partitions and allocate accumulators
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));                     // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));                     // (MMA,MMA_N,MMA_K)

    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(src_accum));                 // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(src_accum));                 // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                      // MMA_K
    CUTE_STATIC_ASSERT_V(size(gmem_tiled_copy_A) == size(tiled_mma));
    CUTE_STATIC_ASSERT_V(size(gmem_tiled_copy_B) == size(tiled_mma));

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(thread_idx);
    Tensor tCsA            = smem_thr_copy_A.partition_S(sA);                  // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);                   // (CPY,CPY_M,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));            // CPY_K

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(thread_idx);
    Tensor tCsB            = smem_thr_copy_B.partition_S(sB);                  // (CPY,CPY_N,CPY_K,PIPE)
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);                   // (CPY,CPY_N,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));            // CPY_K

    //
    // PIPELINED MAIN LOOP
    //

    // Current pipe index in smem to read from
    int smem_pipe_read  = 0;
    // Current pipe index in smem to write to
    int smem_pipe_write = DispatchPolicy::Stages-1;

    Tensor tCsA_p = tCsA(_,_,_,smem_pipe_read);
    Tensor tCsB_p = tCsB(_,_,_,smem_pipe_read);

    // Size of the register pipeline
    auto K_BLOCK_MAX = size<2>(tCrA);

    // PREFETCH register pipeline
    if (K_BLOCK_MAX > 1) {
      // Wait until our first prefetched tile is loaded in
      cp_async_wait<DispatchPolicy::Stages-2>();
      __syncthreads();

      // Prefetch the first rmem from the first k-tile
      copy(smem_tiled_copy_A, tCsA_p(_,_,Int<0>{}), tCrA_copy_view(_,_,Int<0>{}));
      copy(smem_tiled_copy_B, tCsB_p(_,_,Int<0>{}), tCrB_copy_view(_,_,Int<0>{}));
    }

    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > -(DispatchPolicy::Stages-1))
    {
      // Pipeline the outer products with a static for loop.
      //
      // Note, the for_each() function is required here to ensure `k_block` is of type Int<x>.
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block)
      {
        if (k_block == K_BLOCK_MAX - 1)
        {
          // Slice the smem_pipe_read smem
          tCsA_p = tCsA(_,_,_,smem_pipe_read);
          tCsB_p = tCsB(_,_,_,smem_pipe_read);

          // Commit the smem for smem_pipe_read
          cp_async_wait<DispatchPolicy::Stages-2>();
          __syncthreads();
        }

        // Load A, B shmem->regs for k_block+1
        auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;  // static
        copy(smem_tiled_copy_A, tCsA_p(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
        copy(smem_tiled_copy_B, tCsB_p(_,_,k_block_next), tCrB_copy_view(_,_,k_block_next));
        // Copy gmem to smem before computing gemm on each k-pipe
        if (k_block == 0)
        {
          copy(gmem_tiled_copy_A, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,smem_pipe_write));
          copy(gmem_tiled_copy_B, tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,smem_pipe_write));
          cp_async_fence();

          // Advance the tile
          --k_tile_count;
          if (k_tile_count > 0) { ++k_tile_iter; }

          // Advance the pipe -- Doing it here accounts for K_BLOCK_MAX = 1 (no rmem pipe)
          smem_pipe_write = smem_pipe_read;
          ++smem_pipe_read;
          smem_pipe_read = (smem_pipe_read == DispatchPolicy::Stages) ? 0 : smem_pipe_read;
        }

        // Transform before compute
        cute::transform(tCrA(_,_,k_block), TransformA{});
        cute::transform(tCrB(_,_,k_block), TransformB{});
        // Thread-level register gemm for k_block
        cute::gemm(tiled_mma, accum, tCrA(_,_,k_block), tCrB(_,_,k_block), src_accum);
      });

    }

    cp_async_wait<0>();
    __syncthreads();
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class StrassenMiGroup_,
  int Stages,
  class ClusterShape_,
  class TileShape_,
  class ElementA_,
  class StrideA_,
  class ElementB_,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_,
  class PresumClusterShape,
  class PresumTileShapeA_,
  class PresumGmemTiledCopyA,
  class PresumSmemLayoutAtomA,
  class PresumSmemCopyAtomA,
  class PresumTileShapeB_,
  class PresumGmemTiledCopyB,
  class PresumSmemLayoutAtomB,
  class PresumSmemCopyAtomB,
  class PresumOpt_,
  class ProblemShape_>
struct CollectiveStrassenMma<
    StrassenMiGroup_,
    MainloopSm80CpAsync<
      Stages,
      ClusterShape_>,
    TileShape_,
    ElementA_,
    StrideA_,
    ElementB_,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_,
    PresumClusterShape,
    PresumTileShapeA_,
    PresumGmemTiledCopyA,
    PresumSmemLayoutAtomA,
    PresumSmemCopyAtomA,
    PresumTileShapeB_,
    PresumGmemTiledCopyB,
    PresumSmemLayoutAtomB,
    PresumSmemCopyAtomB,
    PresumOpt_,
    ProblemShape_
  >
{
  //
  // Type Aliases
  //
  using StrassenMiGroup = StrassenMiGroup_;
  using PresumOpt = PresumOpt_;
  using DispatchPolicy = MainloopSm80CpAsync<
                          Stages,
                          ClusterShape_>;
  using TileShape = TileShape_;
  // Follow the change in TestSmall: TileShape switch to CtaShape
  // In legacy arch, it should be same
  using CtaShape_MNK = TileShape;
  using ElementA = ElementA_;
  using StrideA = StrideA_;
  using ElementB = ElementB_;
  using StrideB = StrideB_;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using FragmentC = decltype(partition_fragment_C(TiledMma(), take<0,2>(TileShape{})));
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;
  static constexpr bool IsStrassenLayout = false;
  using PresumShape = GemmShape<size<0>(TileShape{}), size<1>(TileShape{}), 1>;
  using PresumShapeA = PresumShape;
  using PresumShapeB = PresumShape;
  using PresumTileShapeA = PresumTileShapeA_;
  using PresumTileShapeB = PresumTileShapeB_;
  using PresumVecTypeA = Array<ElementA, 16/sizeof(ElementA)>;
  using PresumVecTypeB = Array<ElementA, 16/sizeof(ElementA)>;
  using PresumStoreVecType = Array<ElementA, 16/sizeof(ElementA)>;
  using PresumComputeType = ElementAccumulator;
  using PresumComputeToIOTypeA = NumericArrayConverter<ElementA, PresumComputeType, PresumVecTypeA::kElements>;
  using PresumIOToComputeTypeA = NumericArrayConverter<PresumComputeType, ElementA, PresumVecTypeA::kElements>;
  using PresumComputeToIOTypeB = NumericArrayConverter<ElementA, PresumComputeType, PresumVecTypeB::kElements>;
  using PresumIOToComputeTypeB = NumericArrayConverter<PresumComputeType, ElementA, PresumVecTypeB::kElements>;
  static const int NumMMAThreads = size(TiledMma{});
  static constexpr int PresumStoreWarpSize = NumMMAThreads/4;
  static const int kPresumThreads = NumMMAThreads;///4 * (size<0>(PresumTileShapeA{})/2);//TODO: Pass this from StrassenGemmKernel
  using PresumGlobalIteratorA = PresumDetail::GlobalIterator<ElementA, kPresumThreads,  
                                                             PresumShape, PresumStoreVecType, false, false>;
  using PresumGlobalIteratorB = PresumDetail::GlobalIterator<ElementB, kPresumThreads, 
                                                             PresumShape, PresumStoreVecType, true, false>;
  static const uint kPresumComputeIterationsA = size<0>(TileShape{})/size<0>(PresumTileShapeA{});
  static const uint kPresumComputeIterationsB = size<0>(TileShape{})/size<0>(PresumTileShapeB{});

  static const bool is_fused = StrassenMiGroup::hasM0() && StrassenMiGroup::hasM1();
  static const bool is_fused_m2_m3 = StrassenMiGroup::hasM2() && StrassenMiGroup::hasM3();
  static const int PresumStages = DispatchPolicy::Stages;

  static_assert(cute::rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(cute::rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));

  using PresumSmemLayoutA = decltype(make_layout(make_shape(shape<0>(PresumTileShapeA{}), shape<1>(PresumTileShapeA{}), Int<PresumStages>{}), Step<_1,_2,_3>{}));
  using PresumSmemLayoutB = decltype(make_layout(make_shape(shape<0>(PresumTileShapeB{}), shape<1>(PresumTileShapeB{}), Int<PresumStages>{}), Step<_1,_2,_3>{}));

  using PresumSmemShapeA__ = decltype(make_shape(shape<0>(PresumTileShapeB{}), shape<1>(PresumTileShapeB{}), Int<PresumStages>{}));
  using PresumSmemLayoutA__ = decltype(make_layout(make_shape(shape<0>(PresumTileShapeA{}), shape<1>(PresumTileShapeA{})), LayoutRight{}));
  using PresumSmemLayoutB__ = decltype(make_layout(make_shape(shape<0>(PresumTileShapeB{}), shape<1>(PresumTileShapeB{})), LayoutRight{}));

  static_assert(DispatchPolicy::Stages >= 2, "CpAsync mainloop must have at least 2 stages in the pipeline.");

  static const size_t PresumSingleStageSizeA = size<0>(PresumTileShapeA{})*size<1>(PresumTileShapeA{});
  static const size_t PresumSingleStageSizeB = size<0>(PresumTileShapeB{})*size<1>(PresumTileShapeB{});

  struct SharedStorageNoPresum
  {
    cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutA>> smem_a;
    cute::array_aligned<ElementB, cute::cosize_v<SmemLayoutB>> smem_b;
    struct {
      cute::array_aligned<ElementA, 8> _;
    } smem_presum;
  };

  struct SharedStoragePresum
  {
    cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutA>> smem_a;
    cute::array_aligned<ElementB, cute::cosize_v<SmemLayoutB>> smem_b;

    struct {
      cute::array_aligned<ElementA, PresumSingleStageSizeA*PresumStages> smem0;
      cute::array_aligned<ElementA, PresumSingleStageSizeA*PresumStages> smem1;
      cute::array_aligned<ElementA, PresumSingleStageSizeA*PresumStages> smem2;
      cute::array_aligned<ElementA, PresumSingleStageSizeA*PresumStages> smem3;
    } smem_presum;
  };

  using SharedStorage = cute::conditional_t<StrassenMiGroup::hasM0() && (StrassenMiGroup::AllPresums::computeAnyAPresum() || StrassenMiGroup::AllPresums::computeAnyBPresum()),
                                            SharedStoragePresum, SharedStorageNoPresum>;

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A;
    StrideA dA;
    ElementB const* ptr_B;
    StrideB dB;
    __restrict__ ElementA* ptr_presum_A;
    __restrict__ ElementB* ptr_presum_B;

    uint32_t presum_tile_log_multiplier_a_;
    uint32_t presum_tile_log_multiplier_b_;
    uint32_t presum_tile_log_divider_a_;
    uint32_t presum_tile_log_divider_b_;

    CUTLASS_HOST_DEVICE
    uint32_t get_presum_tile_log_multiplier_a() const {
      return (PresumOpt::FixedPresumTileMultilplierLogA != UINT32_MAX) ?
              PresumOpt::FixedPresumTileMultilplierLogA :
              presum_tile_log_multiplier_a_;
    }

    CUTLASS_HOST_DEVICE
    uint32_t get_presum_tile_log_multiplier_b() const {
      return (PresumOpt::FixedPresumTileMultilplierLogB != UINT32_MAX) ?
              PresumOpt::FixedPresumTileMultilplierLogB :
              presum_tile_log_multiplier_b_;
    }

    //Disable log divider
    CUTLASS_HOST_DEVICE
    uint32_t get_presum_tile_log_divider_a() const {
      return 0;//(PresumOpt::FixedPresumTileDividerLogA != UINT32_MAX) ? 
              // PresumOpt::FixedPresumTileDividerLogA :
              // presum_tile_log_divider_a_;
    }

    CUTLASS_HOST_DEVICE
    uint32_t get_presum_tile_log_divider_b() const {
      return 0;//(PresumOpt::FixedPresumTileDividerLogB != UINT32_MAX) ? 
              // PresumOpt::FixedPresumTileDividerLogB :
              // presum_tile_log_divider_b_;
    }

    Arguments() : Arguments(nullptr, StrideA{}, nullptr, StrideB{}) {}

    Arguments(ElementA const* ptr_A, StrideA dA, ElementB const* ptr_B, StrideB dB, ElementA* ptr_presum_A = nullptr, ElementB* ptr_presum_B = nullptr, uint32_t presum_tile_log_multiplier_a = 0, uint32_t presum_tile_log_multiplier_b = 0, uint32_t presum_tile_log_divider_a = 0, uint32_t presum_tile_log_divider_b = 0) :
          ptr_A(ptr_A), dA(dA), ptr_B(ptr_B), dB(dB), ptr_presum_A(ptr_presum_A), ptr_presum_B(ptr_presum_B), presum_tile_log_multiplier_a_(presum_tile_log_multiplier_a), presum_tile_log_multiplier_b_(presum_tile_log_multiplier_b), presum_tile_log_divider_a_(presum_tile_log_divider_a), presum_tile_log_divider_b_(presum_tile_log_divider_b)
          {}

    template<typename Other>
    Arguments(const Other& other) : ptr_A(other.ptr_A), dA(other.dA), ptr_B(other.ptr_B), dB(other.dB),
                                   ptr_presum_A(other.ptr_presum_A), ptr_presum_B(other.ptr_presum_B), presum_tile_log_multiplier_a_(other.presum_tile_log_multiplier_a_), presum_tile_log_multiplier_b_(other.presum_tile_log_multiplier_b_), presum_tile_log_divider_a_(other.presum_tile_log_divider_a_), presum_tile_log_divider_b_(other.presum_tile_log_divider_b_)
    {}
  };

  CUTLASS_HOST_DEVICE
  static uint32_t get_presum_log_multiplier(int k, int m_or_n) {
    //TODO: We want k/2 is a divisor of Number of threads (which is usually 256 for us)
    if (k <= m_or_n) return 0;
    int multiplier = (k + m_or_n - 1)/m_or_n;
    if (multiplier > 8) {
      return 4;
    } else if (multiplier > 4) {
      return 3; //multiply with 8
    } else if (multiplier > 2) {
      return 2; //multiply with 4
    } else if (multiplier > 1) {
      return 1; //multiply with 2
    }
    return 0;
  }

  CUTLASS_HOST_DEVICE
  static uint32_t get_presum_log_divider(int k, int m_or_n) {
    //TODO: We want k/2 is a divisor of Number of threads (which is usually 256 for us)
    if (k >= m_or_n) return 0;
    int divider = m_or_n/k;

    if (divider >= 8) {
      return 3; //divide with 8
    } else if (divider >= 4) {
      return 2; //divide with 4
    } else if (divider >= 2) {
      return 1; //divide with 2
    } else {
      return 0; //divide with 1
    }
  }

  // Device side kernel params
  using Params = Arguments;

  //
  // Methods
  //

  CollectiveStrassenMma() = default;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, ElementA* ptr_presum_a, ElementB* ptr_presum_b, Arguments const& args, void* workspace) {
    auto [M,N,K,L] = problem_shape;

    auto presum_tile_log_multiplier_a = get_presum_log_multiplier(K, N);
    auto presum_tile_log_multiplier_b = get_presum_log_multiplier(K, M);
    auto presum_tile_log_divider_a = get_presum_log_divider(K, N);
    auto presum_tile_log_divider_b = get_presum_log_divider(K, M);

    auto ptr_presum_A = ptr_presum_a;
    auto ptr_presum_B = ptr_presum_b;

    (void) workspace;
    return {
      args.ptr_A,
      args.dA,
      args.ptr_B,
      args.dB,
      ptr_presum_A,
      ptr_presum_B,
      presum_tile_log_multiplier_a,
      presum_tile_log_multiplier_b,
      presum_tile_log_divider_a, //TODO: Specialize to the value when needed
      presum_tile_log_divider_b
    };
  }

  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, Params const& mainloop_params) const {
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;
    auto halfM = M/2; auto halfN = N/2; auto halfK = K/2;

    const int32_t init_L = 1;

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    Tensor mA_mkl = make_tensor(make_gmem_ptr(mainloop_params.ptr_A), make_shape(M,K,L), mainloop_params.dA); //(m,k,l)
    Tensor mB_nkl = make_tensor(make_gmem_ptr(mainloop_params.ptr_B), make_shape(N,K,L), mainloop_params.dB); //(n,k,l)

    // Make tiled views, defer the slice
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});  // (BLK_M,BLK_K,m,k,l)
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});  // (BLK_N,BLK_K,n,k,l)

    auto& stride_a = mainloop_params.dA;
    auto& stride_b = mainloop_params.dB;

    Tensor presum_mA_mkl = make_tensor(make_gmem_ptr(mainloop_params.ptr_presum_A), make_shape(4*halfM,halfK,init_L), make_stride(halfK, get<1>(stride_a), get<2>(stride_a))); //(m,k,l)
    Tensor presum_mB_nkl = make_tensor(make_gmem_ptr(mainloop_params.ptr_presum_B), make_shape(halfN,4*halfK,init_L), make_stride(get<0>(stride_b), halfN, get<2>(stride_b))); //(n,k,l)

    // Make tiled views, defer the slice
    Tensor gA00_mkl = local_tile(domain_offset(make_coord(0, 0, 0), mA_mkl), TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});
    Tensor gA01_mkl = local_tile(domain_offset(make_coord(0, halfK, 0), mA_mkl), TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});
    Tensor gA10_mkl = local_tile(domain_offset(make_coord(halfM, 0, 0), mA_mkl), TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});
    Tensor gA11_mkl = local_tile(domain_offset(make_coord(halfM, halfK, 0), mA_mkl), TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});

    Tensor gB00_nkl = local_tile(domain_offset(make_coord(0, 0, 0), mB_nkl), TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});        // (BLK_N,BLK_K,n,k,l)
    Tensor gB01_nkl = local_tile(domain_offset(make_coord(halfN, 0, 0), mB_nkl), TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});
    Tensor gB10_nkl = local_tile(domain_offset(make_coord(0, halfK, 0), mB_nkl), TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});
    Tensor gB11_nkl = local_tile(domain_offset(make_coord(halfN, halfK, 0), mB_nkl), TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});

    Tensor gPresumA02 = local_tile(domain_offset(make_coord(StrassenMiGroup::AllPresums::indexAPresum(StrassenMiGroup::APresums::A02)*halfM, 0, 0), presum_mA_mkl), TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});
    Tensor gPresumS1 = local_tile(domain_offset(make_coord(StrassenMiGroup::AllPresums::indexAPresum(StrassenMiGroup::APresums::S1)*halfM, 0, 0), presum_mA_mkl), TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});
    Tensor gPresumS2 = local_tile(domain_offset(make_coord(StrassenMiGroup::AllPresums::indexAPresum(StrassenMiGroup::APresums::S2)*halfM, 0, 0), presum_mA_mkl), TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});
    Tensor gPresumA1S2 = local_tile(domain_offset(make_coord(StrassenMiGroup::AllPresums::indexAPresum(StrassenMiGroup::APresums::A1S2)*halfM, 0, 0), presum_mA_mkl), TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});

    Tensor gPresumB31 = local_tile(domain_offset(make_coord(0, StrassenMiGroup::AllPresums::indexBPresum(StrassenMiGroup::BPresums::B31)*halfK, 0), presum_mB_nkl), TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});
    Tensor gPresumB10 = local_tile(domain_offset(make_coord(0, StrassenMiGroup::AllPresums::indexBPresum(StrassenMiGroup::BPresums::B10)*halfK, 0), presum_mB_nkl), TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});
    Tensor gPresumS3 = local_tile(domain_offset(make_coord(0, StrassenMiGroup::AllPresums::indexBPresum(StrassenMiGroup::BPresums::S3)*halfK, 0), presum_mB_nkl), TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});
    Tensor gPresumS3B2 = local_tile(domain_offset(make_coord(0, StrassenMiGroup::AllPresums::indexBPresum(StrassenMiGroup::BPresums::S3B2)*halfK, 0), presum_mB_nkl), TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});

    return cute::make_tuple(gA00_mkl, gA01_mkl, gA10_mkl, gA11_mkl,
                            gPresumA02, gPresumS1, gPresumS2, gPresumA1S2,
                            gB00_nkl, gB01_nkl, gB10_nkl, gB11_nkl,
                            gPresumB31, gPresumB10, gPresumS3, gPresumS3B2);
  }

  template<typename Tuple>
  CUTLASS_DEVICE auto
  get_inputs(Tuple const& load_inputs, int sub_m_idx = 0) {
    auto load_inputs_a = take<0, 8  >(load_inputs);
    auto load_inputs_b = take<8, 8+8>(load_inputs);

    constexpr bool IsFusedM2M3 = StrassenMiGroup::hasM2() && StrassenMiGroup::hasM3();
    constexpr bool IsFusedM2M3M6 = StrassenMiGroup::hasM6() && IsFusedM2M3;
    constexpr bool IsFusedM4M5 = StrassenMiGroup::hasM4() && StrassenMiGroup::hasM5();
    constexpr int MiCount = int(StrassenMiGroup::hasM0()) + int(StrassenMiGroup::hasM1()) +
                            int(StrassenMiGroup::hasM2()) + int(StrassenMiGroup::hasM3()) +
                            int(StrassenMiGroup::hasM4()) + int(StrassenMiGroup::hasM5()) +
                            int(StrassenMiGroup::hasM6());

    if constexpr (StrassenMiGroup::numMs() == 1) {
      if constexpr (StrassenMiGroup::hasM0()) {
        return make_tuple(get<MmaStrassen::APresums::A0>(load_inputs_a),
                          get<MmaStrassen::BPresums::B0>(load_inputs_b));
      } else if constexpr (StrassenMiGroup::hasM1()) {
        return make_tuple(get<MmaStrassen::APresums::A1>(load_inputs_a),
                          get<MmaStrassen::BPresums::B2>(load_inputs_b));
      } else if constexpr (StrassenMiGroup::hasM2()) {
        return make_tuple(get<MmaStrassen::APresums::S2>(load_inputs_a),
                          get<MmaStrassen::BPresums::S3>(load_inputs_b));
      } else if constexpr (StrassenMiGroup::hasM3()) {
        return make_tuple(get<StrassenMiGroup::APresums::A02>(load_inputs_a),
                          get<StrassenMiGroup::BPresums::B31>(load_inputs_b));
      } else if constexpr (StrassenMiGroup::hasM4()) {
        return make_tuple(get<StrassenMiGroup::APresums::S1>(load_inputs_a),
                          get<StrassenMiGroup::BPresums::B10>(load_inputs_b));
      } else if constexpr (StrassenMiGroup::hasM5()) {
        return make_tuple(get<StrassenMiGroup::APresums::A1S2>(load_inputs_a),
                          get<StrassenMiGroup::BPresums::B3>(load_inputs_b));
      } else {
        return make_tuple(get<StrassenMiGroup::APresums::A3>(load_inputs_a),
                          get<StrassenMiGroup::BPresums::S3B2>(load_inputs_b));
      }
    } else {
      if constexpr (StrassenMiGroup::hasM0()) {
        if (!is_fused || sub_m_idx == 0) {
          return make_tuple(get<MmaStrassen::APresums::A0>(load_inputs_a),
                            get<MmaStrassen::BPresums::B0>(load_inputs_b));
        }
      }
      if constexpr (StrassenMiGroup::hasM1()) {
        if (!is_fused || sub_m_idx == 1) {
          return make_tuple(get<MmaStrassen::APresums::A1>(load_inputs_a),
                            get<MmaStrassen::BPresums::B2>(load_inputs_b));
        }
      }
      if constexpr (StrassenMiGroup::hasM2()) {
        if (!IsFusedM2M3 || sub_m_idx == 0) {
          return make_tuple(get<MmaStrassen::APresums::S2>(load_inputs_a),
                            get<MmaStrassen::BPresums::S3>(load_inputs_b));
        }
      }
      if constexpr (StrassenMiGroup::hasM3()) {
        if (!IsFusedM2M3 || sub_m_idx == 1) {
          return make_tuple(get<StrassenMiGroup::APresums::A02>(load_inputs_a),
                            get<StrassenMiGroup::BPresums::B31>(load_inputs_b));
        }
      }
      if constexpr (StrassenMiGroup::hasM4()) {
        if (!IsFusedM4M5 || sub_m_idx == 0) {
          return make_tuple(get<StrassenMiGroup::APresums::S1>(load_inputs_a),
                            get<StrassenMiGroup::BPresums::B10>(load_inputs_b));
        }
      }
      if constexpr (StrassenMiGroup::hasM5()) {
        if (!IsFusedM4M5 || sub_m_idx == 1) {
          return make_tuple(get<StrassenMiGroup::APresums::A1S2>(load_inputs_a),
                            get<StrassenMiGroup::BPresums::B3>(load_inputs_b));
        }
      }
      if constexpr (StrassenMiGroup::hasM6()) {
        if (!IsFusedM2M3M6 || sub_m_idx == 2) {
          return make_tuple(get<StrassenMiGroup::APresums::A3>(load_inputs_a),
                            get<StrassenMiGroup::BPresums::S3B2>(load_inputs_b));
        }
      }

      return make_tuple(get<0>(load_inputs_a), get<0>(load_inputs_b));
    }
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  template <
    class BlockCoord,
    class FrgTensorD,
    class TensorA,
    class TensorB,
    class FrgTensorC,
    class KTileIterator,
    class ResidueMNK,
    class ProblemShape_MNKL
  >
  CUTLASS_DEVICE void
  operator() (
      Params const& mainloop_params,
      BlockCoord const& blk_coord,
      ProblemShape_MNKL const& problem_shape,
      FrgTensorD &accum,
      TensorA gA,                   // (BLK_M, BLK_K, K_TILES)
      TensorB gB,                   // (BLK_N, BLK_K, K_TILES)
      FrgTensorC const &src_accum,
      KTileIterator k_tile_iter, int k_tile_count,
      ResidueMNK residue_mnk,
      int thread_idx,
      char *smem_buf)
  {
    using namespace cute;

    static_assert(is_rmem<FrgTensorD>::value, "D tensor must be rmem resident.");
    static_assert(is_gmem<TensorA>::value,    "A tensor must be gmem resident.");
    static_assert(is_gmem<TensorB>::value,    "B tensor must be gmem resident.");
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(cute::rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute::rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");

    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
    auto [M,N,K,L] = problem_shape;
    auto halfM = M/2, halfN = N/2, halfK = K/2, halfL = L;

    // Construct shared memory tiles
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

    CUTE_STATIC_ASSERT_V(size<0>(gA) == size<0>(sA));                          // BLK_M
    CUTE_STATIC_ASSERT_V(size<1>(gA) == size<1>(sA));                          // BLK_K
    CUTE_STATIC_ASSERT_V(size<0>(gB) == size<0>(sB));                          // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(gB) == size<1>(sB));                          // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(sA) == size<1>(sB));                          // BLK_K
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));        // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));        // PIPE

    // Shift tensor so residue_k is at origin (Can't read any k_coord < residue_k)
    // This aligns the tensor with BLK_K for all but the 0th k_tile
    gA = cute::domain_offset(make_coord(0, get<2>(residue_mnk), 0), gA);
    gB = cute::domain_offset(make_coord(0, get<2>(residue_mnk), 0), gB);

    // Partition the copying of A and B tiles across the threads
    GmemTiledCopyA gmem_tiled_copy_A;
    GmemTiledCopyB gmem_tiled_copy_B;
    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);

    Tensor tAgA = gmem_thr_copy_A.partition_S(gA);                             // (ACPY,ACPY_M,ACPY_K,k)
    Tensor tAsA = gmem_thr_copy_A.partition_D(sA);                             // (ACPY,ACPY_M,ACPY_K,PIPE)
    Tensor tBgB = gmem_thr_copy_B.partition_S(gB);                             // (BCPY,BCPY_N,BCPY_K,k)
    Tensor tBsB = gmem_thr_copy_B.partition_D(sB);                             // (BCPY,BCPY_N,BCPY_K,PIPE)

    //
    // PREDICATES
    //

    // Allocate predicate tensors for m and n
    Tensor tApA = make_tensor<bool>(make_shape(size<1>(tAsA), size<2>(tAsA)), Stride<_1,_0>{});
    Tensor tBpB = make_tensor<bool>(make_shape(size<1>(tBsB), size<2>(tBsB)), Stride<_1,_0>{});

    // Construct identity layout for sA and sB
    Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cB = make_identity_tensor(make_shape(size<0>(sB), size<1>(sB)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tAcA = gmem_thr_copy_A.partition_S(cA);                             // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tBcB = gmem_thr_copy_B.partition_S(cB);                             // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Set predicates for m bounds
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < size<0>(tApA); ++m) {
      tApA(m,0) = get<0>(tAcA(0,m,0)) < get<0>(residue_mnk);  // blk_m coord < residue_m
    }
    // Set predicates for n bounds
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < size<0>(tBpB); ++n) {
      tBpB(n,0) = get<0>(tBcB(0,n,0)) < get<1>(residue_mnk);  // blk_n coord < residue_n
    }

    //
    // PREFETCH
    //

    // Clear the smem tiles to account for predicated off loads
    clear(tAsA);
    clear(tBsB);

    int block_idx = m_coord + n_coord * 1;///params.grid_tiled_shape.m();
    int n_coord_div = (n_coord * (1 << mainloop_params.get_presum_tile_log_multiplier_a())) % (1<<mainloop_params.get_presum_tile_log_divider_a());
    int new_n_coord = (n_coord * (1 << mainloop_params.get_presum_tile_log_multiplier_a())) >> mainloop_params.get_presum_tile_log_divider_a();

    PresumGlobalIteratorA iter_PresumA(
      (ElementA*)mainloop_params.ptr_A, get<0>(mainloop_params.dA),
      {M, K},
      {m_coord * size<0>(TileShape{}), n_coord * size<1>(TileShape{}) * (1 << mainloop_params.get_presum_tile_log_multiplier_a())},
      block_idx, {0, 0}, thread_idx, {0, halfK}, {halfM, 0}, {halfM, halfK}
    );

    PresumGlobalIteratorB iter_PresumB(
      (ElementB*)mainloop_params.ptr_B, get<1>(mainloop_params.dB),
      {K, N},
      {m_coord * size<0>(TileShape{}) * (1 << mainloop_params.get_presum_tile_log_multiplier_b()), n_coord * size<1>(TileShape{})},
      block_idx, {0, 0}, thread_idx, {0, halfN}, {halfK, 0}, {halfK, halfN}
    );

    PresumGlobalIteratorA iter_PresumA_M(
      mainloop_params.ptr_presum_A, halfK,
      {halfM, halfK},
      {m_coord * size<0>(TileShape{}) + ((n_coord_div * size<0>(TileShape{})) >> mainloop_params.get_presum_tile_log_divider_a()),
        new_n_coord * size<1>(TileShape{})},
      block_idx, {0, 0}, thread_idx%kPresumThreads, {1*halfM, 0}, {2*halfM, 0}, {3*halfM, 0}
    );
    PresumGlobalIteratorB iter_PresumB_M(
      mainloop_params.ptr_presum_B, halfN,
      {halfK, halfN},
      {(m_coord * (1 << mainloop_params.get_presum_tile_log_multiplier_b()) * (size<0>(TileShape{}))) >> mainloop_params.get_presum_tile_log_divider_b(),
       n_coord * size<1>(TileShape{})},
      block_idx, {0, 0}, thread_idx%kPresumThreads, {1*halfK, 0}, {2*halfK, 0}, {3*halfK, 0}
    );

    using PresumSharedIterator = PresumDetail::SharedIterator<ElementA, PresumVecTypeA, NumMMAThreads,
                                                              4,
                                                              PresumSingleStageSizeB,
                                                              PresumStages, 4>;
    PresumSharedIterator sharedPreSums((ElementA*)&storage.smem_presum, thread_idx);

    // Start async loads for 0th k-tile, where we take care of the k residue
    if (false) {
      constexpr int k_pipe = 0;

      Tensor tAgAk = tAgA(_,_,_,*k_tile_iter);
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<2>(tAsA); ++k) {
        if (get<1>(tAcA(0,0,k)) >= -get<2>(residue_mnk)) {      // blk_k coord < residue_k (gA shifted)
          copy_if(gmem_tiled_copy_A, tApA(_,k), tAgAk(_,_,k), tAsA(_,_,k,k_pipe));
        }
      }
      Tensor tBgBk = tBgB(_,_,_,*k_tile_iter);
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<2>(tBsB); ++k) {
        if (get<1>(tBcB(0,0,k)) >= -get<2>(residue_mnk)) {      // blk_k coord < residue_k (gB shifted)
          copy_if(gmem_tiled_copy_B, tBpB(_,k), tBgBk(_,_,k), tBsB(_,_,k,k_pipe));
        }
      }
      cp_async_fence();
      ++k_tile_iter;
      --k_tile_count;
    }

    int presumIter = 0;
    auto presumAComputeLoads = StrassenMiGroup::AllPresums::APresumComputeLoads();
    auto presumBComputeLoads = StrassenMiGroup::AllPresums::BPresumComputeLoads();
    const bool need_presum_A = presumAComputeLoads.numAccess() > 0;
    const bool need_presum_B = presumBComputeLoads.numAccess() > 0;
    const int presumComputeIterationsA = kPresumComputeIterationsA * (1 << mainloop_params.get_presum_tile_log_multiplier_a());
    const int presumComputeIterationsB = kPresumComputeIterationsB * (1 << mainloop_params.get_presum_tile_log_multiplier_b());

    auto presum_load = [&] (bool is_prologue) {
      if (need_presum_A && presumIter < presumComputeIterationsA*1 - (PresumStages - 1)) {
          int presum_tile = (presumIter + ((is_prologue) ? 0 : PresumStages-1))/kPresumComputeIterationsA;
          iter_PresumA.reset(presum_tile);
          iter_PresumA.set_iteration(presumIter+ ((is_prologue) ? 0 : PresumStages-1) - presum_tile*kPresumComputeIterationsA);
          uint presum_write_stage = (presumIter + ((is_prologue) ? 0 : PresumStages-1))% PresumStages;
          // printf("1073 %d : %d %d %d ; %d %p\n", StrassenMiGroup::getMi(), presumIter, presum_tile, kPresumComputeIterationsA, presum_write_stage, sharedPreSums.get(0, presum_write_stage, 0));

          if (presumAComputeLoads.hasAccess(MmaStrassen::APresums::A0)) {
            PresumDetail::cp_async_presum(sharedPreSums.get(presumAComputeLoads.index(MmaStrassen::APresums::A0), presum_write_stage, 0),
                                          iter_PresumA.get(0), iter_PresumA_M.validTB() && iter_PresumA_M.valid());
            // if (threadIdx.x == 0 && m_coord == 0 && n_coord == 0) {
            //   auto a0 = *iter_PresumA.get(0);
            //   printf("1002 %d,%d : %d %d %f; %p %d %d\n", iter_PresumA.row, iter_PresumA.col, presumIter, presum_write_stage, a0[0],
            //     iter_PresumA.get(0), iter_PresumA.validTB(), iter_PresumA.valid());
            // }
          }
          if (presumAComputeLoads.hasAccess(MmaStrassen::APresums::A1)) {
            PresumDetail::cp_async_presum(sharedPreSums.get(presumAComputeLoads.index(MmaStrassen::APresums::A1), presum_write_stage, 0),
                                          iter_PresumA.get(1), iter_PresumA_M.validTB() && iter_PresumA_M.valid());
          }
          if (presumAComputeLoads.hasAccess(MmaStrassen::APresums::A2)) {
            PresumDetail::cp_async_presum(sharedPreSums.get(presumAComputeLoads.index(MmaStrassen::APresums::A2), presum_write_stage, 0),
                                          iter_PresumA.get(2), iter_PresumA_M.validTB() && iter_PresumA_M.valid());
          }
          if (presumAComputeLoads.hasAccess(MmaStrassen::APresums::A3)) {
            PresumDetail::cp_async_presum(sharedPreSums.get(presumAComputeLoads.index(MmaStrassen::APresums::A3), presum_write_stage, 0),
                                          iter_PresumA.get(3), iter_PresumA_M.validTB() && iter_PresumA_M.valid());
          }

          // iter_PresumA.inc();
        } else if (need_presum_B and 
                    presumIter < presumComputeIterationsA + presumComputeIterationsB - (PresumStages - 1)) {
          // iter_PresumB.reset();
          // iter_PresumB.row += (presumIter + Base::kStages - 1 - kPresumComputeIterations) * iter_PresumB_M.row_increment();
          uint presum_write_stage = (presumIter + ((is_prologue) ? 0 : PresumStages-1))% PresumStages;
          int presum_tile = (presumIter - presumComputeIterationsA + ((is_prologue) ? 0 : PresumStages-1))/kPresumComputeIterationsB;
          iter_PresumB.reset(presum_tile);
          iter_PresumB.set_iteration(presumIter - presumComputeIterationsA - presum_tile*kPresumComputeIterationsB + ((is_prologue) ? 0 : PresumStages-1));
          if (presumBComputeLoads.hasAccess(MmaStrassen::BPresums::B0)) {
            PresumDetail::cp_async_presum(sharedPreSums.get(presumBComputeLoads.index(MmaStrassen::BPresums::B0), presum_write_stage, 0),
                                          iter_PresumB.get(0), iter_PresumB_M.validTB() && iter_PresumB_M.valid());
          }
          if (presumBComputeLoads.hasAccess(MmaStrassen::BPresums::B1)) {
            PresumDetail::cp_async_presum(sharedPreSums.get(presumBComputeLoads.index(MmaStrassen::BPresums::B1), presum_write_stage, 0),
                                          iter_PresumB.get(1), iter_PresumB_M.validTB() && iter_PresumB_M.valid());
          }
          if (presumBComputeLoads.hasAccess(MmaStrassen::BPresums::B2)) {
            PresumDetail::cp_async_presum(sharedPreSums.get(presumBComputeLoads.index(MmaStrassen::BPresums::B2), presum_write_stage, 0),
                                          iter_PresumB.get(2), iter_PresumB_M.validTB() && iter_PresumB_M.valid());
          }
          if (presumBComputeLoads.hasAccess(MmaStrassen::BPresums::B3)) {
            PresumDetail::cp_async_presum(sharedPreSums.get(presumBComputeLoads.index(MmaStrassen::BPresums::B3), presum_write_stage, 0),
                                          iter_PresumB.get(3), iter_PresumB_M.validTB() && iter_PresumB_M.valid());
          }
          // iter_PresumB.inc();
        }
    };

    auto presum_compute_and_store = [&] () {
      if (presumIter < presumComputeIterationsA) {
        //This code above mac_loop_iter gives some improvement.
        //Changes done after commit: 853df006e0f2bfad3313460b2fcfdabb15d31067
        uint presum_read_stage = presumIter % PresumStages;
        uint presum_tile = presumIter/kPresumComputeIterationsA;
        iter_PresumA_M.reset(presum_tile);
        iter_PresumA_M.set_iteration(presumIter - presum_tile * kPresumComputeIterationsA);
        if (iter_PresumA_M.validTB() && iter_PresumA_M.valid())
        for (int v = 0; v < 1; v += 1) {
          // iter_PresumA_M.reset();
          // iter_PresumA_M.row += presumIter * iter_PresumA_M.row_increment();

          PresumVecTypeA a0; a0.clear();
          PresumVecTypeA a1; a1.clear();
          PresumVecTypeA a2; a2.clear();
          PresumVecTypeA a3; a3.clear();

          if (presumAComputeLoads.hasAccess(MmaStrassen::APresums::A0))
            a0 = PresumDetail::shared_load_128b<PresumVecTypeA>(sharedPreSums.get(presumAComputeLoads.index(MmaStrassen::APresums::A0), presum_read_stage, 0));
          if (presumAComputeLoads.hasAccess(MmaStrassen::APresums::A1))
            a1 = PresumDetail::shared_load_128b<PresumVecTypeA>(sharedPreSums.get(presumAComputeLoads.index(MmaStrassen::APresums::A1), presum_read_stage, 0));
          if (presumAComputeLoads.hasAccess(MmaStrassen::APresums::A2))
            a2 = PresumDetail::shared_load_128b<PresumVecTypeA>(sharedPreSums.get(presumAComputeLoads.index(MmaStrassen::APresums::A2), presum_read_stage, 0));
          if (presumAComputeLoads.hasAccess(MmaStrassen::APresums::A3))
            a3 = PresumDetail::shared_load_128b<PresumVecTypeA>(sharedPreSums.get(presumAComputeLoads.index(MmaStrassen::APresums::A3), presum_read_stage, 0));

          PresumIOToComputeTypeA presum_io_to_compute_type;
          PresumComputeToIOTypeA presum_compute_to_io_type;

          auto s1   = presum_io_to_compute_type(a2) + presum_io_to_compute_type(a3);
          auto s2   = s1 - presum_io_to_compute_type(a0);
          auto a02  = presum_io_to_compute_type(a0) - presum_io_to_compute_type(a2);
          auto a1s2 = presum_io_to_compute_type(a1) - s2;
          // if (StrassenMiGroup::hasM0() && StrassenMiGroup::Level == 2 &&
          //     threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0)
          //     printf("1623 %d ; %f %f %f %f\n", iter_PresumA_M.stride, float(a0[0]), float(a1[0]), float(a2[0]), float(a3[0]));
          // auto s1   = a2 + a3;
          // auto s2   = s1 - a0;
          // auto a02  = a0 - a2;
          // auto a1s2 = a1 - s2;

          using AllPresums = typename StrassenMiGroup::AllPresums;

          if (AllPresums::doesComputeA(MmaStrassen::APresums::A02)) {
            arch::global_store<PresumVecTypeA, sizeof(PresumVecTypeA)>(presum_compute_to_io_type(a02), iter_PresumA_M.get(AllPresums::indexAPresum(MmaStrassen::APresums::A02)),
                                                                      iter_PresumA_M.validTB() && iter_PresumA_M.valid());
          }
          if (AllPresums::doesComputeA(MmaStrassen::APresums::S1)) {
            arch::global_store<PresumVecTypeA, sizeof(PresumVecTypeA)>(presum_compute_to_io_type(s1), iter_PresumA_M.get(AllPresums::indexAPresum(MmaStrassen::APresums::S1)),
                                                                      iter_PresumA_M.validTB() && iter_PresumA_M.valid());
          }
          if (AllPresums::doesComputeA(MmaStrassen::APresums::S2)) {
            arch::global_store<PresumVecTypeA, sizeof(PresumVecTypeA)>(presum_compute_to_io_type(s2), iter_PresumA_M.get(AllPresums::indexAPresum(MmaStrassen::APresums::S2)),
                                                                      iter_PresumA_M.validTB() && iter_PresumA_M.valid());
          }
          if (AllPresums::doesComputeA(MmaStrassen::APresums::A1S2)) {
            arch::global_store<PresumVecTypeA, sizeof(PresumVecTypeA)>(presum_compute_to_io_type(a1s2), iter_PresumA_M.get(AllPresums::indexAPresum(MmaStrassen::APresums::A1S2)),
                                                                      iter_PresumA_M.validTB() && iter_PresumA_M.valid());
          }

          iter_PresumA_M.inc();
        }
      } else if (need_presum_B and presumIter < presumComputeIterationsA + presumComputeIterationsB) {
        uint presum_read_stage = presumIter % PresumStages;
        uint presum_tile = (presumIter - presumComputeIterationsA)/kPresumComputeIterationsB;
        iter_PresumB_M.reset(presum_tile);
        iter_PresumB_M.set_iteration(presumIter - presumComputeIterationsA - presum_tile * kPresumComputeIterationsB);
        if (iter_PresumB_M.validTB() and iter_PresumB_M.valid())
        for (int v = 0; v < 1; v += 1) {
          // iter_PresumB_M.reset();
          // iter_PresumB_M.row += (presumIter - kPresumComputeIterations) * iter_PresumB_M.row_increment();

          PresumVecTypeB b0; b0.clear();
          PresumVecTypeB b1; b1.clear();
          PresumVecTypeB b2; b2.clear();
          PresumVecTypeB b3; b3.clear();

          if (presumBComputeLoads.hasAccess(MmaStrassen::BPresums::B0))
            // PresumDetail::shared_load_128b(&b0, sharedPreSums.get(presumBComputeLoads.index(MmaStrassen::BPresums::B0), presum_read_stage, 0));
            b0 = PresumDetail::shared_load_128b<PresumVecTypeB>(sharedPreSums.get(presumBComputeLoads.index(MmaStrassen::BPresums::B0), presum_read_stage, 0));
          if (presumBComputeLoads.hasAccess(MmaStrassen::BPresums::B1))
            b1 = PresumDetail::shared_load_128b<PresumVecTypeB>(sharedPreSums.get(presumBComputeLoads.index(MmaStrassen::BPresums::B1), presum_read_stage, 0));
          if (presumBComputeLoads.hasAccess(MmaStrassen::BPresums::B2))
            b2 = PresumDetail::shared_load_128b<PresumVecTypeB>(sharedPreSums.get(presumBComputeLoads.index(MmaStrassen::BPresums::B2), presum_read_stage, 0));
          if (presumBComputeLoads.hasAccess(MmaStrassen::BPresums::B3))
            b3 = PresumDetail::shared_load_128b<PresumVecTypeB>(sharedPreSums.get(presumBComputeLoads.index(MmaStrassen::BPresums::B3), presum_read_stage, 0));

          PresumIOToComputeTypeB presum_io_to_compute_type;
          PresumComputeToIOTypeB presum_compute_to_io_type;

          auto b10  = presum_io_to_compute_type(b1) - presum_io_to_compute_type(b0);
          auto b31  = presum_io_to_compute_type(b3) - presum_io_to_compute_type(b1);
          auto s3   = b31 + presum_io_to_compute_type(b0);
          auto s3b2 = s3 - presum_io_to_compute_type(b2);

          using AllPresums = typename StrassenMiGroup::AllPresums;
            // if (StrassenMiGroup::hasM0() && StrassenMiGroup::Level == 2 &&
            //     iter_PresumB_M.tb_offset.column() == 0 && iter_PresumB_M.col == 0 && iter_PresumB_M.tb_offset.row() + iter_PresumB_M.row == 512)
            //     printf("1632 %p %d ; %d %d; %f %f %f %f\n", iter_PresumB_M.get(AllPresums::indexBPresum(MmaStrassen::BPresums::B31)),
            //     iter_PresumB_M.tb_offset.row() + iter_PresumB_M.row, iter_PresumB_M.stride, iter_PresumB_M.VectorLoadElems,
            //     float(b10[0]), float(b31[0]), float(s3[0]), float(s3b2[0]));
              // printf("1632 %p\n", iter_PresumB_M.get(AllPresums::indexBPresum(MmaStrassen::BPresums::B31)));
          if (AllPresums::doesComputeB(MmaStrassen::BPresums::B10)) {
            arch::global_store<PresumVecTypeB, sizeof(PresumVecTypeB)>(presum_compute_to_io_type(b10), iter_PresumB_M.get(AllPresums::indexBPresum(MmaStrassen::BPresums::B10)),
                                                                      iter_PresumB_M.validTB() && iter_PresumB_M.valid());
          }
          if (AllPresums::doesComputeB(MmaStrassen::BPresums::B31)) {
            arch::global_store<PresumVecTypeB, sizeof(PresumVecTypeB)>(presum_compute_to_io_type(b31), iter_PresumB_M.get(AllPresums::indexBPresum(MmaStrassen::BPresums::B31)),
                                                                      iter_PresumB_M.validTB() && iter_PresumB_M.valid());
          }
          if (AllPresums::doesComputeB(MmaStrassen::BPresums::S3)) {
            arch::global_store<PresumVecTypeB, sizeof(PresumVecTypeB)>(presum_compute_to_io_type(s3), iter_PresumB_M.get(AllPresums::indexBPresum(MmaStrassen::BPresums::S3)),
                                                                      iter_PresumB_M.validTB() && iter_PresumB_M.valid());
          }
          if (AllPresums::doesComputeB(MmaStrassen::BPresums::S3B2)) {
            arch::global_store<PresumVecTypeB, sizeof(PresumVecTypeB)>(presum_compute_to_io_type(s3b2), iter_PresumB_M.get(AllPresums::indexBPresum(MmaStrassen::BPresums::S3B2)),
                                                                      iter_PresumB_M.validTB() && iter_PresumB_M.valid());
          }

          iter_PresumB_M.inc();
        }
      }
    };

    // Start async loads for 1st k-tile onwards, no k-residue handling needed
    CUTLASS_PRAGMA_UNROLL
    for (int k_pipe = 0; k_pipe < DispatchPolicy::Stages-1; ++k_pipe) {
      if (k_tile_count <= 0) {
        clear(tApA);
        clear(tBpB);
      }
      copy_if(gmem_tiled_copy_A, tApA, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,k_pipe));  // CpAsync
      copy_if(gmem_tiled_copy_B, tBpB, tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,k_pipe));  // CpAsync
      presum_load(true);
      
      cp_async_fence();
      ++presumIter;
      ++k_tile_iter;
      --k_tile_count;
    }

    //
    // MMA Atom partitioning
    //

    presumIter = 0;

    // Tile MMA compute thread partitions and allocate accumulators
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCrA  = thr_mma.partition_fragment_A(sA(_,_,0));                    // (MMA,MMA_M,MMA_K)
    Tensor tCrB  = thr_mma.partition_fragment_B(sB(_,_,0));                    // (MMA,MMA_N,MMA_K)

    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(src_accum));                 // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(src_accum));                 // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                      // MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_A   = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A     = smem_tiled_copy_A.get_thread_slice(thread_idx);
    Tensor tCsA           = smem_thr_copy_A.partition_S(sA);                   // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);                    // (CPY,CPY_M,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));            // CPY_K

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(thread_idx);
    Tensor tCsB              = smem_thr_copy_B.partition_S(sB);                // (CPY,CPY_N,CPY_K,PIPE)
    Tensor tCrB_copy_view    = smem_thr_copy_B.retile_D(tCrB);                 // (CPY,CPY_N,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));            // CPY_K

    //
    // PIPELINED MAIN LOOP
    //

    // Current pipe index in smem to read from
    int smem_pipe_read  = 0;
    // Current pipe index in smem to write to
    int smem_pipe_write = DispatchPolicy::Stages-1;

    Tensor tCsA_p = tCsA(_,_,_,smem_pipe_read);
    Tensor tCsB_p = tCsB(_,_,_,smem_pipe_read);

    // Size of the register pipeline
    auto K_BLOCK_MAX = size<2>(tCrA);
    
    // PREFETCH register pipeline
    if (K_BLOCK_MAX > 1) {
      // Wait until our first prefetched tile is loaded in
      cp_async_wait<DispatchPolicy::Stages-2>();
      __syncthreads();

      // Prefetch the first rmem from the first k-tile
      copy(smem_tiled_copy_A, tCsA_p(_,_,Int<0>{}), tCrA_copy_view(_,_,Int<0>{}));
      copy(smem_tiled_copy_B, tCsB_p(_,_,Int<0>{}), tCrB_copy_view(_,_,Int<0>{}));
    }

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > -(DispatchPolicy::Stages-1); --k_tile_count)
    {
      // Pipeline the outer products with a static for loop.
      //
      // Note, the for_each() function is required here to ensure `k_block` is of type Int<N>.
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block)
      {
        if (k_block == K_BLOCK_MAX - 1)
        {
          // Slice the smem_pipe_read smem
          tCsA_p = tCsA(_,_,_,smem_pipe_read);
          tCsB_p = tCsB(_,_,_,smem_pipe_read);

          // Commit the smem for smem_pipe_read
          cp_async_wait<DispatchPolicy::Stages-2>();
          __syncthreads();
          presum_compute_and_store();
          ++presumIter;
        }

        // Load A, B shmem->regs for k_block+1
        auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;  // static
        copy(smem_tiled_copy_A, tCsA_p(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
        copy(smem_tiled_copy_B, tCsB_p(_,_,k_block_next), tCrB_copy_view(_,_,k_block_next));

        // Copy gmem to smem before computing gemm on each k-pipe
        if (k_block == 0)
        {
          // Set all predicates to false if we are going to overshoot bounds
          if (k_tile_count <= 0) {
            clear(tApA);
            clear(tBpB);
          }
          copy_if(gmem_tiled_copy_A, tApA, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,smem_pipe_write));
          copy_if(gmem_tiled_copy_B, tBpB, tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,smem_pipe_write));
          presum_load(false);
          cp_async_fence();

          ++k_tile_iter;

          // Advance the pipe -- Doing it here accounts for K_BLOCK_MAX = 1 (no rmem pipe)
          smem_pipe_write = smem_pipe_read;
          ++smem_pipe_read;
          smem_pipe_read = (smem_pipe_read == DispatchPolicy::Stages) ? 0 : smem_pipe_read;
        }

        // Transform before compute
        cute::transform(tCrA(_,_,k_block), TransformA{});
        cute::transform(tCrB(_,_,k_block), TransformB{});

        // Thread-level register gemm for k_block
        cute::gemm(tiled_mma, accum, tCrA(_,_,k_block), tCrB(_,_,k_block), src_accum);
      });

    }

    cp_async_wait<0>();
    __syncthreads();
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
