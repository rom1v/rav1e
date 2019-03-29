// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use super::*;

use crate::context::*;
use crate::encoder::*;
use crate::lrf::*;
use crate::me::*;
use crate::partition::*;
use crate::plane::*;
use crate::quantize::*;
use crate::rdo::*;
use crate::util::*;

use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use std::sync::Mutex;
use std::slice;

#[derive(Debug, Clone)]
pub struct TileRestorationPlane<'a> {
  pub sbo: SuperBlockOffset,
  pub rp: &'a RestorationPlane,
  pub wiener_ref: [[i8; 3]; 2],
  pub sgrproj_ref: [i8; 2],
}

impl<'a> TileRestorationPlane<'a> {
  pub fn new(sbo: SuperBlockOffset, rp: &'a RestorationPlane) -> Self {
    Self { sbo, rp, wiener_ref: [WIENER_TAPS_MID; 2], sgrproj_ref: SGRPROJ_XQD_MID }
  }

  pub fn restoration_unit(&self, tile_sbo: SuperBlockOffset) -> &Mutex<RestorationUnit> {
    let frame_sbo = SuperBlockOffset {
      x: self.sbo.x + tile_sbo.x,
      y: self.sbo.y + tile_sbo.y,
    };
    self.rp.restoration_unit(frame_sbo)
  }
}

/// Tiled version of RestorationState
///
/// Contrary to other views, TileRestorationState is not exposed as mutable
/// because it is (possibly) shared between several tiles (due to restoration
/// unit stretching).
///
/// It contains, for each plane, tile-specific data, and a reference to the
/// frame-wise RestorationPlane, that will provide interior mutability to access
/// restoration units from several tiles.
#[derive(Debug, Clone)]
pub struct TileRestorationState<'a> {
  pub planes: [TileRestorationPlane<'a>; PLANES],
}

impl<'a> TileRestorationState<'a> {
  pub fn new(sbo: SuperBlockOffset, rs: &'a RestorationState) -> Self {
    Self {
      planes: [
        TileRestorationPlane::new(sbo, &rs.planes[0]),
        TileRestorationPlane::new(sbo, &rs.planes[1]),
        TileRestorationPlane::new(sbo, &rs.planes[2]),
      ],
    }
  }
}

/// Tiled view of FrameMotionVectors
#[derive(Debug)]
pub struct TileMotionVectors<'a> {
  data: *const MotionVector,
  // expressed in mi blocks
  x: usize,
  y: usize,
  cols: usize,
  rows: usize,
  stride: usize, // number of cols in the underlying FrameMotionVectors
  phantom: PhantomData<&'a MotionVector>,
}

/// Mutable tiled view of FrameMotionVectors
#[derive(Debug)]
pub struct TileMotionVectorsMut<'a> {
  data: *mut MotionVector,
  // expressed in mi blocks
  // cannot make these fields public, because they must not be written to,
  // otherwise we could break borrowing rules in safe code
  x: usize,
  y: usize,
  cols: usize,
  rows: usize,
  stride: usize, // number of cols in the underlying FrameMotionVectors
  phantom: PhantomData<&'a mut MotionVector>,
}

// common impl for TileMotionVectors and TileMotionVectorsMut
macro_rules! tile_motion_vectors_common {
  // $name: TileMotionVectors or TileMotionVectorsMut
  // $fmvs_ref_type: &'a FrameMotionVectors or &'a mut FrameMotionVectors
  // $index: index or index_mut
  ($name: ident, $fmv_ref_type: ty, $index: ident) => {
    impl<'a> $name<'a> {

      pub fn new(
        frame_mvs: $fmv_ref_type,
        x: usize,
        y: usize,
        cols: usize,
        rows: usize,
      ) -> Self {
        Self {
          data: frame_mvs.$index(y).$index(x), // &(mut) frame_mvs[y][x],
          x,
          y,
          cols,
          rows,
          stride: frame_mvs.cols,
          phantom: PhantomData,
        }
      }

      #[inline(always)]
      pub fn x(&self) -> usize {
        self.x
      }

      #[inline(always)]
      pub fn y(&self) -> usize {
        self.y
      }

      #[inline(always)]
      pub fn cols(&self) -> usize {
        self.cols
      }

      #[inline(always)]
      pub fn rows(&self) -> usize {
        self.rows
      }
    }

    unsafe impl Send for $name<'_> {}
    unsafe impl Sync for $name<'_> {}

    impl Index<usize> for $name<'_> {
      type Output = [MotionVector];
      #[inline]
      fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.rows);
        unsafe {
          let ptr = self.data.add(index * self.stride);
          slice::from_raw_parts(ptr, self.cols)
        }
      }
    }
  }
}

tile_motion_vectors_common!(TileMotionVectors, &'a FrameMotionVectors, index);
tile_motion_vectors_common!(TileMotionVectorsMut, &'a mut FrameMotionVectors, index_mut);

impl TileMotionVectorsMut<'_> {
  #[inline]
  pub fn as_const(&self) -> TileMotionVectors<'_> {
    TileMotionVectors {
      data: self.data,
      x: self.x,
      y: self.y,
      cols: self.cols,
      rows: self.rows,
      stride: self.stride,
      phantom: PhantomData,
    }
  }
}

impl IndexMut<usize> for TileMotionVectorsMut<'_> {
  #[inline]
  fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    assert!(index < self.rows);
    unsafe {
      let ptr = self.data.add(index * self.stride);
      slice::from_raw_parts_mut(ptr, self.cols)
    }
  }
}

/// Tiled view of FrameState
///
/// This is the top-level tiling structure, providing tiling views of its
/// data when necessary.
///
/// It is intended to be created from a tile-interator on FrameState.
///
/// Contrary to PlaneRegionMut and TileMut, there is no const version:
///  - in practice, we don't need it;
///  - it would not be free to convert from TileStateMut to TileState, since
///    several of its fields will also need the instantiation of
///    const-equivalent structures.
///
/// # TileState fields
///
/// The way the FrameState fields are mapped depend on how they are accessed
/// tile-wise and frame-wise.
///
/// Some fields (like "qc") are only used during tile-encoding, so they are only
/// stored in TileState.
///
/// Some other fields (like "input" or "segmentation") are not written
/// tile-wise, so they just reference the matching field in FrameState.
///
/// Some others (like "rec") are written tile-wise, but must be accessible
/// frame-wise once the tile views vanish (e.g. for deblocking).
///
/// The "restoration" field is more complicated: some of its data
/// (restoration units) are written tile-wise, but shared between several
/// tiles. Therefore, they are stored in FrameState with interior mutability
/// (protected by a mutex), and referenced from TileState.
/// See <https://github.com/xiph/rav1e/issues/631#issuecomment-454419152>.
///
/// This is still work-in-progress. Some fields are not managed correctly
/// between tile-wise and frame-wise accesses.
#[derive(Debug)]
pub struct TileStateMut<'a, T: Pixel> {
  pub sbo: SuperBlockOffset,
  pub sb_size_log2: usize,
  pub width: usize,
  pub height: usize,
  pub w_in_b: usize,
  pub h_in_b: usize,
  pub input: &'a Frame<T>, // the whole frame
  pub input_tile: Tile<'a, T>, // the current tile
  pub input_hres: &'a Plane<T>,
  pub input_qres: &'a Plane<T>,
  pub deblock: &'a DeblockState,
  pub rec: TileMut<'a, T>,
  pub qc: QuantizationContext,
  pub cdfs: CDFContext,
  pub segmentation: &'a SegmentationState,
  pub restoration: TileRestorationState<'a>,
  pub mvs: Vec<TileMotionVectorsMut<'a>>,
  pub rdo: RDOTracker,
}

impl<'a, T: Pixel> TileStateMut<'a, T> {
  pub fn new(
    fs: &'a mut FrameState<T>,
    sbo: SuperBlockOffset,
    sb_size_log2: usize,
    width: usize,
    height: usize,
  ) -> Self {
    debug_assert!(width % MI_SIZE == 0, "Tile width must be a multiple of MI_SIZE");
    debug_assert!(height % MI_SIZE == 0, "Tile width must be a multiple of MI_SIZE");
    let luma_rect = Rect {
      x: (sbo.x << sb_size_log2) as isize,
      y: (sbo.y << sb_size_log2) as isize,
      width,
      height,
    };
    Self {
      sbo,
      sb_size_log2,
      width,
      height,
      w_in_b: width >> MI_SIZE_LOG2,
      h_in_b: height >> MI_SIZE_LOG2,
      input: &fs.input,
      input_tile: Tile::new(&fs.input, luma_rect),
      input_hres: &fs.input_hres,
      input_qres: &fs.input_qres,
      deblock: &fs.deblock,
      rec: TileMut::new(&mut fs.rec, luma_rect),
      qc: Default::default(),
      cdfs: CDFContext::new(0),
      segmentation: &fs.segmentation,
      restoration: TileRestorationState::new(sbo, &fs.restoration),
      mvs: fs.frame_mvs.iter_mut().map(|fmvs| {
        TileMotionVectorsMut::new(
          fmvs,
          sbo.x << sb_size_log2 - MI_SIZE_LOG2,
          sbo.y << sb_size_log2 - MI_SIZE_LOG2,
          width >> MI_SIZE_LOG2,
          height >> MI_SIZE_LOG2,
        )
      }).collect(),
      rdo: RDOTracker::new(),
    }
  }

  #[inline]
  pub fn to_frame_block_offset(&self, tile_bo: BlockOffset) -> BlockOffset {
    let bx = self.sbo.x << self.sb_size_log2 - MI_SIZE_LOG2;
    let by = self.sbo.y << self.sb_size_log2 - MI_SIZE_LOG2;
    BlockOffset {
      x: bx + tile_bo.x,
      y: by + tile_bo.y,
    }
  }

  #[inline]
  pub fn to_frame_super_block_offset(&self, tile_sbo: SuperBlockOffset) -> SuperBlockOffset {
    SuperBlockOffset {
      x: self.sbo.x + tile_sbo.x,
      y: self.sbo.y + tile_sbo.y,
    }
  }
}
