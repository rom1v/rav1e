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
use std::slice;

#[derive(Debug, Clone)]
pub struct TileRestorationPlane<'a> {
  pub rp: &'a RestorationPlane,
  pub wiener_ref: [[i8; 3]; 2],
  pub sgrproj_ref: [i8; 2],
}

impl<'a> TileRestorationPlane<'a> {
  pub fn new(rp: &'a RestorationPlane) -> Self {
    Self { rp, wiener_ref: [WIENER_TAPS_MID; 2], sgrproj_ref: SGRPROJ_XQD_MID }
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
  pub fn new(rs: &'a RestorationState) -> Self {
    Self {
      planes: [
        TileRestorationPlane::new(&rs.planes[0]),
        TileRestorationPlane::new(&rs.planes[1]),
        TileRestorationPlane::new(&rs.planes[2]),
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
  pub input: &'a Frame<T>, // the whole frame
  pub input_tile: Tile<'a, T>, // the current tile
  pub input_hres: &'a Plane<T>,
  pub input_qres: &'a Plane<T>,
  pub deblock: &'a DeblockState,
  pub luma_rect: Rect,
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
    luma_rect: Rect,
  ) -> Self {
    assert!(luma_rect.x >= 0);
    assert!(luma_rect.y >= 0);
    assert!(luma_rect.width & (MI_SIZE - 1) == 0, "luma_rect must be a multiple of MI_SIZE");
    assert!(luma_rect.height & (MI_SIZE - 1) == 0, "luma_rect must be a multiple of MI_SIZE");
    Self {
      input: &fs.input,
      input_tile: Tile::new(&fs.input, luma_rect),
      input_hres: &fs.input_hres,
      input_qres: &fs.input_qres,
      deblock: &fs.deblock,
      luma_rect,
      rec: TileMut::new(&mut fs.rec, luma_rect),
      qc: Default::default(),
      cdfs: CDFContext::new(0),
      segmentation: &fs.segmentation,
      restoration: TileRestorationState::new(&fs.restoration),
      mvs: fs.frame_mvs.iter_mut().map(|fmvs| {
        TileMotionVectorsMut::new(
          fmvs,
          luma_rect.x as usize >> MI_SIZE_LOG2,
          luma_rect.y as usize >> MI_SIZE_LOG2,
          luma_rect.width >> MI_SIZE_LOG2,
          luma_rect.height >> MI_SIZE_LOG2,
        )
      }).collect(),
      rdo: RDOTracker::new(),
    }
  }
}

pub struct TileStateIterMut<'a, T: Pixel> {
  fs: *mut FrameState<T>,
  tile_width: usize,
  tile_height: usize,
  tile_cols: usize,
  tile_rows: usize,
  next: usize, // index of the next tile to yield
  phantom: PhantomData<&'a mut FrameState<T>>,
}

impl<'a, T: Pixel> TileStateIterMut<'a, T> {
  pub fn from_frame_state(
    fs: &'a mut FrameState<T>,
    tile_width: usize,
    tile_height: usize,
  ) -> Self {
    let PlaneConfig {
      width: frame_width,
      height: frame_height,
      ..
    } = fs.rec.planes[0].cfg;

    // XXX assert not sufficient if use_128x128_superblock
    assert!((tile_width % MAX_SB_SIZE) == 0, "tiles must be multiple of superblocks");
    assert!((tile_height % MAX_SB_SIZE) == 0, "tiles must be multiple of superblocks");

    let tile_cols = (frame_width + tile_width - 1) / tile_width;
    let tile_rows = (frame_height + tile_height - 1) / tile_height;

    Self {
      fs,
      tile_width,
      tile_height,
      tile_cols,
      tile_rows,
      next: 0,
      phantom: PhantomData,
    }
  }
}

impl<'a, T: Pixel> Iterator for TileStateIterMut<'a, T> {
  type Item = TileStateMut<'a, T>;

  fn next(&mut self) -> Option<Self::Item> {
    if self.next < self.tile_rows * self.tile_cols {
      let fs = unsafe { &mut *self.fs };
      let PlaneConfig {
        width: frame_width,
        height: frame_height,
        ..
      } = fs.rec.planes[0].cfg;

      let x = (self.next % self.tile_cols) * self.tile_width;
      let y = (self.next / self.tile_cols) * self.tile_height;
      let width = self.tile_width.min(frame_width - x);
      let height = self.tile_height.min(frame_height - y);
      let luma_rect = Rect { x: x as isize, y: y as isize, width, height };

      self.next += 1;

      let ts = TileStateMut::new(fs, luma_rect);
      Some(ts)
    } else {
      None
    }
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    let remaining = self.tile_cols * self.tile_rows - self.next;
    (remaining, Some(remaining))
  }
}

impl<'a, T: Pixel> ExactSizeIterator for TileStateIterMut<'a, T> {}
