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
use crate::plane::*;
use crate::quantize::*;
use crate::rdo::*;
use crate::util::*;

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
  pub rdo: RDOTracker,
}

impl<'a, T: Pixel> TileStateMut<'a, T> {
  pub fn new(
    fs: &'a mut FrameState<T>,
    luma_rect: Rect,
  ) -> Self {
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
      rdo: RDOTracker::new(),
    }
  }
}
