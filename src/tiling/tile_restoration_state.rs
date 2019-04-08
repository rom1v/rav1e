// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::context::*;
use crate::lrf::*;

use std::sync::Mutex;

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
  #[inline(always)]
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

