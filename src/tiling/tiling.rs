// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::encoder::*;
use crate::util::*;

#[derive(Clone, Copy)]
pub struct TilingInfo {
  pub frame_width: usize,
  pub frame_height: usize,
  pub tile_width_sb: usize,
  pub tile_height_sb: usize,
  pub cols: usize, // number of columns of tiles within the whole frame
  pub rows: usize, // number of rows of tiles within the whole frame
  pub sb_size_log2: usize,
}

impl TilingInfo {
  pub fn from_tile_size_sb<T: Pixel>(
    fi: &FrameInvariants<T>,
    tile_width_sb: usize,
    tile_height_sb: usize,
  ) -> Self {
    let sb_size_log2 = fi.sb_size_log2();
    let frame_width = fi.width.align_power_of_two(3);
    let frame_height = fi.height.align_power_of_two(3);
    let frame_width_sb = frame_width.align_power_of_two_and_shift(sb_size_log2);
    let frame_height_sb = frame_height.align_power_of_two_and_shift(sb_size_log2);
    let cols = (frame_width_sb + tile_width_sb - 1) / tile_width_sb;
    let rows = (frame_height_sb + tile_height_sb - 1) / tile_height_sb;
    Self {
      frame_width,
      frame_height,
      tile_width_sb,
      tile_height_sb,
      cols,
      rows,
      sb_size_log2,
    }
  }

  #[inline(always)]
  pub fn tile_count(&self) -> usize {
    self.cols * self.rows
  }
}
