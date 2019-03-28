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
use crate::util::*;

use std::marker::PhantomData;

#[derive(Clone, Copy)]
pub struct Tiling {
  frame_width: usize,
  frame_height: usize,
  frame_width_in_sb: usize,
  frame_height_in_sb: usize,
  tile_width_in_sb: usize,
  tile_height_in_sb: usize,
  cols: usize,
  rows: usize,
  sb_size_log2: usize,
}

impl Tiling {
  pub fn from_tile_size<T: Pixel>(
    fi: &FrameInvariants<T>,
    tile_width_in_sb: usize,
    tile_height_in_sb: usize,
  ) -> Self {
    let sb_size_log2 = fi.sb_size_log2();
    let frame_width = fi.width.align_power_of_two(3);
    let frame_height = fi.height.align_power_of_two(3);
    let frame_width_in_sb = frame_width.align_power_of_two_and_shift(sb_size_log2);
    let frame_height_in_sb = frame_height.align_power_of_two_and_shift(sb_size_log2);
    let cols = (frame_width_in_sb + tile_width_in_sb - 1) / tile_width_in_sb;
    let rows = (frame_height_in_sb + tile_height_in_sb - 1) / tile_height_in_sb;
    Self {
      frame_width,
      frame_height,
      frame_width_in_sb,
      frame_height_in_sb,
      tile_width_in_sb,
      tile_height_in_sb,
      cols,
      rows,
      sb_size_log2,
    }
  }

  #[inline(always)]
  pub fn tile_count(&self) -> usize {
    self.cols * self.rows
  }

  pub fn tile<'a, 'b, T: Pixel>(
    &self,
    fs: &'a mut FrameState<T>,
    fb: &'b mut FrameBlocks,
  ) -> TileContextIterMut<'a, 'b, T> {
    TileContextIterMut {
      tiling: *self,
      fs,
      fb,
      next: 0,
      phantom: PhantomData,
    }
  }
}

pub struct TileContextMut<'a, 'b, T: Pixel> {
  fs: TileStateMut<'a, T>,
  fb: BlocksRegionMut<'b>,
}

pub struct TileContextIterMut<'a, 'b, T: Pixel> {
  tiling: Tiling,
  fs: *mut FrameState<T>,
  fb: *mut FrameBlocks,
  next: usize,
  phantom: PhantomData<(&'a mut FrameState<T>, &'b mut FrameBlocks)>,
}

impl<'a, 'b, T: Pixel> Iterator for TileContextIterMut<'a, 'b, T> {
  type Item = TileContextMut<'a, 'b, T>;

  fn next(&mut self) -> Option<Self::Item> {
    None
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    let remaining = self.tiling.cols * self.tiling.rows - self.next;
    (remaining, Some(remaining))
  }
}

impl<T: Pixel> ExactSizeIterator for TileContextIterMut<'_, '_, T> {}
