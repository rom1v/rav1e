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

#[derive(Debug, Clone, Copy)]
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

  pub fn tile_iter_mut<'a, 'b, T: Pixel>(
    &self,
    fs: &'a mut FrameState<T>,
    fb: &'b mut FrameBlocks,
  ) -> TileContextIterMut<'a, 'b, T> {
    TileContextIterMut {
      ti: *self,
      fs,
      fb,
      next: 0,
      phantom: PhantomData,
    }
  }
}

pub struct TileContextMut<'a, 'b, T: Pixel> {
  pub ts: TileStateMut<'a, T>,
  pub tb: BlocksRegionMut<'b>,
}

pub struct TileContextIterMut<'a, 'b, T: Pixel> {
  ti: TilingInfo,
  fs: *mut FrameState<T>,
  fb: *mut FrameBlocks,
  next: usize,
  phantom: PhantomData<(&'a mut FrameState<T>, &'b mut FrameBlocks)>,
}

impl<'a, 'b, T: Pixel> Iterator for TileContextIterMut<'a, 'b, T> {
  type Item = TileContextMut<'a, 'b, T>;

  fn next(&mut self) -> Option<Self::Item> {
    if self.next < self.ti.rows * self.ti.cols {
      let tile_col = self.next % self.ti.cols;
      let tile_row = self.next / self.ti.cols;
      let ctx = TileContextMut {
        ts: {
          let fs = unsafe { &mut *self.fs };
          let tile_width = self.ti.tile_width_sb << self.ti.sb_size_log2;
          let tile_height = self.ti.tile_height_sb << self.ti.sb_size_log2;
          let x = tile_col * tile_width;
          let y = tile_row * tile_height;
          let width = tile_width.min(self.ti.frame_width - x);
          let height = tile_height.min(self.ti.frame_height - y);
          let luma_rect = Rect { x: x as isize, y: y as isize, width, height };
          TileStateMut::new(fs, luma_rect)
        },
        tb: {
          let fb = unsafe { &mut *self.fb };
          let tile_width_mi = self.ti.tile_width_sb << (self.ti.sb_size_log2 - MI_SIZE_LOG2);
          let tile_height_mi = self.ti.tile_height_sb << (self.ti.sb_size_log2 - MI_SIZE_LOG2);
          let x = tile_col * tile_width_mi;
          let y = tile_row * tile_height_mi;
          let cols = tile_width_mi.min(fb.cols - x);
          let rows = tile_height_mi.min(fb.rows - y);
          BlocksRegionMut::new(fb, x, y, cols, rows)
        }
      };
      self.next += 1;
      Some(ctx)
    } else {
      None
    }
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    let remaining = self.ti.cols * self.ti.rows - self.next;
    (remaining, Some(remaining))
  }
}

impl<T: Pixel> ExactSizeIterator for TileContextIterMut<'_, '_, T> {}
