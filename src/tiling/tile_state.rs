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
use crate::plane::*;
use crate::quantize::*;
use crate::rdo::*;
use crate::util::*;

use std::marker::PhantomData;

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
