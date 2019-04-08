// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::context::*;

use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use std::slice;

/// Tiled view of FrameBlocks
#[derive(Debug)]
pub struct TileBlocks<'a> {
  data: *const Block,
  x: usize,
  y: usize,
  cols: usize,
  rows: usize,
  stride: usize, // number of cols in the underlying FrameBlocks
  phantom: PhantomData<&'a Block>,
}

/// Mutable tiled view of FrameBlocks
#[derive(Debug)]
pub struct TileBlocksMut<'a> {
  data: *mut Block,
  // cannot make these fields public, because they must not be written to,
  // otherwise we could break borrowing rules in safe code
  x: usize,
  y: usize,
  cols: usize,
  rows: usize,
  stride: usize, // number of cols in the underlying FrameBlocks
  phantom: PhantomData<&'a mut Block>,
}

// common impl for TileBlocks and TileBlocksMut
macro_rules! tile_blocks_common {
  ($name: ident, $fb_ref_type: ty, $index: ident) => {
    impl<'a> $name<'a> {

      #[inline(always)]
      pub fn new(
        frame_blocks: $fb_ref_type,
        x: usize,
        y: usize,
        cols: usize,
        rows: usize,
      ) -> Self {
        Self {
          data: frame_blocks.$index(y).$index(x), // &(mut) frame_blocks[y][x]
          x,
          y,
          cols,
          rows,
          stride: frame_blocks.cols,
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
      type Output = [Block];
      #[inline(always)]
      fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.rows);
        unsafe {
          let ptr = self.data.add(index * self.stride);
          slice::from_raw_parts(ptr, self.cols)
        }
      }
    }

    // for convenience, also index by BlockOffset
    impl Index<BlockOffset> for $name<'_> {
      type Output = Block;
      #[inline(always)]
      fn index(&self, bo: BlockOffset) -> &Self::Output {
        &self[bo.y][bo.x]
      }
    }
  }
}

tile_blocks_common!(TileBlocks, &'a FrameBlocks, index);
tile_blocks_common!(TileBlocksMut, &'a mut FrameBlocks, index_mut);

impl TileBlocksMut<'_> {
  #[inline(always)]
  pub fn as_const(&self) -> TileBlocks<'_> {
    TileBlocks {
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

impl IndexMut<usize> for TileBlocksMut<'_> {
  #[inline(always)]
  fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    assert!(index < self.rows);
    unsafe {
      let ptr = self.data.add(index * self.stride);
      slice::from_raw_parts_mut(ptr, self.cols)
    }
  }
}

impl IndexMut<BlockOffset> for TileBlocksMut<'_> {
  #[inline(always)]
  fn index_mut(&mut self, bo: BlockOffset) -> &mut Self::Output {
    &mut self[bo.y][bo.x]
  }
}
