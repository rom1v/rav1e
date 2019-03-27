// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::context::*;
use crate::partition::*;

use std::cmp;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use std::slice;

/// Tiled view of FrameBlocks
#[derive(Debug)]
pub struct BlocksRegion<'a> {
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
pub struct BlocksRegionMut<'a> {
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

// common impl for BlocksRegion and BlocksRegionMut
macro_rules! tile_blocks_common {
  ($name: ident, $fb_ref_type: ty, $index: ident) => {
    impl<'a> $name<'a> {

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
      #[inline]
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
      #[inline]
      fn index(&self, bo: BlockOffset) -> &Self::Output {
        &self[bo.y][bo.x]
      }
    }
  }
}

tile_blocks_common!(BlocksRegion, &'a FrameBlocks, index);
tile_blocks_common!(BlocksRegionMut, &'a mut FrameBlocks, index_mut);

impl BlocksRegionMut<'_> {
  #[inline]
  pub fn as_const(&self) -> BlocksRegion<'_> {
    BlocksRegion {
      data: self.data,
      x: self.x,
      y: self.y,
      cols: self.cols,
      rows: self.rows,
      stride: self.stride,
      phantom: PhantomData,
    }
  }

  #[inline]
  pub fn above_of(&self, bo: BlockOffset) -> &Block {
    &self[bo.y - 1][bo.x]
  }

  #[inline]
  pub fn left_of(&self, bo: BlockOffset) -> &Block {
    &self[bo.y][bo.x - 1]
  }

  #[inline]
  pub fn above_left_of(&self, bo: BlockOffset) -> &Block {
    &self[bo.y - 1][bo.x - 1]
  }

  pub fn for_each<F>(&mut self, bo: BlockOffset, bsize: BlockSize, f: F)
  where
    F: Fn(&mut Block) -> ()
  {
    let bw = bsize.width_mi();
    let bh = bsize.height_mi();
    for y in 0..bh {
      for x in 0..bw {
        f(&mut self[bo.y + y as usize][bo.x + x as usize]);
      }
    }
  }

  pub fn set_mode(
    &mut self, bo: BlockOffset, bsize: BlockSize, mode: PredictionMode
  ) {
    self.for_each(bo, bsize, |block| block.mode = mode);
  }

  pub fn set_block_size(&mut self, bo: BlockOffset, bsize: BlockSize) {
    let n4_w = bsize.width_mi();
    let n4_h = bsize.height_mi();
    self.for_each(bo, bsize, |block| { block.bsize = bsize; block.n4_w = n4_w; block.n4_h = n4_h } );
  }

  pub fn set_tx_size(&mut self, bo: BlockOffset, bsize: BlockSize, tx_size: TxSize) {
    self.for_each(bo, bsize, |block| { block.txsize = tx_size } );
  }

  pub fn set_skip(&mut self, bo: BlockOffset, bsize: BlockSize, skip: bool) {
    self.for_each(bo, bsize, |block| block.skip = skip);
  }

  pub fn set_segmentation_idx(&mut self, bo: BlockOffset, bsize: BlockSize, idx: u8) {
    self.for_each(bo, bsize, |block| block.segmentation_idx = idx);
  }

  pub fn set_ref_frames(&mut self, bo: BlockOffset, bsize: BlockSize, r: [usize; 2]) {
    self.for_each(bo, bsize, |block| block.ref_frames = r);
  }

  pub fn set_motion_vectors(&mut self, bo: BlockOffset, bsize: BlockSize, mvs: [MotionVector; 2]) {
    self.for_each(bo, bsize, |block| block.mv = mvs);
  }

  pub fn set_cdef(&mut self, sbo: SuperBlockOffset, cdef_index: u8) {
    let bo = sbo.block_offset(0, 0);
    // Checkme: Is 16 still the right block unit for 128x128 superblocks?
    let bw = cmp::min(bo.x + MAX_MIB_SIZE, self.cols);
    let bh = cmp::min(bo.y + MAX_MIB_SIZE, self.rows);
    for y in bo.y..bh {
      for x in bo.x..bw {
        self[y as usize][x as usize].cdef_index = cdef_index;
      }
    }
  }

  pub fn get_cdef(&mut self, sbo: SuperBlockOffset) -> u8 {
    let bo = sbo.block_offset(0, 0);
    self[bo.y][bo.x].cdef_index
  }
}

impl IndexMut<usize> for BlocksRegionMut<'_> {
  #[inline]
  fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    assert!(index < self.rows);
    unsafe {
      let ptr = self.data.add(index * self.stride);
      slice::from_raw_parts_mut(ptr, self.cols)
    }
  }
}

impl IndexMut<BlockOffset> for BlocksRegionMut<'_> {
  #[inline]
  fn index_mut(&mut self, bo: BlockOffset) -> &mut Self::Output {
    &mut self[bo.y][bo.x]
  }
}
