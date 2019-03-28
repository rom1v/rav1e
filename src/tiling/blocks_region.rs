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
use crate::util::*;

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

  pub fn set_ref_frames(&mut self, bo: BlockOffset, bsize: BlockSize, r: [RefType; 2]) {
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

pub struct TileBlocksIterMut<'a> {
  fb: *mut FrameBlocks,
  tile_width: usize, // in blocks
  tile_height: usize, // in blocks
  cols: usize, // number of columns of tiles within the frame
  rows: usize, // number of rows of tiles withing the frame
  next: usize, // index of the next tile to yield
  phantom: PhantomData<&'a mut FrameBlocks>,
}

impl<'a> TileBlocksIterMut<'a> {

  /// Return an iterator over the blocks regions split by tiles
  ///
  /// # Arguments
  ///
  /// * `fb` - The whole FrameBlocks instance
  /// * `sb_size_log2` - The log2 size of a superblock (typically 6 or 7)
  /// * `tile_width_in_sb` - The width of a tile, in number of superblocks
  /// * `tile_height_in_sb` - The width of a tile, in number of superblocks
  pub fn from_frame_blocks(
    fb: &'a mut FrameBlocks,
    sb_size_log2: usize,
    tile_width_in_sb: usize,
    tile_height_in_sb: usize,
  ) -> Self {
    let frame_width_in_sb = fb.cols.align_power_of_two_and_shift(sb_size_log2 - MI_SIZE_LOG2);
    let frame_height_in_sb = fb.rows.align_power_of_two_and_shift(sb_size_log2 - MI_SIZE_LOG2);

    Self {
      fb,
      tile_width: tile_width_in_sb << (sb_size_log2 - MI_SIZE_LOG2),
      tile_height: tile_height_in_sb << (sb_size_log2 - MI_SIZE_LOG2),
      cols: (frame_width_in_sb + tile_width_in_sb - 1) / tile_width_in_sb,
      rows: (frame_height_in_sb + tile_height_in_sb - 1) / tile_height_in_sb,
      next: 0,
      phantom: PhantomData,
    }
  }
}

impl<'a> Iterator for TileBlocksIterMut<'a> {
  type Item = BlocksRegionMut<'a>;

  fn next(&mut self) -> Option<Self::Item> {
    if self.next < self.rows * self.cols {
      let fb = unsafe { &mut *self.fb };

      let x = (self.next % self.cols) * self.tile_width;
      let y = (self.next / self.cols) * self.tile_height;
      let cols = self.tile_width.min(fb.cols - x);
      let rows = self.tile_height.min(fb.rows - y);

      self.next += 1;

      let tb = BlocksRegionMut::new(fb, x, y, cols, rows);
      Some(tb)
    } else {
      None
    }
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    let remaining = self.cols * self.rows - self.next;
    (remaining, Some(remaining))
  }
}

impl ExactSizeIterator for TileBlocksIterMut<'_> {}

#[cfg(test)]
pub mod test {
  use super::*;

  #[test]
  fn test_tile_blocks_iter_len() {
    // frame size 160x144, 40x36 in 4x4-blocks
    let mut fb = FrameBlocks::new(40, 36);

    {
      let mut iter = fb.tile_blocks_iter_mut(6, 2, 2);
      assert_eq!(4, iter.len());
      assert!(iter.next().is_some());
      assert_eq!(3, iter.len());
      assert!(iter.next().is_some());
      assert_eq!(2, iter.len());
      assert!(iter.next().is_some());
      assert_eq!(1, iter.len());
      assert!(iter.next().is_some());
      assert_eq!(0, iter.len());
      assert!(iter.next().is_none());
    }

    {
      let mut iter = fb.tile_blocks_iter_mut(6, 1, 1);
      assert_eq!(9, iter.len());
      assert!(iter.next().is_some());
      assert_eq!(8, iter.len());
      assert!(iter.next().is_some());
      assert_eq!(7, iter.len());
      assert!(iter.next().is_some());
      assert_eq!(6, iter.len());
      assert!(iter.next().is_some());
      assert_eq!(5, iter.len());
      assert!(iter.next().is_some());
      assert_eq!(4, iter.len());
      assert!(iter.next().is_some());
      assert_eq!(3, iter.len());
      assert!(iter.next().is_some());
      assert_eq!(2, iter.len());
      assert!(iter.next().is_some());
      assert_eq!(1, iter.len());
      assert!(iter.next().is_some());
      assert_eq!(0, iter.len());
      assert!(iter.next().is_none());
    }
  }

  #[inline]
  fn rect(region: &BlocksRegionMut<'_>) -> (usize, usize, usize, usize) {
    (region.x, region.y, region.cols, region.rows)
  }

  #[test]
  fn test_tile_blocks_area() {
    let mut fb = FrameBlocks::new(40, 36);

    let tbs = fb.tile_blocks_iter_mut(6, 1, 1).collect::<Vec<_>>();
    // the FrameBlocks must be split into 9 BlocksRegion:
    //
    //   16x16 16x16  8x16
    //   16x16 16x16  8x16
    //   16x 4 16x4   8x 4

    assert_eq!(9, tbs.len());

    assert_eq!((0, 0, 16, 16), rect(&tbs[0]));
    assert_eq!((16, 0, 16, 16), rect(&tbs[1]));
    assert_eq!((32, 0, 8, 16), rect(&tbs[2]));

    assert_eq!((0, 16, 16, 16), rect(&tbs[3]));
    assert_eq!((16, 16, 16, 16), rect(&tbs[4]));
    assert_eq!((32, 16, 8, 16), rect(&tbs[5]));

    assert_eq!((0, 32, 16, 4), rect(&tbs[6]));
    assert_eq!((16, 32, 16, 4), rect(&tbs[7]));
    assert_eq!((32, 32, 8, 4), rect(&tbs[8]));
  }

  #[test]
  fn test_tile_blocks_write() {
    let mut fb = FrameBlocks::new(40, 36);

    {
      let mut tbs = fb.tile_blocks_iter_mut(6, 1, 1).collect::<Vec<_>>();

      {
        // top-left tile
        let tb = &mut tbs[0];
        // block (4, 3)
        tb[3][4].n4_w = 42;
        // block (8, 5)
        tb[5][8].segmentation_idx = 14;
      }

      {
        // middle-right tile
        let tb = &mut tbs[5];
        // block (0, 1)
        tb[1][0].n4_h = 11;
        // block (7, 5)
        tb[5][7].cdef_index = 3;
      }

      {
        // bottom-middle tile
        let tb = &mut tbs[7];
        // block (3, 2)
        tb[2][3].mode = PredictionMode::PAETH_PRED;
        // block (1, 1)
        tb[1][1].n4_w = 8;
      }
    }

    // check that writes on tiles correctly affected the underlying blocks

    assert_eq!(42, fb[3][4].n4_w);
    assert_eq!(14, fb[5][8].segmentation_idx);

    assert_eq!(11, fb[17][32].n4_h);
    assert_eq!(3, fb[21][39].cdef_index);

    assert_eq!(PredictionMode::PAETH_PRED, fb[34][19].mode);
    assert_eq!(8, fb[33][17].n4_w);
  }
}
