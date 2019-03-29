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

#[cfg(test)]
pub mod test {
  use super::*;
  use crate::api::*;
  use crate::partition::*;

  fn create_frame_invariants(
    width: usize,
    height: usize,
    chroma_sampling: ChromaSampling,
  ) -> FrameInvariants<u16> {
    // FrameInvariants aligns to the next multiple of 8, so using other values could make tests confusing
    assert!(width & 7 == 0);
    assert!(height & 7 == 0);
    let config = EncoderConfig {
      width,
      height,
      bit_depth: 8,
      chroma_sampling,
      ..Default::default()
    };
    let sequence = Sequence::new(&config);
    FrameInvariants::new(config, sequence)
  }

  #[test]
  fn test_tile_state_iter_len() {
    let fi = create_frame_invariants(160, 144, ChromaSampling::Cs420);
    let mut fs = FrameState::new(&fi);
    // frame size 160x144, 40x36 in 4x4-blocks
    let mut fb = FrameBlocks::new(fi.w_in_b, fi.h_in_b);

    {
      let ti = TilingInfo::from_tile_size_sb(&fi, 2, 2);
      let mut iter = ti.tile_iter_mut(&mut fs, &mut fb);
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
      let ti = TilingInfo::from_tile_size_sb(&fi, 1, 1);
      let mut iter = ti.tile_iter_mut(&mut fs, &mut fb);
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
  fn rect<T: Pixel>(region: &PlaneRegionMut<'_, T>) -> (isize, isize, usize, usize) {
    let &Rect { x, y, width, height } = region.rect();
    (x, y, width, height)
  }

  #[test]
  fn test_tile_state_area() {
    let fi = create_frame_invariants(160, 144, ChromaSampling::Cs420);
    let mut fs = FrameState::new(&fi);
    let mut fb = FrameBlocks::new(fi.w_in_b, fi.h_in_b);

    let ti = TilingInfo::from_tile_size_sb(&fi, 1, 1);
    let iter = ti.tile_iter_mut(&mut fs, &mut fb);
    let tile_states = iter.map(|ctx| ctx.ts).collect::<Vec<_>>();

    // the frame must be split into 9 tiles:
    //
    //       luma (Y)             chroma (U)            chroma (V)
    //   64x64 64x64 32x64     32x32 32x32 16x32     32x32 32x32 16x32
    //   64x64 64x64 32x64     32x32 32x32 16x32     32x32 32x32 16x32
    //   64x16 64x16 32x16     32x 8 32x 8 16x 8     32x 8 32x 8 16x 8

    assert_eq!(9, tile_states.len());

    let tile = &tile_states[0].rec; // the top-left tile
    assert_eq!((0, 0, 64, 64), rect(&tile.planes[0]));
    assert_eq!((0, 0, 32, 32), rect(&tile.planes[1]));
    assert_eq!((0, 0, 32, 32), rect(&tile.planes[2]));

    let tile = &tile_states[1].rec; // the top-middle tile
    assert_eq!((64, 0, 64, 64), rect(&tile.planes[0]));
    assert_eq!((32, 0, 32, 32), rect(&tile.planes[1]));
    assert_eq!((32, 0, 32, 32), rect(&tile.planes[2]));

    let tile = &tile_states[2].rec; // the top-right tile
    assert_eq!((128, 0, 32, 64), rect(&tile.planes[0]));
    assert_eq!((64, 0, 16, 32), rect(&tile.planes[1]));
    assert_eq!((64, 0, 16, 32), rect(&tile.planes[2]));

    let tile = &tile_states[3].rec; // the middle-left tile
    assert_eq!((0, 64, 64, 64), rect(&tile.planes[0]));
    assert_eq!((0, 32, 32, 32), rect(&tile.planes[1]));
    assert_eq!((0, 32, 32, 32), rect(&tile.planes[2]));

    let tile = &tile_states[4].rec; // the center tile
    assert_eq!((64, 64, 64, 64), rect(&tile.planes[0]));
    assert_eq!((32, 32, 32, 32), rect(&tile.planes[1]));
    assert_eq!((32, 32, 32, 32), rect(&tile.planes[2]));

    let tile = &tile_states[5].rec; // the middle-right tile
    assert_eq!((128, 64, 32, 64), rect(&tile.planes[0]));
    assert_eq!((64, 32, 16, 32), rect(&tile.planes[1]));
    assert_eq!((64, 32, 16, 32), rect(&tile.planes[2]));

    let tile = &tile_states[6].rec; // the bottom-left tile
    assert_eq!((0, 128, 64, 16), rect(&tile.planes[0]));
    assert_eq!((0, 64, 32, 8), rect(&tile.planes[1]));
    assert_eq!((0, 64, 32, 8), rect(&tile.planes[2]));

    let tile = &tile_states[7].rec; // the bottom-middle tile
    assert_eq!((64, 128, 64, 16), rect(&tile.planes[0]));
    assert_eq!((32, 64, 32, 8), rect(&tile.planes[1]));
    assert_eq!((32, 64, 32, 8), rect(&tile.planes[2]));

    let tile = &tile_states[8].rec; // the bottom-right tile
    assert_eq!((128, 128, 32, 16), rect(&tile.planes[0]));
    assert_eq!((64, 64, 16, 8), rect(&tile.planes[1]));
    assert_eq!((64, 64, 16, 8), rect(&tile.planes[2]));
  }

  #[inline]
  fn b_area(region: &BlocksRegionMut<'_>) -> (usize, usize, usize, usize) {
    (region.x(), region.y(), region.cols(), region.rows())
  }

  #[test]
  fn test_tile_blocks_area() {
    let fi = create_frame_invariants(160, 144, ChromaSampling::Cs420);
    let mut fs = FrameState::new(&fi);
    let mut fb = FrameBlocks::new(fi.w_in_b, fi.h_in_b);

    let ti = TilingInfo::from_tile_size_sb(&fi, 1, 1);
    let iter = ti.tile_iter_mut(&mut fs, &mut fb);
    let tbs = iter.map(|ctx| ctx.tb).collect::<Vec<_>>();

    // the FrameBlocks must be split into 9 BlocksRegion:
    //
    //   16x16 16x16  8x16
    //   16x16 16x16  8x16
    //   16x 4 16x4   8x 4

    assert_eq!(9, tbs.len());

    assert_eq!((0, 0, 16, 16), b_area(&tbs[0]));
    assert_eq!((16, 0, 16, 16), b_area(&tbs[1]));
    assert_eq!((32, 0, 8, 16), b_area(&tbs[2]));

    assert_eq!((0, 16, 16, 16), b_area(&tbs[3]));
    assert_eq!((16, 16, 16, 16), b_area(&tbs[4]));
    assert_eq!((32, 16, 8, 16), b_area(&tbs[5]));

    assert_eq!((0, 32, 16, 4), b_area(&tbs[6]));
    assert_eq!((16, 32, 16, 4), b_area(&tbs[7]));
    assert_eq!((32, 32, 8, 4), b_area(&tbs[8]));
  }

  #[test]
  fn test_tile_state_write() {
    let fi = create_frame_invariants(160, 144, ChromaSampling::Cs420);
    let mut fs = FrameState::new(&fi);
    let mut fb = FrameBlocks::new(fi.w_in_b, fi.h_in_b);

    {
      let ti = TilingInfo::from_tile_size_sb(&fi, 1, 1);
      let iter = ti.tile_iter_mut(&mut fs, &mut fb);
      let mut tile_states = iter.map(|ctx| ctx.ts).collect::<Vec<_>>();

      {
        // row 12 of Y-plane of the top-left tile
        let tile_plane = &mut tile_states[0].rec.planes[0];
        let row = &mut tile_plane[12];
        assert_eq!(64, row.len());
        &mut row[35..41].copy_from_slice(&[4, 42, 12, 18, 15, 31]);
      }

      {
        // row 8 of U-plane of the middle-right tile
        let tile_plane = &mut tile_states[5].rec.planes[1];
        let row = &mut tile_plane[8];
        assert_eq!(16, row.len());
        &mut row[..4].copy_from_slice(&[14, 121, 1, 3]);
      }

      {
        // row 1 of V-plane of the bottom-middle tile
        let tile_plane = &mut tile_states[7].rec.planes[2];
        let row = &mut tile_plane[1];
        assert_eq!(32, row.len());
        &mut row[11..16].copy_from_slice(&[6, 5, 2, 11, 8]);
      }
    }

    // check that writes on tiles correctly affected the underlying frame

    let plane = &fs.rec.planes[0];
    let y = plane.cfg.yorigin + 12;
    let x = plane.cfg.xorigin + 35;
    let idx = y * plane.cfg.stride + x;
    assert_eq!(&[4, 42, 12, 18, 15, 31], &plane.data[idx..idx + 6]);

    let plane = &fs.rec.planes[1];
    let offset = (64, 32); // middle-right tile, chroma plane
    let y = plane.cfg.yorigin + offset.1 + 8;
    let x = plane.cfg.xorigin + offset.0;
    let idx = y * plane.cfg.stride + x;
    assert_eq!(&[14, 121, 1, 3], &plane.data[idx..idx + 4]);

    let plane = &fs.rec.planes[2];
    let offset = (32, 64); // bottom-middle tile, chroma plane
    let y = plane.cfg.yorigin + offset.1 + 1;
    let x = plane.cfg.xorigin + offset.0 + 11;
    let idx = y * plane.cfg.stride + x;
    assert_eq!(&[6, 5, 2, 11, 8], &plane.data[idx..idx + 5]);
  }

  #[test]
  fn test_motion_vectors_write() {
    let fi = create_frame_invariants(160, 144, ChromaSampling::Cs420);
    let mut fs = FrameState::new(&fi);
    let mut fb = FrameBlocks::new(fi.w_in_b, fi.h_in_b);

    {
      let ti = TilingInfo::from_tile_size_sb(&fi, 1, 1);
      let iter = ti.tile_iter_mut(&mut fs, &mut fb);
      let mut tile_states = iter.map(|ctx| ctx.ts).collect::<Vec<_>>();

      {
        // block (8, 5) of the top-left tile (of the first ref frame)
        let mvs = &mut tile_states[0].mvs[0];
        mvs[5][8] = MotionVector { col: 42, row: 38 };
        println!("{:?}", mvs[5][8]);
      }

      {
        // block (4, 2) of the middle-right tile (of ref frame 2)
        let mvs = &mut tile_states[5].mvs[2];
        mvs[2][3] = MotionVector { col: 2, row: 14 };
      }
    }

    // check that writes on tiled views affected the underlying motion vectors

    let mvs = &fs.frame_mvs[0];
    assert_eq!(MotionVector { col: 42, row: 38 }, mvs[5][8]);

    let mvs = &fs.frame_mvs[2];
    let mix = (128 >> MI_SIZE_LOG2) + 3;
    let miy = (64 >> MI_SIZE_LOG2) + 2;
    assert_eq!(MotionVector { col: 2, row: 14 }, mvs[miy][mix]);
  }

  #[test]
  fn test_tile_blocks_write() {
    let fi = create_frame_invariants(160, 144, ChromaSampling::Cs420);
    let mut fs = FrameState::new(&fi);
    let mut fb = FrameBlocks::new(fi.w_in_b, fi.h_in_b);

    {
      let ti = TilingInfo::from_tile_size_sb(&fi, 1, 1);
      let iter = ti.tile_iter_mut(&mut fs, &mut fb);
      let mut tbs = iter.map(|ctx| ctx.tb).collect::<Vec<_>>();

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
