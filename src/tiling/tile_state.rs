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
use crate::me::*;
use crate::partition::*;
use crate::plane::*;
use crate::quantize::*;
use crate::rdo::*;
use crate::util::*;

use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use std::slice;

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

/// Tiled view of FrameMotionVectors
#[derive(Debug)]
pub struct TileMotionVectors<'a> {
  data: *const MotionVector,
  // expressed in mi blocks
  x: usize,
  y: usize,
  cols: usize,
  rows: usize,
  stride: usize, // number of cols in the underlying FrameMotionVectors
  phantom: PhantomData<&'a MotionVector>,
}

/// Mutable tiled view of FrameMotionVectors
#[derive(Debug)]
pub struct TileMotionVectorsMut<'a> {
  data: *mut MotionVector,
  // expressed in mi blocks
  // cannot make these fields public, because they must not be written to,
  // otherwise we could break borrowing rules in safe code
  x: usize,
  y: usize,
  cols: usize,
  rows: usize,
  stride: usize, // number of cols in the underlying FrameMotionVectors
  phantom: PhantomData<&'a mut MotionVector>,
}

// common impl for TileMotionVectors and TileMotionVectorsMut
macro_rules! tile_motion_vectors_common {
  // $name: TileMotionVectors or TileMotionVectorsMut
  // $fmvs_ref_type: &'a FrameMotionVectors or &'a mut FrameMotionVectors
  // $index: index or index_mut
  ($name: ident, $fmv_ref_type: ty, $index: ident) => {
    impl<'a> $name<'a> {

      pub fn new(
        frame_mvs: $fmv_ref_type,
        x: usize,
        y: usize,
        cols: usize,
        rows: usize,
      ) -> Self {
        Self {
          data: frame_mvs.$index(y).$index(x), // &(mut) frame_mvs[y][x],
          x,
          y,
          cols,
          rows,
          stride: frame_mvs.cols,
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
      type Output = [MotionVector];
      #[inline]
      fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.rows);
        unsafe {
          let ptr = self.data.add(index * self.stride);
          slice::from_raw_parts(ptr, self.cols)
        }
      }
    }
  }
}

tile_motion_vectors_common!(TileMotionVectors, &'a FrameMotionVectors, index);
tile_motion_vectors_common!(TileMotionVectorsMut, &'a mut FrameMotionVectors, index_mut);

impl TileMotionVectorsMut<'_> {
  #[inline]
  pub fn as_const(&self) -> TileMotionVectors<'_> {
    TileMotionVectors {
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

impl IndexMut<usize> for TileMotionVectorsMut<'_> {
  #[inline]
  fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    assert!(index < self.rows);
    unsafe {
      let ptr = self.data.add(index * self.stride);
      slice::from_raw_parts_mut(ptr, self.cols)
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
  pub mvs: Vec<TileMotionVectorsMut<'a>>,
  pub rdo: RDOTracker,
}

impl<'a, T: Pixel> TileStateMut<'a, T> {
  pub fn new(
    fs: &'a mut FrameState<T>,
    luma_rect: Rect,
  ) -> Self {
    assert!(luma_rect.x >= 0);
    assert!(luma_rect.y >= 0);
    assert!(luma_rect.width & (MI_SIZE - 1) == 0, "luma_rect must be a multiple of MI_SIZE");
    assert!(luma_rect.height & (MI_SIZE - 1) == 0, "luma_rect must be a multiple of MI_SIZE");
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
      mvs: fs.frame_mvs.iter_mut().map(|fmvs| {
        TileMotionVectorsMut::new(
          fmvs,
          luma_rect.x as usize >> MI_SIZE_LOG2,
          luma_rect.y as usize >> MI_SIZE_LOG2,
          luma_rect.width >> MI_SIZE_LOG2,
          luma_rect.height >> MI_SIZE_LOG2,
        )
      }).collect(),
      rdo: RDOTracker::new(),
    }
  }

  #[inline]
  pub fn to_frame_plane_offset(&self, tile_po: PlaneOffset) -> PlaneOffset {
    PlaneOffset {
      x: self.luma_rect.x + tile_po.x,
      y: self.luma_rect.y + tile_po.y,
    }
  }

  #[inline]
  pub fn to_frame_block_offset(&self, tile_bo: BlockOffset) -> BlockOffset {
    debug_assert!(self.luma_rect.x as usize % MI_SIZE == 0);
    debug_assert!(self.luma_rect.y as usize % MI_SIZE == 0);
    BlockOffset {
      x: ((self.luma_rect.x as usize) >> MI_SIZE) + tile_bo.x,
      y: ((self.luma_rect.y as usize) >> MI_SIZE) + tile_bo.y,
    }
  }
}

pub struct TileStateIterMut<'a, T: Pixel> {
  fs: *mut FrameState<T>,
  frame_width: usize,
  frame_height: usize,
  tile_width: usize,
  tile_height: usize,
  cols: usize, // number of columns of tiles within the frame
  rows: usize, // number of rows of tiles within the frame
  next: usize, // index of the next tile to yield
  phantom: PhantomData<&'a mut FrameState<T>>,
}

impl<'a, T: Pixel> TileStateIterMut<'a, T> {

  /// Return an iterator over `TileStateMut`
  ///
  /// # Arguments
  ///
  /// * `fs` - The whole FrameState instance
  /// * `sb_size_log2` - The log2 size of a superblock (typically 6 or 7)
  /// * `tile_width_in_sb` - The width of a tile, in number of superblocks
  /// * `tile_height_in_sb` - The width of a tile, in number of superblocks
  pub fn from_frame_state(
    fs: &'a mut FrameState<T>,
    sb_size_log2: usize,
    tile_width_in_sb: usize,
    tile_height_in_sb: usize,
  ) -> Self {
    let PlaneConfig {
      width: frame_width,
      height: frame_height,
      ..
    } = fs.rec.planes[0].cfg;

    let frame_width_in_sb = frame_width.align_power_of_two_and_shift(sb_size_log2);
    let frame_height_in_sb = frame_width.align_power_of_two_and_shift(sb_size_log2);

    Self {
      fs,
      frame_width,
      frame_height,
      tile_width: tile_width_in_sb << sb_size_log2,
      tile_height: tile_height_in_sb << sb_size_log2,
      cols: (frame_width_in_sb + tile_width_in_sb - 1) / tile_width_in_sb,
      rows: (frame_height_in_sb + tile_height_in_sb - 1) / tile_height_in_sb,
      next: 0,
      phantom: PhantomData,
    }
  }
}

impl<'a, T: Pixel> Iterator for TileStateIterMut<'a, T> {
  type Item = TileStateMut<'a, T>;

  fn next(&mut self) -> Option<Self::Item> {
    if self.next < self.rows * self.cols {
      let fs = unsafe { &mut *self.fs };

      let x = (self.next % self.cols) * self.tile_width;
      let y = (self.next / self.cols) * self.tile_height;
      let width = self.tile_width.min(self.frame_width - x);
      let height = self.tile_height.min(self.frame_height - y);
      let luma_rect = Rect { x: x as isize, y: y as isize, width, height };

      self.next += 1;

      let ts = TileStateMut::new(fs, luma_rect);
      Some(ts)
    } else {
      None
    }
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    let remaining = self.cols * self.rows - self.next;
    (remaining, Some(remaining))
  }
}

impl<T: Pixel> ExactSizeIterator for TileStateIterMut<'_, T> {}

#[cfg(test)]
pub mod test {
  use super::*;
  use crate::api::*;

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

    {
      let mut iter = fs.tile_state_iter_mut(6, 2, 2);
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
      let mut iter = fs.tile_state_iter_mut(6, 1, 1);
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

    let tile_states = fs.tile_state_iter_mut(6, 1, 1).collect::<Vec<_>>();
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

  #[test]
  fn test_tile_state_write() {
    let fi = create_frame_invariants(160, 144, ChromaSampling::Cs420);
    let mut fs = FrameState::new(&fi);

    {
      let mut tile_states =
        fs.tile_state_iter_mut(6, 1, 1).collect::<Vec<_>>();

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

    {
      let mut tile_states =
        fs.tile_state_iter_mut(6, 1, 1).collect::<Vec<_>>();

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
}
