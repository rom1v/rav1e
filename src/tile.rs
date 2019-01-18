// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use context::*;
use encoder::*;
use lrf::*;
use plane::*;
use quantize::*;

use std::marker::PhantomData;
use std::slice;

#[derive(Debug, Clone)]
pub struct PlaneRegionConfig {
  // coordinates of the region relative to the raw plane (including padding)
  pub xorigin: usize,
  pub yorigin: usize,
  pub width: usize,
  pub height: usize,
}

#[derive(Debug, Clone)]
pub struct PlaneRegionMut<'a> {
  data: *mut u16,
  pub plane_cfg: &'a PlaneConfig,
  pub cfg: PlaneRegionConfig,
  phantom: PhantomData<&'a mut u16>,
}

impl<'a> PlaneRegionMut<'a> {
  // exposed as unsafe because nothing prevents the caller to retrieve overlapping regions
  unsafe fn new(plane: &'a mut Plane, cfg: PlaneRegionConfig) -> Self {
    Self {
      data: plane.data.as_mut_ptr(),
      plane_cfg: &plane.cfg,
      cfg,
      phantom: PhantomData,
    }
  }

  unsafe fn from_luma_coordinates(
    plane: &'a mut Plane,
    luma_x: usize,
    luma_y: usize,
    luma_width: usize,
    luma_height: usize,
  ) -> Self {
    let x = luma_x >> plane.cfg.xdec;
    let y = luma_y >> plane.cfg.ydec;
    let w = luma_width >> plane.cfg.xdec;
    let h = luma_height >> plane.cfg.ydec;

    let xorigin = plane.cfg.xorigin + x;
    let yorigin = plane.cfg.yorigin + y;
    let width = w.min(plane.cfg.width - x);
    let height = h.min(plane.cfg.height - y);

    let cfg = PlaneRegionConfig { xorigin, yorigin, width, height };

    Self::new(plane, cfg)
  }

  pub fn row(&self, y: usize) -> &[u16] {
    assert!(y < self.cfg.height);
    let offset =
      (self.cfg.yorigin + y) * self.plane_cfg.stride + self.cfg.xorigin;
    unsafe { slice::from_raw_parts(self.data.add(offset), self.cfg.width) }
  }

  pub fn row_mut(&mut self, y: usize) -> &mut [u16] {
    assert!(y < self.cfg.height);
    let offset =
      (self.cfg.yorigin + y) * self.plane_cfg.stride + self.cfg.xorigin;
    unsafe { slice::from_raw_parts_mut(self.data.add(offset), self.cfg.width) }
  }

  pub fn subregion_sized(&'a self, x: usize, y: usize, width: usize, height: usize) -> PlaneSubRegion<'a> {
    PlaneSubRegion {
      region: self,
      cfg: PlaneSubRegionConfig { x, y, width, height },
    }
  }

  pub fn subregion(&'a self, x: usize, y: usize) -> PlaneSubRegion<'a> {
    let width = self.cfg.width - x;
    let height = self.cfg.height - y;
    self.subregion_sized(x, y, width, height)
  }

  pub fn subregion_sized_mut(&'a mut self, x: usize, y: usize, width: usize, height: usize) -> PlaneSubRegionMut<'a> {
    PlaneSubRegionMut {
      region: self,
      cfg: PlaneSubRegionConfig { x, y, width, height },
    }
  }

  pub fn subregion_mut(&'a mut self, x: usize, y: usize) -> PlaneSubRegionMut<'a> {
    let width = self.cfg.width - x;
    let height = self.cfg.height - y;
    self.subregion_sized_mut(x, y, width, height)
  }

  pub fn data_ptr(&self) -> *const u16 {
    let offset = self.cfg.yorigin * self.plane_cfg.stride + self.cfg.xorigin;
    unsafe { self.data.add(offset) }
  }

  pub fn data_ptr_mut(&mut self) -> *mut u16 {
    let offset = self.cfg.yorigin * self.plane_cfg.stride + self.cfg.xorigin;
    unsafe { self.data.add(offset) }
  }
}

#[derive(Debug, Clone)]
pub struct PlaneSubRegionConfig {
  // coordinates of the subregion relative to the region
  pub x: usize,
  pub y: usize,
  pub width: usize,
  pub height: usize,
}

pub struct PlaneSubRegion<'a> {
  region: &'a PlaneRegionMut<'a>,
  cfg: PlaneSubRegionConfig,
}

pub struct PlaneSubRegionMut<'a> {
  region: &'a mut PlaneRegionMut<'a>,
  cfg: PlaneSubRegionConfig,
}

impl<'a> PlaneSubRegion<'a> {
  pub fn row(&self, y: usize) -> &[u16] {
    let region_row = self.region.row(self.cfg.y + y);
    &region_row[self.cfg.x..self.cfg.x + self.cfg.width]
  }
}

impl<'a> PlaneSubRegionMut<'a> {
  pub fn row(&self, y: usize) -> &[u16] {
    let region_row = self.region.row(self.cfg.y + y);
    &region_row[self.cfg.x..self.cfg.x + self.cfg.width]
  }

  pub fn row_mut(&mut self, y: usize) -> &mut [u16] {
    let region_row = self.region.row_mut(self.cfg.y + y);
    &mut region_row[self.cfg.x..self.cfg.x + self.cfg.width]
  }
}

#[derive(Debug, Clone)]
pub struct TileMut<'a> {
  pub planes: [PlaneRegionMut<'a>; PLANES],
}

impl<'a> TileMut<'a> {
  unsafe fn new(
    frame: &'a mut Frame,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
  ) -> Self {
    // we cannot retrieve &mut of slice items directly and safely
    let mut planes_iter = frame.planes.iter_mut();
    Self {
      planes: {
        [
          PlaneRegionMut::from_luma_coordinates(
            planes_iter.next().unwrap(),
            x,
            y,
            width,
            height,
          ),
          PlaneRegionMut::from_luma_coordinates(
            planes_iter.next().unwrap(),
            x,
            y,
            width,
            height,
          ),
          PlaneRegionMut::from_luma_coordinates(
            planes_iter.next().unwrap(),
            x,
            y,
            width,
            height,
          ),
        ]
      },
    }
  }
}

pub struct TileIterMut<'a> {
  frame: *mut Frame,
  tile_width: usize,
  tile_height: usize,
  x: usize,
  y: usize,
  phantom: PhantomData<&'a mut Frame>,
}

impl<'a> TileIterMut<'a> {
  pub fn from_frame(
    frame: &'a mut Frame,
    tile_width: usize,
    tile_height: usize,
  ) -> Self {
    Self { frame, x: 0, y: 0, tile_width, tile_height, phantom: PhantomData }
  }
}

impl<'a> Iterator for TileIterMut<'a> {
  type Item = TileMut<'a>;

  fn next(&mut self) -> Option<TileMut<'a>> {
    let frame = unsafe { &mut *self.frame };
    let width = frame.planes[0].cfg.width;
    let height = frame.planes[0].cfg.height;
    if self.x >= width || self.y >= height {
      None
    } else {
      // use current (x, y) for the current TileMut
      let (x, y) = (self.x, self.y);

      // update (self.x, self.y) for the next TileMut
      self.x += self.tile_width;
      if self.x >= width {
        self.x = 0;
        self.y += self.tile_height;
      }

      let tile = unsafe {
        TileMut::new(frame, x, y, self.tile_width, self.tile_height)
      };
      Some(tile)
    }
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    let frame = unsafe { &mut *self.frame };
    let planes = &mut frame.planes;
    let width = planes[0].cfg.width;
    let height = planes[0].cfg.height;

    let cols = (width + self.tile_width - 1) / self.tile_width;
    let rows = (height + self.tile_height - 1) / self.tile_height;

    let consumed =
      (self.y / self.tile_height) * rows + (self.x / self.tile_width);
    let remaining = cols * rows - consumed;

    (remaining, Some(remaining))
  }
}

#[derive(Clone, Debug)]
pub struct RestorationTilePlane<'a> {
  pub plane: &'a RestorationPlane,
  pub wiener_ref: [[i8; 3]; 2],
  pub sgrproj_ref: [i8; 2],
}

impl<'a> RestorationTilePlane<'a> {
  pub fn new(rp: &'a RestorationPlane) -> Self {
    Self {
      plane: rp,
      wiener_ref: [WIENER_TAPS_MID; 2],
      sgrproj_ref: SGRPROJ_XQD_MID,
    }
  }
}

// contrary to other views, RestorationTileState is not exposed as mutable
// it is (possibly) shared between several tiles (due to restoration unit stretching)
// so it provides "interior" mutability protected by mutex or atomic access
#[derive(Clone, Debug)]
pub struct RestorationTileState<'a> {
  pub planes: [RestorationTilePlane<'a>; PLANES],
}

impl<'a> RestorationTileState<'a> {
  pub fn new(rs: &'a RestorationState) -> Self {
    Self {
      planes: [
        RestorationTilePlane::new(&rs.plane[0]),
        RestorationTilePlane::new(&rs.plane[1]),
        RestorationTilePlane::new(&rs.plane[2]),
      ],
    }
  }
}

#[derive(Clone, Debug)]
pub struct TileStateMut<'a> {
  pub input: &'a Frame,
  pub input_hres: &'a Plane,
  pub input_qres: &'a Plane,
  pub x: usize,
  pub y: usize,
  pub width: usize,
  pub height: usize,
  pub rec: TileMut<'a>,
  pub qc: QuantizationContext,
  pub cdfs: CDFContext,
  pub segmentation: SegmentationState,
  pub restoration: RestorationTileState<'a>,
}

impl<'a> TileStateMut<'a> {
  // exposed as unsafe because nothing prevents the caller to retrieve overlapping regions
  unsafe fn new(
    fs: &'a mut FrameState,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
  ) -> Self {
    Self {
      input: &fs.input,
      input_hres: &fs.input_hres,
      input_qres: &fs.input_qres,
      x,
      y,
      width,
      height,
      rec: TileMut::new(&mut fs.rec, x, y, width, height),
      qc: Default::default(),
      cdfs: CDFContext::new(0),
      segmentation: Default::default(),
      restoration: RestorationTileState::new(&fs.restoration),
    }
  }
}

pub struct TileStateIterMut<'a> {
  fs: *mut FrameState,
  tile_width: usize,
  tile_height: usize,
  tile_cols: usize,
  tile_rows: usize,
  next: usize, // index of the next tile to provide
  phantom: PhantomData<&'a mut FrameState>,
}

impl<'a> TileStateIterMut<'a> {
  pub fn from_frame_state(
    fs: &'a mut FrameState,
    tile_width: usize,
    tile_height: usize,
  ) -> Self {
    let frame_width = fs.rec.planes[0].cfg.width;
    let frame_height = fs.rec.planes[0].cfg.height;

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

impl<'a> Iterator for TileStateIterMut<'a> {
  type Item = TileStateMut<'a>;

  fn next(&mut self) -> Option<TileStateMut<'a>> {
    if self.next < self.tile_rows * self.tile_cols {
      let x = (self.next % self.tile_cols) * self.tile_width;
      let y = (self.next / self.tile_rows) * self.tile_height;
      self.next += 1;

      let ts = unsafe {
        let fs = &mut *self.fs;
        TileStateMut::new(fs, x, y, self.tile_width, self.tile_height)
      };
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

#[cfg(test)]
pub mod test {
  use super::*;
  use api::*;

  #[test]
  fn test_tile_count() {
    let mut frame = Frame::new(80, 60, ChromaSampling::Cs420);

    {
      let mut iter = frame.tile_iter_mut(40, 30);
      assert_eq!(4, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(3, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(2, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(1, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(0, iter.size_hint().0);
      assert!(iter.next().is_none());
    }

    {
      let mut iter = frame.tile_iter_mut(32, 24);
      assert_eq!(9, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(8, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(7, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(6, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(5, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(4, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(3, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(2, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(1, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(0, iter.size_hint().0);
      assert!(iter.next().is_none());
    }
  }

  #[test]
  fn test_tile_area() {
    let mut frame = Frame::new(72, 68, ChromaSampling::Cs420);

    let planes_origins = [
      (frame.planes[0].cfg.xorigin, frame.planes[0].cfg.yorigin),
      (frame.planes[1].cfg.xorigin, frame.planes[1].cfg.yorigin),
      (frame.planes[2].cfg.xorigin, frame.planes[2].cfg.yorigin),
    ];

    let tiles = frame.tile_iter_mut(32, 32).collect::<Vec<_>>();
    // the frame must be split into 9 tiles:
    //
    //       luma (Y)             chroma (U)            chroma (V)
    //   32x32 32x32  8x32     16x16 16x16  4x16     16x16 16x16  4x16
    //   32x32 32x32  8x32     16x16 16x16  4x16     16x16 16x16  4x16
    //   32x 4 32x 4  8x 4     16x 2 16x 2  4x 2     16x 2 16x 2  4x 2

    assert_eq!(9, tiles.len());

    let tile = &tiles[0]; // the top-left tile
    assert_eq!(32, tile.planes[0].cfg.width);
    assert_eq!(32, tile.planes[0].cfg.height);
    assert_eq!(16, tile.planes[1].cfg.width);
    assert_eq!(16, tile.planes[1].cfg.height);
    assert_eq!(16, tile.planes[2].cfg.width);
    assert_eq!(16, tile.planes[2].cfg.height);
    assert_eq!(planes_origins[0].0, tile.planes[0].cfg.xorigin);
    assert_eq!(planes_origins[0].1, tile.planes[0].cfg.yorigin);
    assert_eq!(planes_origins[1].0, tile.planes[1].cfg.xorigin);
    assert_eq!(planes_origins[1].1, tile.planes[1].cfg.yorigin);
    assert_eq!(planes_origins[2].0, tile.planes[2].cfg.xorigin);
    assert_eq!(planes_origins[2].1, tile.planes[2].cfg.yorigin);

    let tile = &tiles[1]; // the top-middle tile
    assert_eq!(32, tile.planes[0].cfg.width);
    assert_eq!(32, tile.planes[0].cfg.height);
    assert_eq!(16, tile.planes[1].cfg.width);
    assert_eq!(16, tile.planes[1].cfg.height);
    assert_eq!(16, tile.planes[2].cfg.width);
    assert_eq!(16, tile.planes[2].cfg.height);
    assert_eq!(planes_origins[0].0 + 32, tile.planes[0].cfg.xorigin);
    assert_eq!(planes_origins[0].1, tile.planes[0].cfg.yorigin);
    assert_eq!(planes_origins[1].0 + 16, tile.planes[1].cfg.xorigin);
    assert_eq!(planes_origins[1].1, tile.planes[1].cfg.yorigin);
    assert_eq!(planes_origins[2].0 + 16, tile.planes[2].cfg.xorigin);
    assert_eq!(planes_origins[2].1, tile.planes[2].cfg.yorigin);

    let tile = &tiles[2]; // the top-right tile
    assert_eq!(8, tile.planes[0].cfg.width);
    assert_eq!(32, tile.planes[0].cfg.height);
    assert_eq!(4, tile.planes[1].cfg.width);
    assert_eq!(16, tile.planes[1].cfg.height);
    assert_eq!(4, tile.planes[2].cfg.width);
    assert_eq!(16, tile.planes[2].cfg.height);
    assert_eq!(planes_origins[0].0 + 64, tile.planes[0].cfg.xorigin);
    assert_eq!(planes_origins[0].1, tile.planes[0].cfg.yorigin);
    assert_eq!(planes_origins[1].0 + 32, tile.planes[1].cfg.xorigin);
    assert_eq!(planes_origins[1].1, tile.planes[1].cfg.yorigin);
    assert_eq!(planes_origins[2].0 + 32, tile.planes[2].cfg.xorigin);
    assert_eq!(planes_origins[2].1, tile.planes[2].cfg.yorigin);

    let tile = &tiles[3]; // the middle-left tile
    assert_eq!(32, tile.planes[0].cfg.width);
    assert_eq!(32, tile.planes[0].cfg.height);
    assert_eq!(16, tile.planes[1].cfg.width);
    assert_eq!(16, tile.planes[1].cfg.height);
    assert_eq!(16, tile.planes[2].cfg.width);
    assert_eq!(16, tile.planes[2].cfg.height);
    assert_eq!(planes_origins[0].0, tile.planes[0].cfg.xorigin);
    assert_eq!(planes_origins[0].1 + 32, tile.planes[0].cfg.yorigin);
    assert_eq!(planes_origins[1].0, tile.planes[1].cfg.xorigin);
    assert_eq!(planes_origins[1].1 + 16, tile.planes[1].cfg.yorigin);
    assert_eq!(planes_origins[2].0, tile.planes[2].cfg.xorigin);
    assert_eq!(planes_origins[2].1 + 16, tile.planes[2].cfg.yorigin);

    let tile = &tiles[4]; // the center tile
    assert_eq!(32, tile.planes[0].cfg.width);
    assert_eq!(32, tile.planes[0].cfg.height);
    assert_eq!(16, tile.planes[1].cfg.width);
    assert_eq!(16, tile.planes[1].cfg.height);
    assert_eq!(16, tile.planes[2].cfg.width);
    assert_eq!(16, tile.planes[2].cfg.height);
    assert_eq!(planes_origins[0].0 + 32, tile.planes[0].cfg.xorigin);
    assert_eq!(planes_origins[0].1 + 32, tile.planes[0].cfg.yorigin);
    assert_eq!(planes_origins[1].0 + 16, tile.planes[1].cfg.xorigin);
    assert_eq!(planes_origins[1].1 + 16, tile.planes[1].cfg.yorigin);
    assert_eq!(planes_origins[2].0 + 16, tile.planes[2].cfg.xorigin);
    assert_eq!(planes_origins[2].1 + 16, tile.planes[2].cfg.yorigin);

    let tile = &tiles[5]; // the middle-right tile
    assert_eq!(8, tile.planes[0].cfg.width);
    assert_eq!(32, tile.planes[0].cfg.height);
    assert_eq!(4, tile.planes[1].cfg.width);
    assert_eq!(16, tile.planes[1].cfg.height);
    assert_eq!(4, tile.planes[2].cfg.width);
    assert_eq!(16, tile.planes[2].cfg.height);
    assert_eq!(planes_origins[0].0 + 64, tile.planes[0].cfg.xorigin);
    assert_eq!(planes_origins[0].1 + 32, tile.planes[0].cfg.yorigin);
    assert_eq!(planes_origins[1].0 + 32, tile.planes[1].cfg.xorigin);
    assert_eq!(planes_origins[1].1 + 16, tile.planes[1].cfg.yorigin);
    assert_eq!(planes_origins[2].0 + 32, tile.planes[2].cfg.xorigin);
    assert_eq!(planes_origins[2].1 + 16, tile.planes[2].cfg.yorigin);

    let tile = &tiles[6]; // the bottom-left tile
    assert_eq!(32, tile.planes[0].cfg.width);
    assert_eq!(4, tile.planes[0].cfg.height);
    assert_eq!(16, tile.planes[1].cfg.width);
    assert_eq!(2, tile.planes[1].cfg.height);
    assert_eq!(16, tile.planes[2].cfg.width);
    assert_eq!(2, tile.planes[2].cfg.height);
    assert_eq!(planes_origins[0].0, tile.planes[0].cfg.xorigin);
    assert_eq!(planes_origins[0].1 + 64, tile.planes[0].cfg.yorigin);
    assert_eq!(planes_origins[1].0, tile.planes[1].cfg.xorigin);
    assert_eq!(planes_origins[1].1 + 32, tile.planes[1].cfg.yorigin);
    assert_eq!(planes_origins[2].0, tile.planes[2].cfg.xorigin);
    assert_eq!(planes_origins[2].1 + 32, tile.planes[2].cfg.yorigin);

    let tile = &tiles[7]; // the bottom-middle tile
    assert_eq!(32, tile.planes[0].cfg.width);
    assert_eq!(4, tile.planes[0].cfg.height);
    assert_eq!(16, tile.planes[1].cfg.width);
    assert_eq!(2, tile.planes[1].cfg.height);
    assert_eq!(16, tile.planes[2].cfg.width);
    assert_eq!(2, tile.planes[2].cfg.height);
    assert_eq!(planes_origins[0].0 + 32, tile.planes[0].cfg.xorigin);
    assert_eq!(planes_origins[0].1 + 64, tile.planes[0].cfg.yorigin);
    assert_eq!(planes_origins[1].0 + 16, tile.planes[1].cfg.xorigin);
    assert_eq!(planes_origins[1].1 + 32, tile.planes[1].cfg.yorigin);
    assert_eq!(planes_origins[2].0 + 16, tile.planes[2].cfg.xorigin);
    assert_eq!(planes_origins[2].1 + 32, tile.planes[2].cfg.yorigin);

    let tile = &tiles[8]; // the bottom-right tile
    assert_eq!(8, tile.planes[0].cfg.width);
    assert_eq!(4, tile.planes[0].cfg.height);
    assert_eq!(4, tile.planes[1].cfg.width);
    assert_eq!(2, tile.planes[1].cfg.height);
    assert_eq!(4, tile.planes[2].cfg.width);
    assert_eq!(2, tile.planes[2].cfg.height);
    assert_eq!(planes_origins[0].0 + 64, tile.planes[0].cfg.xorigin);
    assert_eq!(planes_origins[0].1 + 64, tile.planes[0].cfg.yorigin);
    assert_eq!(planes_origins[1].0 + 32, tile.planes[1].cfg.xorigin);
    assert_eq!(planes_origins[1].1 + 32, tile.planes[1].cfg.yorigin);
    assert_eq!(planes_origins[2].0 + 32, tile.planes[2].cfg.xorigin);
    assert_eq!(planes_origins[2].1 + 32, tile.planes[2].cfg.yorigin);
  }

  #[test]
  fn test_tile_write() {
    let mut frame = Frame::new(72, 68, ChromaSampling::Cs420);

    {
      let mut tiles = frame.tile_iter_mut(32, 32).collect::<Vec<_>>();

      {
        // row 12 of Y-plane of the top-left tile
        let row = tiles[0].planes[0].row_mut(12);
        assert_eq!(32, row.len());
        &mut row[5..11].copy_from_slice(&[4, 42, 12, 18, 15, 31]);
      }

      {
        // row 8 of U-plane of the middle-right tile
        let row = tiles[5].planes[1].row_mut(8);
        assert_eq!(4, row.len());
        &mut row[..].copy_from_slice(&[14, 121, 1, 3]);
      }

      {
        // row 1 of V-plane of the bottom-middle tile
        let row = tiles[7].planes[2].row_mut(1);
        assert_eq!(16, row.len());
        &mut row[11..16].copy_from_slice(&[6, 5, 2, 11, 8]);
      }
    }

    // check that writes on tiles correctly affected the underlying frame

    let plane = &frame.planes[0];
    let y = plane.cfg.yorigin + 12;
    let x = plane.cfg.xorigin + 5;
    let idx = y * plane.cfg.stride + x;
    assert_eq!(&[4, 42, 12, 18, 15, 31], &plane.data[idx..idx + 6]);

    let plane = &frame.planes[1];
    let offset = (32, 16); // middle-right tile, chroma plane
    let y = plane.cfg.yorigin + offset.1 + 8;
    let x = plane.cfg.xorigin + offset.0;
    let idx = y * plane.cfg.stride + x;
    assert_eq!(&[14, 121, 1, 3], &plane.data[idx..idx + 4]);

    let plane = &frame.planes[2];
    let offset = (16, 32); // bottom-middle tile, chroma plane
    let y = plane.cfg.yorigin + offset.1 + 1;
    let x = plane.cfg.xorigin + offset.0 + 11;
    let idx = y * plane.cfg.stride + x;
    assert_eq!(&[6, 5, 2, 11, 8], &plane.data[idx..idx + 5]);
  }

  fn create_frame_invariants(width: usize, height: usize, chroma_sampling: ChromaSampling) -> FrameInvariants {
    // FrameInvariants aligns to the next multiple of 8, so using other values could make tests confusing
    assert!(width & 7 == 0);
    assert!(height & 7 == 0);
    let config = Default::default();
    let frame_info = FrameInfo {
      width,
      height,
      bit_depth: 8,
      chroma_sampling,
      ..Default::default()
    };
    let sequence = Sequence::new(&frame_info);
    FrameInvariants::new(width, height, config, sequence)
  }

  #[test]
  fn test_tile_state_count() {
    let fi = create_frame_invariants(80, 64, ChromaSampling::Cs420);
    let mut fs = FrameState::new(&fi);

    {
      let mut iter = fs.tile_state_iter_mut(40, 32);
      assert_eq!(4, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(3, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(2, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(1, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(0, iter.size_hint().0);
      assert!(iter.next().is_none());
    }

    {
      let mut iter = fs.tile_state_iter_mut(32, 24);
      assert_eq!(9, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(8, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(7, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(6, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(5, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(4, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(3, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(2, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(1, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(0, iter.size_hint().0);
      assert!(iter.next().is_none());
    }
  }

  #[test]
  fn test_tile_state_area() {
    let fi = create_frame_invariants(80, 72, ChromaSampling::Cs420);
    let mut fs = FrameState::new(&fi);

    let planes_origins = [
      (fs.rec.planes[0].cfg.xorigin, fs.rec.planes[0].cfg.yorigin),
      (fs.rec.planes[1].cfg.xorigin, fs.rec.planes[1].cfg.yorigin),
      (fs.rec.planes[2].cfg.xorigin, fs.rec.planes[2].cfg.yorigin),
    ];

    let tile_states = fs.tile_state_iter_mut(32, 32).collect::<Vec<_>>();
    // the frame must be split into 9 tiles:
    //
    //       luma (Y)             chroma (U)            chroma (V)
    //   32x32 32x32 16x32     16x16 16x16  8x16     16x16 16x16  8x16
    //   32x32 32x32 16x32     16x16 16x16  8x16     16x16 16x16  8x16
    //   32x 8 32x 8 16x 8     16x 4 16x 4  8x 4     16x 4 16x 4  8x 4

    assert_eq!(9, tile_states.len());

    let tile = &tile_states[0].rec; // the top-left tile
    assert_eq!(32, tile.planes[0].cfg.width);
    assert_eq!(32, tile.planes[0].cfg.height);
    assert_eq!(16, tile.planes[1].cfg.width);
    assert_eq!(16, tile.planes[1].cfg.height);
    assert_eq!(16, tile.planes[2].cfg.width);
    assert_eq!(16, tile.planes[2].cfg.height);
    assert_eq!(planes_origins[0].0, tile.planes[0].cfg.xorigin);
    assert_eq!(planes_origins[0].1, tile.planes[0].cfg.yorigin);
    assert_eq!(planes_origins[1].0, tile.planes[1].cfg.xorigin);
    assert_eq!(planes_origins[1].1, tile.planes[1].cfg.yorigin);
    assert_eq!(planes_origins[2].0, tile.planes[2].cfg.xorigin);
    assert_eq!(planes_origins[2].1, tile.planes[2].cfg.yorigin);

    let tile = &tile_states[1].rec; // the top-middle tile
    assert_eq!(32, tile.planes[0].cfg.width);
    assert_eq!(32, tile.planes[0].cfg.height);
    assert_eq!(16, tile.planes[1].cfg.width);
    assert_eq!(16, tile.planes[1].cfg.height);
    assert_eq!(16, tile.planes[2].cfg.width);
    assert_eq!(16, tile.planes[2].cfg.height);
    assert_eq!(planes_origins[0].0 + 32, tile.planes[0].cfg.xorigin);
    assert_eq!(planes_origins[0].1, tile.planes[0].cfg.yorigin);
    assert_eq!(planes_origins[1].0 + 16, tile.planes[1].cfg.xorigin);
    assert_eq!(planes_origins[1].1, tile.planes[1].cfg.yorigin);
    assert_eq!(planes_origins[2].0 + 16, tile.planes[2].cfg.xorigin);
    assert_eq!(planes_origins[2].1, tile.planes[2].cfg.yorigin);

    let tile = &tile_states[2].rec; // the top-right tile
    assert_eq!(16, tile.planes[0].cfg.width);
    assert_eq!(32, tile.planes[0].cfg.height);
    assert_eq!(8, tile.planes[1].cfg.width);
    assert_eq!(16, tile.planes[1].cfg.height);
    assert_eq!(8, tile.planes[2].cfg.width);
    assert_eq!(16, tile.planes[2].cfg.height);
    assert_eq!(planes_origins[0].0 + 64, tile.planes[0].cfg.xorigin);
    assert_eq!(planes_origins[0].1, tile.planes[0].cfg.yorigin);
    assert_eq!(planes_origins[1].0 + 32, tile.planes[1].cfg.xorigin);
    assert_eq!(planes_origins[1].1, tile.planes[1].cfg.yorigin);
    assert_eq!(planes_origins[2].0 + 32, tile.planes[2].cfg.xorigin);
    assert_eq!(planes_origins[2].1, tile.planes[2].cfg.yorigin);

    let tile = &tile_states[3].rec; // the middle-left tile
    assert_eq!(32, tile.planes[0].cfg.width);
    assert_eq!(32, tile.planes[0].cfg.height);
    assert_eq!(16, tile.planes[1].cfg.width);
    assert_eq!(16, tile.planes[1].cfg.height);
    assert_eq!(16, tile.planes[2].cfg.width);
    assert_eq!(16, tile.planes[2].cfg.height);
    assert_eq!(planes_origins[0].0, tile.planes[0].cfg.xorigin);
    assert_eq!(planes_origins[0].1 + 32, tile.planes[0].cfg.yorigin);
    assert_eq!(planes_origins[1].0, tile.planes[1].cfg.xorigin);
    assert_eq!(planes_origins[1].1 + 16, tile.planes[1].cfg.yorigin);
    assert_eq!(planes_origins[2].0, tile.planes[2].cfg.xorigin);
    assert_eq!(planes_origins[2].1 + 16, tile.planes[2].cfg.yorigin);

    let tile = &tile_states[4].rec; // the center tile
    assert_eq!(32, tile.planes[0].cfg.width);
    assert_eq!(32, tile.planes[0].cfg.height);
    assert_eq!(16, tile.planes[1].cfg.width);
    assert_eq!(16, tile.planes[1].cfg.height);
    assert_eq!(16, tile.planes[2].cfg.width);
    assert_eq!(16, tile.planes[2].cfg.height);
    assert_eq!(planes_origins[0].0 + 32, tile.planes[0].cfg.xorigin);
    assert_eq!(planes_origins[0].1 + 32, tile.planes[0].cfg.yorigin);
    assert_eq!(planes_origins[1].0 + 16, tile.planes[1].cfg.xorigin);
    assert_eq!(planes_origins[1].1 + 16, tile.planes[1].cfg.yorigin);
    assert_eq!(planes_origins[2].0 + 16, tile.planes[2].cfg.xorigin);
    assert_eq!(planes_origins[2].1 + 16, tile.planes[2].cfg.yorigin);

    let tile = &tile_states[5].rec; // the middle-right tile
    assert_eq!(16, tile.planes[0].cfg.width);
    assert_eq!(32, tile.planes[0].cfg.height);
    assert_eq!(8, tile.planes[1].cfg.width);
    assert_eq!(16, tile.planes[1].cfg.height);
    assert_eq!(8, tile.planes[2].cfg.width);
    assert_eq!(16, tile.planes[2].cfg.height);
    assert_eq!(planes_origins[0].0 + 64, tile.planes[0].cfg.xorigin);
    assert_eq!(planes_origins[0].1 + 32, tile.planes[0].cfg.yorigin);
    assert_eq!(planes_origins[1].0 + 32, tile.planes[1].cfg.xorigin);
    assert_eq!(planes_origins[1].1 + 16, tile.planes[1].cfg.yorigin);
    assert_eq!(planes_origins[2].0 + 32, tile.planes[2].cfg.xorigin);
    assert_eq!(planes_origins[2].1 + 16, tile.planes[2].cfg.yorigin);

    let tile = &tile_states[6].rec; // the bottom-left tile
    assert_eq!(32, tile.planes[0].cfg.width);
    assert_eq!(8, tile.planes[0].cfg.height);
    assert_eq!(16, tile.planes[1].cfg.width);
    assert_eq!(4, tile.planes[1].cfg.height);
    assert_eq!(16, tile.planes[2].cfg.width);
    assert_eq!(4, tile.planes[2].cfg.height);
    assert_eq!(planes_origins[0].0, tile.planes[0].cfg.xorigin);
    assert_eq!(planes_origins[0].1 + 64, tile.planes[0].cfg.yorigin);
    assert_eq!(planes_origins[1].0, tile.planes[1].cfg.xorigin);
    assert_eq!(planes_origins[1].1 + 32, tile.planes[1].cfg.yorigin);
    assert_eq!(planes_origins[2].0, tile.planes[2].cfg.xorigin);
    assert_eq!(planes_origins[2].1 + 32, tile.planes[2].cfg.yorigin);

    let tile = &tile_states[7].rec; // the bottom-middle tile
    assert_eq!(32, tile.planes[0].cfg.width);
    assert_eq!(8, tile.planes[0].cfg.height);
    assert_eq!(16, tile.planes[1].cfg.width);
    assert_eq!(4, tile.planes[1].cfg.height);
    assert_eq!(16, tile.planes[2].cfg.width);
    assert_eq!(4, tile.planes[2].cfg.height);
    assert_eq!(planes_origins[0].0 + 32, tile.planes[0].cfg.xorigin);
    assert_eq!(planes_origins[0].1 + 64, tile.planes[0].cfg.yorigin);
    assert_eq!(planes_origins[1].0 + 16, tile.planes[1].cfg.xorigin);
    assert_eq!(planes_origins[1].1 + 32, tile.planes[1].cfg.yorigin);
    assert_eq!(planes_origins[2].0 + 16, tile.planes[2].cfg.xorigin);
    assert_eq!(planes_origins[2].1 + 32, tile.planes[2].cfg.yorigin);

    let tile = &tile_states[8].rec; // the bottom-right tile
    assert_eq!(16, tile.planes[0].cfg.width);
    assert_eq!(8, tile.planes[0].cfg.height);
    assert_eq!(8, tile.planes[1].cfg.width);
    assert_eq!(4, tile.planes[1].cfg.height);
    assert_eq!(8, tile.planes[2].cfg.width);
    assert_eq!(4, tile.planes[2].cfg.height);
    assert_eq!(planes_origins[0].0 + 64, tile.planes[0].cfg.xorigin);
    assert_eq!(planes_origins[0].1 + 64, tile.planes[0].cfg.yorigin);
    assert_eq!(planes_origins[1].0 + 32, tile.planes[1].cfg.xorigin);
    assert_eq!(planes_origins[1].1 + 32, tile.planes[1].cfg.yorigin);
    assert_eq!(planes_origins[2].0 + 32, tile.planes[2].cfg.xorigin);
    assert_eq!(planes_origins[2].1 + 32, tile.planes[2].cfg.yorigin);
  }

  #[test]
  fn test_tile_state_write() {
    let fi = create_frame_invariants(80, 72, ChromaSampling::Cs420);
    let mut fs = FrameState::new(&fi);

    {
      let mut tile_states = fs.tile_state_iter_mut(32, 32).collect::<Vec<_>>();

      {
        // row 12 of Y-plane of the top-left tile
        let row = tile_states[0].rec.planes[0].row_mut(12);
        assert_eq!(32, row.len());
        &mut row[5..11].copy_from_slice(&[4, 42, 12, 18, 15, 31]);
      }

      {
        // row 8 of U-plane of the middle-right tile
        let row = tile_states[5].rec.planes[1].row_mut(8);
        assert_eq!(8, row.len());
        &mut row[..4].copy_from_slice(&[14, 121, 1, 3]);
      }

      {
        // row 1 of V-plane of the bottom-middle tile
        let row = tile_states[7].rec.planes[2].row_mut(1);
        assert_eq!(16, row.len());
        &mut row[11..16].copy_from_slice(&[6, 5, 2, 11, 8]);
      }
    }

    // check that writes on tiles correctly affected the underlying frame

    let plane = &fs.rec.planes[0];
    let y = plane.cfg.yorigin + 12;
    let x = plane.cfg.xorigin + 5;
    let idx = y * plane.cfg.stride + x;
    assert_eq!(&[4, 42, 12, 18, 15, 31], &plane.data[idx..idx + 6]);

    let plane = &fs.rec.planes[1];
    let offset = (32, 16); // middle-right tile, chroma plane
    let y = plane.cfg.yorigin + offset.1 + 8;
    let x = plane.cfg.xorigin + offset.0;
    let idx = y * plane.cfg.stride + x;
    assert_eq!(&[14, 121, 1, 3], &plane.data[idx..idx + 4]);

    let plane = &fs.rec.planes[2];
    let offset = (16, 32); // bottom-middle tile, chroma plane
    let y = plane.cfg.yorigin + offset.1 + 1;
    let x = plane.cfg.xorigin + offset.0 + 11;
    let idx = y * plane.cfg.stride + x;
    assert_eq!(&[6, 5, 2, 11, 8], &plane.data[idx..idx + 5]);
  }
}
