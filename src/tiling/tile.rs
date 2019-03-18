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

// Same as Rect (used by PlaneRegion), but with unsigned (x, y) for convenience
#[derive(Debug, Clone, Copy)]
pub struct TileRect {
  pub x: usize,
  pub y: usize,
  pub width: usize,
  pub height: usize,
}

impl TileRect {
  #[inline(always)]
  pub fn decimated(&self, xdec: usize, ydec: usize) -> Self {
    Self {
      x: self.x >> xdec,
      y: self.y >> ydec,
      width: self.width >> xdec,
      height: self.height >> ydec,
    }
  }
}

impl From<TileRect> for Rect {
  #[inline(always)]
  fn from(tile_rect: TileRect) -> Rect {
    Rect {
      x: tile_rect.x as isize,
      y: tile_rect.y as isize,
      width: tile_rect.width,
      height: tile_rect.height,
    }
  }
}

#[derive(Debug)]
pub struct Tile<'a, T: Pixel> {
  pub planes: [PlaneRegion<'a, T>; PLANES],
}

#[derive(Debug)]
pub struct TileMut<'a, T: Pixel> {
  pub planes: [PlaneRegionMut<'a, T>; PLANES],
}

impl<'a, T: Pixel> Tile<'a, T> {
  pub fn new(frame: &'a Frame<T>, luma_rect: TileRect) -> Self {
    Self {
      planes: [
        {
          let plane = &frame.planes[0];
          PlaneRegion::new(plane, luma_rect.into())
        },
        {
          let plane = &frame.planes[1];
          let rect = luma_rect.decimated(plane.cfg.xdec, plane.cfg.ydec);
          PlaneRegion::new(plane, rect.into())
        },
        {
          let plane = &frame.planes[2];
          let rect = luma_rect.decimated(plane.cfg.xdec, plane.cfg.ydec);
          PlaneRegion::new(plane, rect.into())
        },
      ],
    }
  }
}

impl<'a, T: Pixel> TileMut<'a, T> {
  pub fn new(frame: &'a mut Frame<T>, luma_rect: TileRect) -> Self {
    // we cannot retrieve &mut of slice items directly and safely
    let mut planes_iter = frame.planes.iter_mut();
    Self {
      planes: [
        {
          let plane = planes_iter.next().unwrap();
          PlaneRegionMut::new(plane, luma_rect.into())
        },
        {
          let plane = planes_iter.next().unwrap();
          let rect = luma_rect.decimated(plane.cfg.xdec, plane.cfg.ydec);
          PlaneRegionMut::new(plane, rect.into())
        },
        {
          let plane = planes_iter.next().unwrap();
          let rect = luma_rect.decimated(plane.cfg.xdec, plane.cfg.ydec);
          PlaneRegionMut::new(plane, rect.into())
        },
      ],
    }
  }

  #[inline]
  pub fn as_const(&self) -> Tile<'_, T> {
    Tile {
      planes: [
        self.planes[0].as_const(),
        self.planes[1].as_const(),
        self.planes[2].as_const(),
      ],
    }
  }
}
