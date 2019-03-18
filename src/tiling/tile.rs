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

#[derive(Debug)]
pub struct Tile<'a, T: Pixel> {
  pub planes: [PlaneRegion<'a, T>; PLANES],
}

#[derive(Debug)]
pub struct TileMut<'a, T: Pixel> {
  pub planes: [PlaneRegionMut<'a, T>; PLANES],
}

impl<'a, T: Pixel> Tile<'a, T> {
  pub fn new(frame: &'a Frame<T>, luma_rect: Rect) -> Self {
    Self {
      planes: [
        {
          let plane = &frame.planes[0];
          PlaneRegion::new(plane, luma_rect)
        },
        {
          let plane = &frame.planes[1];
          let rect = luma_rect.decimated(plane.cfg.xdec, plane.cfg.ydec);
          PlaneRegion::new(plane, rect)
        },
        {
          let plane = &frame.planes[2];
          let rect = luma_rect.decimated(plane.cfg.xdec, plane.cfg.ydec);
          PlaneRegion::new(plane, rect)
        },
      ],
    }
  }
}

impl<'a, T: Pixel> TileMut<'a, T> {
  pub fn new(frame: &'a mut Frame<T>, luma_rect: Rect) -> Self {
    // we cannot retrieve &mut of slice items directly and safely
    let mut planes_iter = frame.planes.iter_mut();
    Self {
      planes: [
        {
          let plane = planes_iter.next().unwrap();
          PlaneRegionMut::new(plane, luma_rect)
        },
        {
          let plane = planes_iter.next().unwrap();
          let rect = luma_rect.decimated(plane.cfg.xdec, plane.cfg.ydec);
          PlaneRegionMut::new(plane, rect)
        },
        {
          let plane = planes_iter.next().unwrap();
          let rect = luma_rect.decimated(plane.cfg.xdec, plane.cfg.ydec);
          PlaneRegionMut::new(plane, rect)
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
