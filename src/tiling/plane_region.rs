// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::context::*;
use crate::plane::*;
use crate::util::*;

use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use std::slice;

#[derive(Debug, Clone, Copy)]
pub struct Rect {
  // coordinates relative to the plane origin (xorigin, yorigin)
  pub x: isize,
  pub y: isize,
  pub width: usize,
  pub height: usize,
}

impl Rect {
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

/// Structure to describe a rectangle area in several ways
///
/// To retrieve a subregion from a region, we need to provide the subregion
/// bounds, relative to its parent region. The subregion must always be included
/// in its parent region.
///
/// For that purpose, we could just use a rectangle (x, y, width, height), but
/// this would be too cumbersome to use in practice. For example, we often need
/// to pass a subregion from an offset, using the same bottom-right corner as
/// its parent, or to pass a subregion expressed in block offset instead of
/// pixel offset.
///
/// Area provides a flexible way to describe a subregion.
#[derive(Debug, Clone, Copy)]
pub enum Area {
  /// A well-defined rectangle
  Rect { x: isize, y: isize, width: usize, height: usize },
  /// A rectangle starting at offset (x, y) and ending at the bottom-right
  /// corner of the parent
  StartingAt { x: isize, y: isize },
  /// A well-defined rectangle with offset expressed in blocks
  BlockRect { bo: BlockOffset, width: usize, height: usize },
  /// a rectangle starting at given block offset until the bottom-right corner
  /// of the parent
  BlockStartingAt { bo: BlockOffset },
}

impl Area {
  #[inline]
  pub fn to_rect(
    &self,
    xdec: usize,
    ydec: usize,
    parent_width: usize,
    parent_height: usize,
  ) -> Rect {
    match *self {
      Area::Rect { x, y, width, height } => Rect { x, y, width, height },
      Area::StartingAt { x, y } => Rect {
        x,
        y,
        width: (parent_width as isize - x) as usize,
        height: (parent_height as isize - y) as usize,
      },
      Area::BlockRect { bo, width, height } => Rect {
        x: (bo.x >> xdec << BLOCK_TO_PLANE_SHIFT) as isize,
        y: (bo.y >> ydec << BLOCK_TO_PLANE_SHIFT) as isize,
        width,
        height,
      },
      Area::BlockStartingAt { bo } => Area::StartingAt {
        x: (bo.x >> xdec << BLOCK_TO_PLANE_SHIFT) as isize,
        y: (bo.y >> ydec << BLOCK_TO_PLANE_SHIFT) as isize,
      }.to_rect(xdec, ydec, parent_width, parent_height)
    }
  }
}

/// Bounded region of a plane
///
/// This allows to give access to a rectangular area of a plane without
/// giving access to the whole plane.
#[derive(Debug)]
pub struct PlaneRegion<'a, T: Pixel> {
  data: *const T, // points to (plane_cfg.x, plane_cfg.y)
  pub plane_cfg: &'a PlaneConfig,
  rect: Rect,
  phantom: PhantomData<&'a T>,
}

/// This allows to give mutable access to a rectangular area of the plane
/// without giving access to the whole plane.
#[derive(Debug)]
pub struct PlaneRegionMut<'a, T: Pixel> {
  data: *mut T, // points to (plane_cfg.x, plane_cfg.y)
  pub plane_cfg: &'a PlaneConfig,
  rect: Rect,
  phantom: PhantomData<&'a mut T>,
}

// common impl for PlaneRegion and PlaneRegionMut
macro_rules! plane_region_common {
  // $name: PlaneRegion or PlaneRegionMut
  // $plane_ref_type: &'a Plane<T> or &'a mut Plane<T>
  // $as_ptr: as_ptr or as_mut_ptr
  ($name: ident, $plane_ref_type: ty, $as_ptr: ident) => {
    impl<'a, T: Pixel> $name<'a, T> {

      pub fn new(plane: $plane_ref_type, rect: Rect) -> Self {
        assert!(plane.cfg.xorigin as isize + rect.x + rect.width as isize <= plane.cfg.stride as isize);
        assert!(plane.cfg.yorigin as isize + rect.y + rect.height as isize <= plane.cfg.alloc_height as isize);
        let origin = (plane.cfg.yorigin as isize + rect.y) * plane.cfg.stride as isize
                     + plane.cfg.xorigin as isize + rect.x;
        Self {
          data: unsafe { plane.data.$as_ptr().offset(origin) },
          plane_cfg: &plane.cfg,
          rect,
          phantom: PhantomData,
        }
      }

      #[inline]
      pub fn data_ptr(&self) -> *const T {
        self.data
      }

      #[inline]
      pub fn rect(&self) -> &Rect {
        // cannot make the field public, because it must not be written to,
        // otherwise we could break borrowing rules in safe code
        &self.rect
      }

      #[inline]
      pub fn rows_iter(&self) -> RowsIter<'_, T> {
        RowsIter {
          data: self.data,
          stride: self.plane_cfg.stride,
          width: self.rect.width,
          remaining: self.rect.height,
          phantom: PhantomData,
        }
      }

      /// Return a view to a subregion of the plane
      ///
      /// The subregion must be included in (i.e. must not exceed) this region.
      ///
      /// It is described by an `Area`, relative to this region.
      ///
      /// # Example
      ///
      /// ```
      /// # use rav1e::tiling::*;
      /// # fn f(region: &PlaneRegion<'_, u16>) {
      /// // a subregion from (10, 8) to the end of the region
      /// let subregion = region.subregion(Area::StartingAt { x: 10, y: 8 });
      /// # }
      /// ```
      ///
      /// ```
      /// # use rav1e::context::*;
      /// # use rav1e::tiling::*;
      /// # fn f(region: &PlaneRegion<'_, u16>) {
      /// // a subregion from the top-left of block (2, 3) having size (64, 64)
      /// let bo = BlockOffset { x: 2, y: 3 };
      /// let subregion = region.subregion(Area::BlockRect { bo, width: 64, height: 64 });
      /// # }
      /// ```
      pub fn subregion(&self, area: Area) -> PlaneRegion<'_, T> {
        let rect = area.to_rect(
          self.plane_cfg.xdec,
          self.plane_cfg.ydec,
          self.rect.width,
          self.rect.height,
        );
        assert!(rect.x >= 0 && rect.x as usize <= self.rect.width);
        assert!(rect.y >= 0 && rect.y as usize <= self.rect.height);
        let data = unsafe {
          self.data.add(rect.y as usize * self.plane_cfg.stride + rect.x as usize)
        };
        let absolute_rect = Rect {
          x: self.rect.x + rect.x,
          y: self.rect.y + rect.y,
          width: rect.width,
          height: rect.height,
        };
        PlaneRegion {
          data,
          plane_cfg: &self.plane_cfg,
          rect: absolute_rect,
          phantom: PhantomData,
        }
      }
    }

    unsafe impl<T: Pixel> Send for $name<'_, T> {}
    unsafe impl<T: Pixel> Sync for $name<'_, T> {}

    impl<'a, T: Pixel> Index<usize> for $name<'a, T> {
      type Output = [T];

      fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.rect.height);
        unsafe {
          let ptr = self.data.add(index * self.plane_cfg.stride);
          slice::from_raw_parts(ptr, self.rect.width)
        }
      }
    }
  }
}

plane_region_common!(PlaneRegion, &'a Plane<T>, as_ptr);
plane_region_common!(PlaneRegionMut, &'a mut Plane<T>, as_mut_ptr);

impl<'a, T: Pixel> PlaneRegionMut<'a, T> {
  #[inline]
  pub fn data_ptr_mut(&mut self) -> *mut T {
    self.data
  }

  #[inline]
  pub fn rows_iter_mut(&mut self) -> RowsIterMut<'_, T> {
    RowsIterMut {
      data: self.data,
      stride: self.plane_cfg.stride,
      width: self.rect.width,
      remaining: self.rect.height,
      phantom: PhantomData,
    }
  }

  /// Return a mutable view to a subregion of the plane
  ///
  /// The subregion must be included in (i.e. must not exceed) this region.
  ///
  /// It is described by an `Area`, relative to this region.
  ///
  /// # Example
  ///
  /// ```
  /// # use rav1e::tiling::*;
  /// # fn f(region: &mut PlaneRegionMut<'_, u16>) {
  /// // a mutable subregion from (10, 8) having size (32, 32)
  /// let subregion = region.subregion_mut(Area::Rect { x: 10, y: 8, width: 32, height: 32 });
  /// # }
  /// ```
  ///
  /// ```
  /// # use rav1e::context::*;
  /// # use rav1e::tiling::*;
  /// # fn f(region: &mut PlaneRegionMut<'_, u16>) {
  /// // a mutable subregion from the top-left of block (2, 3) to the end of the region
  /// let bo = BlockOffset { x: 2, y: 3 };
  /// let subregion = region.subregion_mut(Area::BlockStartingAt { bo });
  /// # }
  /// ```
  pub fn subregion_mut(&mut self, area: Area) -> PlaneRegionMut<'_, T> {
    let rect = area.to_rect(
      self.plane_cfg.xdec,
      self.plane_cfg.ydec,
      self.rect.width,
      self.rect.height,
    );
    assert!(rect.x >= 0 && rect.x as usize <= self.rect.width);
    assert!(rect.y >= 0 && rect.y as usize <= self.rect.height);
    let data = unsafe {
      self.data.add(rect.y as usize * self.plane_cfg.stride + rect.x as usize)
    };
    let absolute_rect = Rect {
      x: self.rect.x + rect.x,
      y: self.rect.y + rect.y,
      width: rect.width,
      height: rect.height,
    };
    PlaneRegionMut {
      data,
      plane_cfg: &self.plane_cfg,
      rect: absolute_rect,
      phantom: PhantomData,
    }
  }

  #[inline]
  pub fn as_const(&self) -> PlaneRegion<'_, T> {
    PlaneRegion {
      data: self.data,
      plane_cfg: self.plane_cfg,
      rect: self.rect,
      phantom: PhantomData,
    }
  }
}

impl<'a, T: Pixel> IndexMut<usize> for PlaneRegionMut<'a, T> {
  fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    assert!(index < self.rect.height);
    unsafe {
      let ptr = self.data.add(index * self.plane_cfg.stride);
      slice::from_raw_parts_mut(ptr, self.rect.width)
    }
  }
}

pub struct RowsIter<'a, T: Pixel> {
  data: *const T,
  stride: usize,
  width: usize,
  remaining: usize,
  phantom: PhantomData<&'a T>,
}

pub struct RowsIterMut<'a, T: Pixel> {
  data: *mut T,
  stride: usize,
  width: usize,
  remaining: usize,
  phantom: PhantomData<&'a mut T>,
}

impl<'a, T: Pixel> Iterator for RowsIter<'a, T> {
  type Item = &'a [T];

  fn next(&mut self) -> Option<Self::Item> {
    if self.remaining > 0 {
      let row = unsafe {
        let ptr = self.data;
        self.data = self.data.add(self.stride);
        slice::from_raw_parts(ptr, self.width)
      };
      Some(row)
    } else {
      None
    }
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    (self.remaining, Some(self.remaining))
  }
}

impl<'a, T: Pixel> Iterator for RowsIterMut<'a, T> {
  type Item = &'a mut [T];

  fn next(&mut self) -> Option<Self::Item> {
    if self.remaining > 0 {
      let row = unsafe {
        let ptr = self.data;
        self.data = self.data.add(self.stride);
        slice::from_raw_parts_mut(ptr, self.width)
      };
      Some(row)
    } else {
      None
    }
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    (self.remaining, Some(self.remaining))
  }
}

impl<T: Pixel> ExactSizeIterator for RowsIter<'_, T> {}
impl<T: Pixel> ExactSizeIterator for RowsIterMut<'_, T> {}
