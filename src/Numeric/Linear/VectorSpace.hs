{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}

module Numeric.Linear.VectorSpace
  ( VectorSpace (..),
    GenericVectorSpace,
    Additive (..),
    GenericAdditive,
  )
where

import Control.Lens (Profunctor (dimap))
import Data.Coerce (coerce)
import Data.Function (on)
import GHC.Generics
import Numeric.Backprop

infixr 8 *^

infix 7 >.<

infixl 6 ^+^

class Additive v where
  (^+^) :: v -> v -> v
  default (^+^) :: GenericAdditive v => v -> v -> v
  (^+^) = fmap to . (gadd `on` from)
  {-# INLINE (^+^) #-}

type GenericAdditive v = (Generic v, GAdd (Rep v))

class (Num k, Additive v) => VectorSpace k v where
  (*^) :: k -> v -> v
  default (*^) :: GenericVectorSpace k v => k -> v -> v
  {-# INLINE (*^) #-}
  (*^) = dimap from to . gscale
  (>.<) :: v -> v -> k
  default (>.<) :: GenericVectorSpace k v => v -> v -> k
  {-# INLINE (>.<) #-}
  (>.<) = gdot `on` from

type GenericVectorSpace k v = (Generic v, GVectorSpace k (Rep v))

class GAdd f where
  gadd :: f () -> f () -> f ()

class (Num k, GAdd f) => GVectorSpace k f where
  gscale :: k -> f () -> f ()
  gdot :: f () -> f () -> k

instance GAdd U1 where
  gadd = mempty
  {-# INLINE gadd #-}

instance Num k => GVectorSpace k U1 where
  gscale = const $ const U1
  {-# INLINE gscale #-}
  gdot = const $ const 0
  {-# INLINE gdot #-}

instance GAdd f => GAdd (M1 i c f) where
  gadd = coerce $ gadd @f
  {-# INLINE gadd #-}

instance GVectorSpace k f => GVectorSpace k (M1 i c f) where
  gscale = coerce $ gscale @k @f
  {-# INLINE gscale #-}
  gdot = coerce $ gdot @k @f
  {-# INLINE gdot #-}

instance (GAdd f, GAdd g) => GAdd (f :*: g) where
  gadd (f1 :*: g1) (f2 :*: g2) = gadd f1 f2 :*: gadd g1 g2
  {-# INLINE gadd #-}

instance (GVectorSpace k f, GVectorSpace k g) => GVectorSpace k (f :*: g) where
  gscale c (f :*: g) = gscale c f :*: gscale c g
  {-# INLINE gscale #-}
  gdot (f1 :*: g1) (f2 :*: g2) = gdot f1 f2 + gdot g1 g2
  {-# INLINE gdot #-}

instance Additive v => GAdd (K1 i v) where
  gadd = coerce $ (^+^) @v
  {-# INLINE gadd #-}

instance VectorSpace k c => GVectorSpace k (K1 i c) where
  gscale = coerce $ (*^) @k @c
  {-# INLINE gscale #-}
  gdot = coerce $ (>.<) @k @c
  {-# INLINE gdot #-}

instance Additive Double where
  (^+^) = (+)
  {-# INLINE (^+^) #-}

instance VectorSpace Double Double where
  (*^) = (*)
  {-# INLINE (*^) #-}
  (>.<) = (*)
  {-# INLINE (>.<) #-}

instance (Backprop v, Reifies s W) => Additive (BVar s v) where
  (^+^) = add
  {-# INLINE (^+^) #-}

instance
  (VectorSpace k v, Reifies s W, Backprop k, Backprop v) =>
  VectorSpace (BVar s k) (BVar s v)
  where
  (*^) = liftOp2 $
    op2 $ \c v ->
      (c *^ v, \dz -> (dz >.< v, c *^ dz))
  {-# INLINE (*^) #-}
  (>.<) = liftOp2 $
    op2 $ \v u ->
      (v >.< u, \dz -> (dz *^ u, dz *^ v))
  {-# INLINE (>.<) #-}
