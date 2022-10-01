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
  (*^) = dimap from to . gscale
  {-# INLINE (*^) #-}
  (>.<) :: v -> v -> k
  default (>.<) :: GenericVectorSpace k v => v -> v -> k
  (>.<) = gdot `on` from
  {-# INLINE (>.<) #-}
  reps :: k -> v
  default reps :: GenericVectorSpace k v => k -> v
  reps = to . greps
  {-# INLINE reps #-}
  sums :: v -> k
  default sums :: GenericVectorSpace k v => v -> k
  sums = gsums . from
  {-# INLINE sums #-}

type GenericVectorSpace k v = (Generic v, GVectorSpace k (Rep v))

class GAdd f where
  gadd :: f () -> f () -> f ()

class (Num k, GAdd f) => GVectorSpace k f where
  gscale :: k -> f () -> f ()
  gdot :: f () -> f () -> k
  greps :: k -> f ()
  gsums :: f () -> k

instance GAdd U1 where
  gadd = mempty
  {-# INLINE gadd #-}

instance Num k => GVectorSpace k U1 where
  gscale = const $ const U1
  {-# INLINE gscale #-}
  gdot = const $ const 0
  {-# INLINE gdot #-}
  greps = const U1
  {-# INLINE greps #-}
  gsums = const 0
  {-# INLINE gsums #-}

instance GAdd f => GAdd (M1 i c f) where
  gadd = coerce $ gadd @f
  {-# INLINE gadd #-}

instance GVectorSpace k f => GVectorSpace k (M1 i c f) where
  gscale = coerce $ gscale @k @f
  {-# INLINE gscale #-}
  gdot = coerce $ gdot @k @f
  {-# INLINE gdot #-}
  greps = coerce $ greps @k @f
  {-# INLINE greps #-}
  gsums = coerce $ gsums @k @f
  {-# INLINE gsums #-}

instance (GAdd f, GAdd g) => GAdd (f :*: g) where
  gadd (f1 :*: g1) (f2 :*: g2) = gadd f1 f2 :*: gadd g1 g2
  {-# INLINE gadd #-}

instance (GVectorSpace k f, GVectorSpace k g) => GVectorSpace k (f :*: g) where
  gscale c (f :*: g) = gscale c f :*: gscale c g
  {-# INLINE gscale #-}
  gdot (f1 :*: g1) (f2 :*: g2) = gdot f1 f2 + gdot g1 g2
  {-# INLINE gdot #-}
  greps = (:*:) <$> greps <*> greps
  {-# INLINE greps #-}
  gsums (f :*: g) = gsums f + gsums g
  {-# INLINE gsums #-}

instance Additive v => GAdd (K1 i v) where
  gadd = coerce $ (^+^) @v
  {-# INLINE gadd #-}

instance VectorSpace k c => GVectorSpace k (K1 i c) where
  gscale = coerce $ (*^) @k @c
  {-# INLINE gscale #-}
  gdot = coerce $ (>.<) @k @c
  {-# INLINE gdot #-}
  greps = coerce $ reps @k @c
  {-# INLINE greps #-}
  gsums = coerce $ sums @k @c
  {-# INLINE gsums #-}

instance Additive Double where
  (^+^) = (+)
  {-# INLINE (^+^) #-}

instance VectorSpace Double Double where
  (*^) = (*)
  {-# INLINE (*^) #-}
  (>.<) = (*)
  {-# INLINE (>.<) #-}
  reps = id
  {-# INLINE reps #-}
  sums = id
  {-# INLINE sums #-}

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
  reps = liftOp1 $
    op1 $ \v ->
      (reps v, const 0)
  {-# INLINE reps #-}
  sums = liftOp1 $
    op1 $ \v ->
      (sums v, reps)
  {-# INLINE sums #-}
