{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# OPTIONS_GHC -fllvm -funbox-strict-fields #-}

module Numeric.Linear.Accelerate.Internal
  ( ToShape,
    AccTensor (..),
    RawTensor (..),
    KnownDims (..),
    SDims (..),
    sizeDims,
    repeatedScalar,
    repeatRaw,
    asScalar,
    dimVal,
    repeatedExp,
    mapTensor,
    zipTensorWith,
    foldTensor,
  )
where

import Data.Array.Accelerate as A hiding (generate)
import Data.Coerce (coerce)
import Data.Kind (Constraint, Type)
import Data.Proxy (Proxy (..))
import Data.Type.Natural
import Numeric.Backprop (Backprop (..))
import Numeric.Linear.VectorSpace
import RIO (NFData)
import Prelude as P

dimVal :: forall n. KnownNat n => Int
dimVal = P.fromIntegral $ natVal @n Proxy

type SDims :: [Nat] -> Type
data SDims dims where
  SZ :: SDims '[]
  (:%) :: SNat n -> SDims ns -> SDims (n ': ns)

infixr 9 :%

deriving instance Show (SDims ns)

deriving instance P.Eq (SDims ns)

deriving instance P.Ord (SDims ns)

newtype RawTensor dims a = RawTensor (Array (ToShape dims) a)

deriving newtype instance (KnownDims dims, A.Elt a) => NFData (RawTensor dims a)

deriving newtype instance
  (KnownDims dims, A.Elt a) => Arrays (RawTensor dims a)

type ToShape :: [Nat] -> Type
type family ToShape dims where
  ToShape '[] = Z
  ToShape (n ': ns) = ToShape ns :. Int

deriving instance (KnownDims dims, Elt a, Show a) => Show (RawTensor dims a)

newtype AccTensor dims a = AccTensor (Acc (Array (ToShape dims) a))

deriving instance (KnownDims dims, Elt a) => Show (AccTensor dims a)

type KnownDims :: [Nat] -> Constraint
class
  ( Plain (ToShape dims) ~ ToShape dims
  , Elt (Plain (ToShape dims))
  , Shape (ToShape dims)
  , Slice (ToShape dims)
  , P.Eq (CoSliceShape (CoSliceShape (ToShape dims)))
  , KnownDims' dims
  ) =>
  KnownDims dims
  where
  sDims :: SDims dims
  theShape :: ToShape dims

sizeDims :: forall dims. KnownDims dims => Int
sizeDims = go $ sDims @dims
  where
    go :: SDims ds -> Int
    go SZ = 1
    go (sn :% sd') = P.fromIntegral (toNatural sn) * go sd'

instance KnownDims '[] where
  sDims = SZ
  {-# INLINE sDims #-}
  theShape = Z
  {-# INLINE theShape #-}

instance (KnownNat n, KnownDims ns) => KnownDims (n ': ns) where
  sDims = sNat :% sDims
  {-# INLINE sDims #-}
  theShape = theShape @ns :. P.fromIntegral (natVal @n Proxy)
  {-# INLINE theShape #-}

type KnownDims' :: [Nat] -> Constraint
type family KnownDims' ns where
  KnownDims' '[] = ()
  KnownDims' (n ': ns) = (KnownNat n, KnownDims ns)

instance (KnownDims dims, A.Num a) => Backprop (AccTensor dims a) where
  zero = const 0
  {-# INLINE zero #-}
  one = const 1
  {-# INLINE one #-}
  add = (+)
  {-# INLINE add #-}

instance (A.Num a, KnownDims dims) => P.Num (AccTensor dims a) where
  (+) = zipTensorWith (+)
  {-# INLINE (+) #-}
  (-) = zipTensorWith (-)
  {-# INLINE (-) #-}
  (*) = zipTensorWith (*)
  {-# INLINE (*) #-}
  abs = mapTensor abs
  {-# INLINE abs #-}
  signum = mapTensor signum
  {-# INLINE signum #-}
  fromInteger = repeatedExp . P.fromInteger
  {-# INLINE fromInteger #-}
  negate = mapTensor negate
  {-# INLINE negate #-}

instance (A.Fractional a, KnownDims dims) => P.Fractional (AccTensor dims a) where
  fromRational = repeatedExp . P.fromRational
  {-# INLINE fromRational #-}
  (/) = zipTensorWith (/)
  {-# INLINE (/) #-}
  recip = mapTensor recip
  {-# INLINE recip #-}

instance (A.Floating a, KnownDims dims) => P.Floating (AccTensor dims a) where
  pi = repeatedExp pi
  exp = mapTensor exp
  {-# INLINE exp #-}
  log = mapTensor log
  {-# INLINE log #-}
  logBase = zipTensorWith logBase
  {-# INLINE logBase #-}
  sin = mapTensor sin
  {-# INLINE sin #-}
  cos = mapTensor cos
  {-# INLINE cos #-}
  tan = mapTensor tan
  {-# INLINE tan #-}
  asin = mapTensor asin
  {-# INLINE asin #-}
  acos = mapTensor acos
  {-# INLINE acos #-}
  atan = mapTensor atan
  {-# INLINE atan #-}
  sinh = mapTensor sinh
  {-# INLINE sinh #-}
  cosh = mapTensor cosh
  {-# INLINE cosh #-}
  tanh = mapTensor tanh
  {-# INLINE tanh #-}
  asinh = mapTensor asinh
  {-# INLINE asinh #-}
  acosh = mapTensor acosh
  {-# INLINE acosh #-}
  atanh = mapTensor atanh
  {-# INLINE atanh #-}

mapTensor ::
  forall dims a b.
  (KnownDims dims, Elt a, Elt b) =>
  (Exp a -> Exp b) ->
  AccTensor dims a ->
  AccTensor dims b
{-# INLINE mapTensor #-}
mapTensor = coerce $ A.map @(ToShape dims)

zipTensorWith ::
  forall dims a b c.
  (KnownDims dims, Elt a, Elt b, Elt c) =>
  (Exp a -> Exp b -> Exp c) ->
  AccTensor dims a ->
  AccTensor dims b ->
  AccTensor dims c
zipTensorWith = coerce $ A.zipWith @(ToShape dims)

repeatRaw :: forall dims a. (KnownDims dims, Elt a) => a -> RawTensor dims a
repeatRaw = RawTensor . A.fromList (theShape @dims) . repeat

repeatedExp :: forall dims a. (KnownDims dims, Elt a) => Exp a -> AccTensor dims a
{-# INLINE repeatedExp #-}
repeatedExp = AccTensor . A.fill (A.constant $ theShape @dims)

repeatedScalar :: forall dims a. (KnownDims dims, Elt a) => AccTensor '[] a -> AccTensor dims a
{-# INLINE repeatedScalar #-}
repeatedScalar = AccTensor . A.fill (A.constant $ theShape @dims) . A.the . coerce

asScalar :: Elt a => a -> AccTensor '[] a
asScalar = AccTensor . A.unit . A.constant

foldTensor ::
  forall dims a.
  (KnownDims dims, Elt a) =>
  (Exp a -> Exp a -> Exp a) ->
  Exp a ->
  AccTensor dims a ->
  AccTensor '[] a
{-# INLINE foldTensor #-}
foldTensor = coerce $ foldAll @(ToShape dims)

instance (KnownDims dims, A.Num a) => Additive (AccTensor dims a) where
  (^+^) = (+)
  {-# INLINE (^+^) #-}

instance
  (KnownDims dims, A.Num a) =>
  VectorSpace (AccTensor '[] a) (AccTensor dims a)
  where
  (*^) = coerce $ A.zipWith (*) . A.replicate (constant $ theShape @dims)
  {-# INLINE (*^) #-}
  (>.<) = fmap (foldTensor (+) 0) . zipTensorWith (*)
  {-# INLINE (>.<) #-}
  reps = repeatedScalar
  {-# INLINE reps #-}
  sums = foldTensor (+) 0
  {-# INLINE sums #-}
