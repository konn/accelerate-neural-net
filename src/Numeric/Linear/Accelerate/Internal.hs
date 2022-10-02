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
    theShape,
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
  SNil :: SDims '[]
  (:%) :: SNat n -> SDims ns -> SDims (n ': ns)

infixr 9 :%

deriving instance Show (SDims ns)

deriving instance P.Eq (SDims ns)

deriving instance P.Ord (SDims ns)

newtype RawTensor dims a = RawTensor (Array (ToShape dims) a)

deriving newtype instance (KnownDims dims, A.Elt a) => NFData (RawTensor dims a)

deriving newtype instance
  (KnownDims dims, A.Elt a) => Arrays (RawTensor dims a)

data SShape a where
  SZ :: SShape Z
  (:%.) :: SShape acc -> Int -> SShape (acc :. Int)

infixl 3 :%.

fromSShape :: SShape a -> a
fromSShape SZ = Z
fromSShape (acc :%. n) = fromSShape acc :. n

sToShape' :: SShape sh -> SDims dims -> SShape (ToShape' sh dims)
sToShape' acc SNil = acc
sToShape' acc (n :% ns) = sToShape' (acc :%. P.fromIntegral (toNatural n)) ns

theShape :: forall dims. KnownDims dims => ToShape dims
theShape = fromSShape $ sToShape' SZ (sDims @dims)

type ToShape :: [Nat] -> Type

type ToShape dims = ToShape' Z dims

type ToShape' :: Type -> [Nat] -> Type
type family ToShape' acc dims where
  ToShape' acc '[] = acc
  ToShape' acc (n ': ns) = ToShape' (acc :. Int) ns

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

sizeDims :: forall dims. KnownDims dims => Int
sizeDims = go $ sDims @dims
  where
    go :: SDims ds -> Int
    go SNil = 1
    go (sn :% sd') = P.fromIntegral (toNatural sn) * go sd'

instance KnownDims '[] where
  sDims = SNil
  {-# INLINE sDims #-}

instance
  ( KnownNat n
  , KnownDims ns
  , A.Elt (A.Plain (ToShape (n ': ns)))
  , Shape (ToShape (n ': ns))
  , Slice (ToShape (n ': ns))
  , Plain (ToShape (n ': ns)) ~ ToShape (n ': ns)
  , P.Eq (A.CoSliceShape (A.CoSliceShape (ToShape (n ': ns))))
  ) =>
  KnownDims (n ': ns)
  where
  sDims = sNat :% sDims
  {-# INLINE sDims #-}

type KnownDims' :: [Nat] -> Constraint
type family KnownDims' ns where
  KnownDims' '[] = ()
  KnownDims' (n ': ns) = (KnownNat n, KnownDims ns)

instance (KnownDims dims, A.Num a) => Backprop (AccTensor dims a) where
  zero = const $ repeatedExp 0
  {-# INLINE zero #-}
  one = const $ repeatedExp 1
  {-# INLINE one #-}
  add = zipTensorWith (+)
  {-# INLINE add #-}

instance (A.FromIntegral Int a, A.Num a, KnownDims dims) => P.Num (AccTensor dims a) where
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
  fromInteger = repeatedExp . A.fromIntegral . A.constant . P.fromInteger @Int
  {-# INLINE fromInteger #-}
  negate = mapTensor negate
  {-# INLINE negate #-}

instance (A.FromIntegral Int a, A.ToFloating Double a, A.Floating a, KnownDims dims) => P.Fractional (AccTensor dims a) where
  fromRational = repeatedExp . A.toFloating . A.constant . P.fromRational @Double
  {-# INLINE fromRational #-}
  (/) = zipTensorWith (/)
  {-# INLINE (/) #-}
  recip = mapTensor recip
  {-# INLINE recip #-}

instance
  ( A.ToFloating Double a
  , A.FromIntegral Int a
  , A.Floating a
  , KnownDims dims
  ) =>
  P.Floating (AccTensor dims a)
  where
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
  (^+^) = add
  {-# INLINE (^+^) #-}

instance
  (KnownDims dims, A.Num a, FromIntegral Int a) =>
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
