{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# OPTIONS_GHC -fllvm #-}

module Numeric.Linear.Accelerate.Forward
  ( (!.!),
    (.*),
    (/.),
    (!*),
    (!*!),
    (><),
    sumRows,
    sumCols,
    duplicateAsCols,
    duplicateAsRows,
    relu,
    sigmoid,
    transpose,
    softmax,
  )
where

import Control.Lens (Profunctor (..))
import Data.Array.Accelerate hiding (transpose)
import qualified Data.Array.Accelerate as A
import qualified Data.Array.Accelerate.Numeric.LinearAlgebra as ABlas
import Data.Coerce (coerce)
import GHC.TypeNats (KnownNat)
import Numeric.Linear.Accelerate
import Numeric.Linear.Accelerate.Internal
import Prelude as P

infixr 8 !.!

(!.!) :: (KnownDims dims, A.Num a) => AccTensor dims a -> AccTensor dims a -> AccTensor '[] a
{-# INLINE (!.!) #-}
(!.!) = fmap (foldTensor (+) 0) . zipTensorWith (*)

infixr 8 .*

infixl 8 /.

infixl 8 !*, !*!

(!*) :: (ABlas.Numeric a) => AccMatrix i o a -> AccVector i a -> AccVector o a
{-# INLINE (!*) #-}
(!*) = coerce (ABlas.#>)

(!*!) :: (ABlas.Numeric a) => AccMatrix l o a -> AccMatrix i l a -> AccMatrix i o a
{-# INLINE (!*!) #-}
(!*!) = coerce (ABlas.<>)

(.*) ::
  forall dims a.
  (KnownDims dims, A.Num a) =>
  AccScalar a ->
  AccTensor dims a ->
  AccTensor dims a
{-# INLINE (.*) #-}
(.*) = coerce $ A.zipWith (*) . A.replicate (constant $ theShape @dims)

(/.) ::
  forall dims a.
  (KnownDims dims, A.Fractional a) =>
  AccTensor dims a ->
  AccScalar a ->
  AccTensor dims a
{-# INLINE (/.) #-}
(/.) = coerce $ lmap (A.replicate (constant $ theShape @dims)) . A.zipWith (/)

sumRows :: (A.Num a) => AccMatrix n m a -> AccVector m a
{-# INLINE sumRows #-}
sumRows = coerce $ A.fold (+) 0

sumCols :: (A.Num a) => AccMatrix n m a -> AccVector n a
{-# INLINE sumCols #-}
sumCols = coerce $ A.fold (+) 0 . A.transpose

duplicateAsRows :: forall m n a. (KnownNat m, Elt a) => AccVector n a -> AccMatrix n m a
{-# INLINE duplicateAsRows #-}
duplicateAsRows = coerce $ A.replicate (constant $ Z :. dimVal @m :. All)

duplicateAsCols :: forall m n a. (KnownNat m, Elt a) => AccVector n a -> AccMatrix m n a
{-# INLINE duplicateAsCols #-}
duplicateAsCols = coerce $ A.replicate (constant $ Z :. All :. dimVal @m)

relu :: (KnownDims ns, A.Num a, A.Ord a) => AccTensor ns a -> AccTensor ns a
{-# INLINE relu #-}
relu = mapTensor $ A.max 0

sigmoid :: P.Floating a => a -> a
{-# INLINE sigmoid #-}
sigmoid = recip . (1 +) . exp . negate

transpose :: forall n m a. (Elt a) => AccMatrix n m a -> AccMatrix m n a
{-# INLINE transpose #-}
transpose = coerce A.transpose

infix 7 ><

(><) :: ABlas.Numeric a => AccVector n a -> AccVector m a -> AccMatrix m n a
{-# INLINE (><) #-}
(><) = coerce (ABlas.><)

softmax :: (KnownNat n, KnownNat m, A.Floating a) => AccMatrix n m a -> AccMatrix n m a
{-# INLINE softmax #-}
softmax us =
  let exps = exp us
   in exps / duplicateAsRows (sumCols exps)
