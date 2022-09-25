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
{-# OPTIONS_GHC -Wno-orphans #-}

module Numeric.Linear.Accelerate.Backprop
  ( module Numeric.Linear.Accelerate,
    (!.!),
    (.*),
    (!*),
    (!*!),
    (><),
    mapTensorBP,
    zipTensorWithBP,
    sumRows,
    sumCols,
    duplicateAsRows,
    duplicateAsCols,
    relu,
    sigmoid,
    transpose,
    softmax,
    sum,
  )
where

import Control.Arrow ((&&&))
import qualified Data.Array.Accelerate as A
import qualified Data.Array.Accelerate.Numeric.LinearAlgebra as ABlas
import qualified Data.Bifunctor as Bi
import Data.Type.Natural
import Numeric.Backprop
import qualified Numeric.Backprop.Num as NumBP
import Numeric.Linear.Accelerate
import Numeric.Linear.Accelerate.Forward (sigmoid)
import qualified Numeric.Linear.Accelerate.Forward as F
import Prelude hiding (sum)

infixr 8 !.!

(!.!) ::
  (KnownDims dims, A.Num a, Reifies s W) =>
  BVar s (AccTensor dims a) ->
  BVar s (AccTensor dims a) ->
  BVar s (AccTensor '[] a)
{-# INLINE (!.!) #-}
(!.!) = liftOp2 $
  op2 $ \x y ->
    (x F.!.! y, \dz -> (dz F..* y, dz F..* x))

infixr 8 .*

(.*) ::
  (KnownDims dims, A.Num a, Reifies s W) =>
  BVar s (AccScalar a) ->
  BVar s (AccTensor dims a) ->
  BVar s (AccTensor dims a)
{-# INLINE (.*) #-}
(.*) = liftOp2 $
  op2 $ \c v ->
    (c F..* v, \dz -> (dz F.!.! v, c F..* dz))

infixl 8 !*, !*!

(!*) ::
  (ABlas.Numeric a, Reifies s W, KnownNat i, KnownNat o) =>
  BVar s (AccMatrix i o a) ->
  BVar s (AccVector i a) ->
  BVar s (AccVector o a)
{-# INLINE (!*) #-}
(!*) = liftOp2 $
  op2 $ \mNM vN ->
    ( mNM F.!* vN
    , \dzdy ->
        ( dzdy F.>< vN
        , F.transpose mNM F.!* dzdy
        )
    )

transpose ::
  (ABlas.Numeric a, Reifies s W, KnownNat n, KnownNat m) =>
  BVar s (AccMatrix n m a) ->
  BVar s (AccMatrix m n a)
{-# INLINE transpose #-}
transpose = liftOp1 $ op1 $ \v -> (F.transpose v, F.transpose)

(!*!) ::
  (ABlas.Numeric a, Reifies s W, KnownNat l, KnownNat o, KnownNat i) =>
  BVar s (AccMatrix l o a) ->
  BVar s (AccMatrix i l a) ->
  BVar s (AccMatrix i o a)
{-# INLINE (!*!) #-}
(!*!) = liftOp2 $
  op2 $ \x y ->
    ( x F.!*! y
    , \d ->
        (d F.!*! F.transpose y, F.transpose x F.!*! d)
    )

infix 7 ><

(><) ::
  (ABlas.Numeric a, KnownNat n, KnownNat m, Reifies s W) =>
  BVar s (AccVector n a) ->
  BVar s (AccVector m a) ->
  BVar s (AccMatrix m n a)
{-# INLINE (><) #-}
(><) = liftOp2 $
  op2 $ \x y ->
    ( x F.>< y
    , \d -> (d F.!* y, F.transpose d F.!* x)
    )

sumRows ::
  (A.Num a, Reifies s W, KnownNat n, KnownNat m) =>
  BVar s (AccMatrix n m a) ->
  BVar s (AccVector m a)
{-# INLINE sumRows #-}
sumRows = liftOp1 $
  op1 $ \mat ->
    (F.sumRows mat, F.duplicateAsCols)

sumCols :: (KnownNat n, A.Num a, Reifies s W, KnownNat m) => BVar s (AccMatrix n m a) -> BVar s (AccVector n a)
{-# INLINE sumCols #-}
sumCols = liftOp1 $ op1 $ \mat -> (F.sumCols mat, F.duplicateAsRows)

duplicateAsRows :: forall m n a s. (KnownNat m, KnownNat n, A.Num a, Reifies s W) => BVar s (AccVector n a) -> BVar s (AccMatrix n m a)
{-# INLINE duplicateAsRows #-}
duplicateAsRows =
  liftOp1 $
    op1 $ \x ->
      ( F.duplicateAsRows @m x
      , F.sumCols
      )

duplicateAsCols :: forall m n a s. (KnownNat m, KnownNat n, A.Num a, Reifies s W) => BVar s (AccVector n a) -> BVar s (AccMatrix m n a)
{-# INLINE duplicateAsCols #-}
duplicateAsCols = liftOp1 $ op1 $ \x -> (F.duplicateAsCols x, F.sumRows)

relu :: (KnownDims ns, A.Num a, A.Ord a, Reifies s W) => BVar s (AccTensor ns a) -> BVar s (AccTensor ns a)
{-# INLINE relu #-}
relu =
  liftOp1 $
    op1 $
      F.relu &&& zipTensorWith (\x d -> (x A.< 0) A.? (0, d))

zipTensorWithBP ::
  (KnownDims dims, A.Num a, Reifies s W) =>
  ( forall s'.
    Reifies s' W =>
    BVar s' (A.Exp a) ->
    BVar s' (A.Exp a) ->
    BVar s' (A.Exp a)
  ) ->
  BVar s (AccTensor dims a) ->
  BVar s (AccTensor dims a) ->
  BVar s (AccTensor dims a)
{-# INLINEABLE zipTensorWithBP #-}
zipTensorWithBP f = liftOp2 $
  op2 $ \a b ->
    let (fxy, dx, dy) = unzipTensor3 $ zipTensorWith (fmap (A.lift . \(x, (y, z)) -> (x, y, z)) . NumBP.backprop2 f) a b
     in (fxy, \d -> (d * dx, d * dy))

mapTensorBP ::
  (KnownDims dims, A.Num a, Reifies s W) =>
  ( forall s'.
    Reifies s' W =>
    BVar s' (A.Exp a) ->
    BVar s' (A.Exp a)
  ) ->
  BVar s (AccTensor dims a) ->
  BVar s (AccTensor dims a)
{-# INLINE mapTensorBP #-}
mapTensorBP f =
  liftOp1 . op1 $
    Bi.second (*)
      . unzipTensor
      . mapTensor (A.lift . NumBP.backprop f)

softmax :: (KnownNat n, KnownNat m, A.Floating a, Reifies s W) => BVar s (AccMatrix n m a) -> BVar s (AccMatrix n m a)
{-# INLINE softmax #-}
softmax us =
  let exps = exp us
   in exps / duplicateAsRows (sumCols exps)

sum :: (KnownDims dims, A.Num a, Reifies s W) => BVar s (AccTensor dims a) -> BVar s (AccScalar a)
{-# INLINE sum #-}
sum = liftOp1 $ op1 $ \x -> (foldTensor (+) 0 x, repeatedScalar)
