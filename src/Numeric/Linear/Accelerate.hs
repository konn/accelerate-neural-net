{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# OPTIONS_GHC -fllvm -funbox-strict-fields #-}

module Numeric.Linear.Accelerate
  ( TensorLike,
    AccTensor,
    MatrixOf,
    VectorOf,
    ScalarOf,
    AccScalar,
    AccVector,
    AccMatrix,
    toRawTensor,
    asScalar,
    useTensor,
    dimVal,
    getAccTensor,
    RawTensor,
    RawScalar,
    RawVector,
    RawMatrix,
    getRawTensor,
    runTensor,
    runTensor1,
    rawFromList,
    repeated,
    generate,
    replicateRawA,
    repeatRaw,
    repeatedExp,
    repeatedScalar,
    mapTensor,
    unzipTensor,
    unzipTensor3,
    zipTensorWith,
    foldTensor,
    ToShape,
    SDims (..),
    KnownDims (..),
    theShape,
    (|||),
  )
where

import Control.Monad (guard, replicateM)
import Data.Array.Accelerate as A hiding (generate, transpose)
import qualified Data.Array.Accelerate as Acc
import Data.Coerce (coerce)
import Data.Kind (Type)
import GHC.TypeNats (Nat, type (+))
import Numeric.Linear.Accelerate.Internal
import Prelude as P

default (Int)

type MatrixOf :: ([Nat] -> k) -> Nat -> Nat -> k

type MatrixOf tensor i o = tensor '[o, i]

type VectorOf :: ([Nat] -> k) -> Nat -> k

type VectorOf tensor i = tensor '[i]

type ScalarOf :: ([Nat] -> k) -> k

type ScalarOf tensor = tensor '[]

type AccScalar = ScalarOf AccTensor

type AccMatrix i o = MatrixOf AccTensor i o

type AccVector i = VectorOf AccTensor i

type RawScalar = ScalarOf RawTensor

type RawMatrix i o = MatrixOf RawTensor i o

type RawVector i = VectorOf RawTensor i

getRawTensor :: RawTensor dims a -> Array (ToShape dims) a
{-# INLINE getRawTensor #-}
getRawTensor = coerce

runTensor ::
  (KnownDims dims, Elt a) =>
  (forall x. Arrays x => Acc x -> x) ->
  AccTensor dims a ->
  RawTensor dims a
{-# INLINE runTensor #-}
runTensor runner = RawTensor . runner . getAccTensor

runTensor1 ::
  forall dims dims' a b.
  (KnownDims dims, KnownDims dims', Elt a, Elt b) =>
  (forall x y. (Arrays x, Arrays y) => (Acc x -> Acc y) -> x -> y) ->
  (AccTensor dims a -> AccTensor dims' b) ->
  RawTensor dims a ->
  RawTensor dims' b
{-# INLINE runTensor1 #-}
runTensor1 runner1 f =
  RawTensor . runner1 (getAccTensor . f . AccTensor @dims) . getRawTensor

(|||) :: A.Elt a => AccMatrix i o a -> AccMatrix i' o a -> AccMatrix (i + i') o a
(|||) = coerce (A.++)

-- >>> :t AInt.run1
-- AInt.run1 :: (Arrays a, Arrays b) => (Acc a -> Acc b) -> a -> b

getAccTensor :: AccTensor dims a -> Acc (Array (ToShape dims) a)
{-# INLINE getAccTensor #-}
getAccTensor = coerce

repeated :: forall dims a. (KnownDims dims, Elt a) => a -> AccTensor dims a
{-# INLINE repeated #-}
repeated = AccTensor . A.lift . A.fromList (theShape @dims) . repeat

rawFromList :: forall dims a. (KnownDims dims, Elt a) => [a] -> RawTensor dims a
{-# INLINE rawFromList #-}
rawFromList = RawTensor . A.fromList (theShape @dims)

replicateRawA ::
  forall dims a f.
  (KnownDims dims, Elt a, Applicative f) =>
  f a ->
  f (RawTensor dims a)
{-# INLINE replicateRawA #-}
replicateRawA = fmap rawFromList . replicateM (sizeDims @dims)

generate ::
  forall dims a.
  (KnownDims dims, Elt a) =>
  (Exp (ToShape dims) -> Exp a) ->
  AccTensor dims a
{-# INLINE generate #-}
generate = coerce $ Acc.generate (constant $ theShape @dims)

unzipTensor3 :: (KnownDims dims, Elt a, Elt b, Elt c) => AccTensor dims (a, b, c) -> (AccTensor dims a, AccTensor dims b, AccTensor dims c)
{-# INLINE unzipTensor3 #-}
unzipTensor3 = coerce A.unzip3

unzipTensor :: (KnownDims dims, Elt a, Elt b) => AccTensor dims (a, b) -> (AccTensor dims a, AccTensor dims b)
{-# INLINE unzipTensor #-}
unzipTensor = coerce A.unzip

type TensorLike = [Nat] -> Type -> Type

useTensor :: (KnownDims dim, Elt a) => RawTensor dim a -> AccTensor dim a
useTensor = coerce use

toRawTensor ::
  forall dims a.
  KnownDims dims =>
  Array (ToShape dims) a ->
  Maybe (RawTensor dims a)
toRawTensor arr = do
  guard $ theShape @dims P.== A.arrayShape arr
  pure $ RawTensor arr
