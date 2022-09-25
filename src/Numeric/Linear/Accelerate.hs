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
{-# OPTIONS_GHC -fllvm -funbox-strict-fields -Wno-orhpans #-}

module Numeric.Linear.Accelerate
  ( AccTensor,
    AccScalar,
    AccVector,
    AccMatrix,
    dimVal,
    getAccTensor,
    RawTensor,
    getRawTensor,
    runTensor,
    runTensor1,
    repeated,
    generate,
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
  )
where

import Data.Array.Accelerate as A hiding (generate, transpose)
import qualified Data.Array.Accelerate as Acc
import Data.Coerce (coerce)
import Numeric.Linear.Accelerate.Internal
import Prelude as P

default (Int)

type AccScalar = AccTensor '[]

type AccMatrix i o = AccTensor '[i, o]

type AccVector i = AccTensor '[i]

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

-- >>> :t AInt.run1
-- AInt.run1 :: (Arrays a, Arrays b) => (Acc a -> Acc b) -> a -> b

getAccTensor :: AccTensor dims a -> Acc (Array (ToShape dims) a)
{-# INLINE getAccTensor #-}
getAccTensor = coerce

repeated :: forall dims a. (KnownDims dims, Elt a) => a -> AccTensor dims a
{-# INLINE repeated #-}
repeated = AccTensor . A.lift . A.fromList (theShape @dims) . repeat

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
