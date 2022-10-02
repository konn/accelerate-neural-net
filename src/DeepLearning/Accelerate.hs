{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedLabels #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE NoMonomorphismRestriction #-}

module DeepLearning.Accelerate
  ( NeuralNetwork (..),
    Spec (..),
    type L,
    type Skip,
    crossEntropy,
    evalNNWith,
    evalNNA,
    randomNN,
    AffineWeights (..),
    LinearWeights (..),
    BatchNormWeights (..),
    LayerNormWeights (..),
    Pass (..),
    Activation (..),
    SActivation (..),
    sActivation,
    KnownActivation,
    LayerKind (..),
    Weights (..),
    weightsIso,
    auxParamsIso,
    AuxParams (..),
    SLayerKind (..),
    KnownLayerKind,
    sLayerKind,
    HoistTensor (..),
    applyActivation,
    runLayer,
    runNN,
    gradNN,
    trainGDA,
    trainGDWith,
    Network (..),
    zipNetworkWith,
    mapNetwork,
    traverseNetwork,
    LayerKindProxy (..),
    NetworkShape (.., IsOutput, IsCons, IsSkip),
    KnownNetwork,
    networkShape,
    toNetworkShape,
    LossFunction,
  )
where

import Control.Arrow ((>>>))
import Control.Lens hiding (repeated)
import Data.Array.Accelerate (Acc, Arrays)
import qualified Data.Array.Accelerate as A
import qualified Data.Array.Accelerate.Numeric.LinearAlgebra as AB
import qualified Data.Bifunctor as Bi
import Data.Coerce (coerce)
import Data.Generics.Labels ()
import Data.Kind (Type)
import Data.List (iterate')
import Data.Proxy (Proxy (..))
import Data.Reflection (Given (..))
import Data.Tuple (swap)
import Data.Void (Void)
import GHC.Generics (Generic)
import GHC.TypeNats (KnownNat, Nat)
import Numeric.Backprop
import Numeric.Linear.Accelerate.Backprop
import Numeric.Linear.VectorSpace (Additive (..), VectorSpace (..))
import RIO (NFData)
import System.Random.MWC.Distributions (normal)
import System.Random.Stateful (RandomGenM)

type AffineWeights :: Nat -> Nat -> TensorLike -> Type -> Type
data AffineWeights i o t a = AffineWeights
  { scale :: !(MatrixOf t i o a)
  , bias :: !(VectorOf t o a)
  }
  deriving (Generic)

deriving instance
  ( KnownNat i
  , KnownNat o
  , Show (VectorOf t o a)
  , Show (MatrixOf t i o a)
  ) =>
  Show (AffineWeights i o t a)

deriving anyclass instance
  ( KnownNat i
  , KnownNat o
  , NFData (VectorOf t o a)
  , NFData (MatrixOf t i o a)
  ) =>
  NFData (AffineWeights i o t a)

deriving instance
  ( KnownNat i
  , KnownNat o
  , Eq (VectorOf t o a)
  , Eq (MatrixOf t i o a)
  ) =>
  Eq (AffineWeights i o t a)

deriving anyclass instance
  ( KnownNat i
  , KnownNat o
  , A.Num a
  ) =>
  Backprop (AffineWeights i o AccTensor a)

deriving anyclass instance
  (A.Num a, KnownNat i, KnownNat o) =>
  Additive (AffineWeights i o AccTensor a)

deriving anyclass instance
  (A.Num a, A.FromIntegral Int a, KnownNat i, KnownNat o) =>
  VectorSpace (AccScalar a) (AffineWeights i o AccTensor a)

deriving anyclass instance
  (KnownNat i, KnownNat o, A.Elt a) =>
  Arrays (AffineWeights i o RawTensor a)

type LinearWeights :: Nat -> Nat -> TensorLike -> Type -> Type
newtype LinearWeights i o t a = LinearWeights {scale :: MatrixOf t i o a}
  deriving (Generic)

deriving instance
  ( KnownNat i
  , KnownNat o
  , Show (MatrixOf t i o a)
  ) =>
  Show (LinearWeights i o t a)

deriving anyclass instance
  ( KnownNat i
  , KnownNat o
  , NFData (MatrixOf t i o a)
  ) =>
  NFData (LinearWeights i o t a)

deriving instance
  Eq (MatrixOf t i o a) =>
  Eq (LinearWeights i o t a)

deriving anyclass instance
  ( KnownNat i
  , KnownNat o
  , A.Num a
  ) =>
  Backprop (LinearWeights i o AccTensor a)

deriving anyclass instance
  (A.Num a, KnownNat i, KnownNat o) =>
  Additive (LinearWeights i o AccTensor a)

deriving anyclass instance
  (A.Num a, A.FromIntegral Int a, KnownNat i, KnownNat o) =>
  VectorSpace (AccScalar a) (LinearWeights i o AccTensor a)

deriving anyclass instance
  (KnownNat i, KnownNat o, A.Elt a) =>
  Arrays (LinearWeights i o RawTensor a)

type BatchNormWeights :: Nat -> TensorLike -> Type -> Type
data BatchNormWeights o t a = BatchNormWeights {scale, shift :: VectorOf t o a}
  deriving (Generic)

deriving instance
  ( KnownNat o
  , Show (VectorOf t o a)
  ) =>
  Show (BatchNormWeights o t a)

deriving anyclass instance
  ( KnownNat o
  , NFData (VectorOf t o a)
  ) =>
  NFData (BatchNormWeights o t a)

deriving instance
  Eq (VectorOf t o a) =>
  Eq (BatchNormWeights o t a)

deriving anyclass instance
  ( KnownNat o
  , A.Num a
  ) =>
  Backprop (BatchNormWeights o AccTensor a)

deriving anyclass instance
  (A.Num a, KnownNat o) =>
  Additive (BatchNormWeights o AccTensor a)

deriving anyclass instance
  (A.Num a, A.FromIntegral Int a, KnownNat o) =>
  VectorSpace (AccScalar a) (BatchNormWeights o AccTensor a)

deriving anyclass instance
  (KnownNat o, A.Elt a) =>
  Arrays (BatchNormWeights o RawTensor a)

type BatchNormAuxParams :: Nat -> TensorLike -> Type -> Type
data BatchNormAuxParams o t a = BatchNormAuxParams {mean, variance :: VectorOf t o a}
  deriving (Generic)

deriving instance
  ( KnownNat o
  , Show (VectorOf t o a)
  ) =>
  Show (BatchNormAuxParams o t a)

deriving anyclass instance
  ( KnownNat o
  , NFData (VectorOf t o a)
  ) =>
  NFData (BatchNormAuxParams o t a)

deriving instance
  Eq (VectorOf t o a) =>
  Eq (BatchNormAuxParams o t a)

deriving anyclass instance
  ( KnownNat o
  , A.Num a
  ) =>
  Backprop (BatchNormAuxParams o AccTensor a)

deriving anyclass instance
  (A.Num a, KnownNat o) =>
  Additive (BatchNormAuxParams o AccTensor a)

deriving anyclass instance
  (A.Num a, A.FromIntegral Int a, KnownNat o) =>
  VectorSpace (AccScalar a) (BatchNormAuxParams o AccTensor a)

deriving anyclass instance
  (KnownNat o, A.Elt a) =>
  Arrays (BatchNormAuxParams o RawTensor a)

type LayerNormWeights :: Nat -> TensorLike -> Type -> Type
data LayerNormWeights o t a = LayerNormWeights {scale, shift :: VectorOf t o a}
  deriving (Generic)

deriving instance
  ( KnownNat o
  , Show (VectorOf t o a)
  ) =>
  Show (LayerNormWeights o t a)

deriving anyclass instance
  ( KnownNat o
  , NFData (VectorOf t o a)
  ) =>
  NFData (LayerNormWeights o t a)

deriving instance
  Eq (VectorOf t o a) =>
  Eq (LayerNormWeights o t a)

deriving anyclass instance
  ( KnownNat o
  , A.Num a
  ) =>
  Backprop (LayerNormWeights o AccTensor a)

deriving anyclass instance
  (A.Num a, KnownNat o) =>
  Additive (LayerNormWeights o AccTensor a)

deriving anyclass instance
  (A.Num a, A.FromIntegral Int a, KnownNat o) =>
  VectorSpace (AccScalar a) (LayerNormWeights o AccTensor a)

deriving anyclass instance
  (KnownNat o, A.Elt a) =>
  Arrays (LayerNormWeights o RawTensor a)

instance
  (A.Num a, KnownNat i, KnownNat o) =>
  Additive (Weights l i o AccTensor a)
  where
  (AffineW aw) ^+^ (AffineW aw') = AffineW $ aw ^+^ aw'
  (LinearW lw) ^+^ (LinearW lw') = LinearW $ lw ^+^ lw'
  ActivateW ^+^ ActivateW = ActivateW
  (BatchnormW bnw) ^+^ (BatchnormW bnw') = BatchnormW $ bnw ^+^ bnw'
  (LayernormW lnw) ^+^ (LayernormW lnw') = LayernormW $ lnw ^+^ lnw'
  {-# INLINE (^+^) #-}

instance
  (A.Num a, A.FromIntegral Int a, KnownLayerKind l i o) =>
  VectorSpace (AccScalar a) (Weights l i o AccTensor a)
  where
  c *^ (AffineW aw) = AffineW $ c *^ aw
  c *^ (LinearW lw) = LinearW $ c *^ lw
  _ *^ ActivateW = ActivateW
  c *^ (BatchnormW bnw) = BatchnormW $ c *^ bnw
  c *^ (LayernormW lnw) = LayernormW $ c *^ lnw
  {-# INLINE (*^) #-}
  (AffineW aw) >.< (AffineW aw') = aw >.< aw'
  (LinearW lw) >.< (LinearW lw') = lw >.< lw'
  ActivateW >.< ActivateW = 0
  (BatchnormW bnw) >.< (BatchnormW bnw') = bnw >.< bnw'
  (LayernormW lnw) >.< (LayernormW lnw') = lnw >.< lnw'
  {-# INLINE (>.<) #-}
  sums (AffineW aw) = sums aw
  sums (LinearW lw) = sums lw
  sums ActivateW = 0
  sums (BatchnormW bnw) = sums bnw
  sums (LayernormW lnw) = sums lnw
  {-# INLINE sums #-}
  reps = case sLayerKind @l @i @o of
    SAff -> AffineW . reps
    SLin -> LinearW . reps
    SAct {} -> const ActivateW
    SBatN -> BatchnormW . reps
    SLayN -> LayernormW . reps
  {-# INLINE reps #-}

instance
  (forall l x y. KnownLayerKind l x y => Additive (h l x y tensor a)) =>
  Additive (Network h i is o tensor a)
  where
  Output ^+^ Output = Output
  (h :- net) ^+^ (h' :- net') = (h ^+^ h') :- (net ^+^ net')
  (blk ::- net) ^+^ (blk' ::- net') = (blk ^+^ blk') ::- (net ^+^ net')
  {-# INLINE (^+^) #-}

instance
  ( forall l x y. KnownLayerKind l x y => VectorSpace c (h l x y tensor a)
  , forall l x y. KnownLayerKind l x y => Additive (h l x y tensor a)
  , KnownNetwork i is o
  , Num c
  ) =>
  VectorSpace c (Network h i is o tensor a)
  where
  _ *^ Output = Output
  c *^ (h :- net) = (c *^ h) :- (c *^ net)
  c *^ (blk ::- net) = (c *^ blk) ::- (c *^ net)
  {-# INLINE (*^) #-}
  Output >.< Output = 0
  (h :- net) >.< (h' :- net') = h >.< h' + net >.< net'
  (blk ::- net) >.< (blk' ::- net') = blk >.< blk' + net >.< net'
  {-# INLINE (>.<) #-}
  sums = go 0
    where
      go :: c -> Network h x ls y tensor a -> c
      go !acc Output = acc
      go !acc (h :- net') = go (acc + sums h) net'
      go !acc (blk ::- net') = go (go acc blk) net'
  {-# INLINE sums #-}
  reps =
    traverseNetwork
      ( getLayerKind >>> \case
          SAff -> reps
          SLin -> reps
          SAct {} -> reps
          SBatN -> reps
          SLayN -> reps
      )
      (unSkeleton networkShape)
  {-# INLINE reps #-}

data Pass = Train | Eval
  deriving (Show, Eq, Ord, Generic)
  deriving anyclass (NFData)

data Activation = ReLU | Sigmoid | Softmax | NoOp
  deriving (Show, Eq, Ord, Generic)
  deriving anyclass (NFData)

data SActivation (a :: Activation) where
  SReLU :: SActivation 'ReLU
  SSigmoid :: SActivation 'Sigmoid
  SSoftmax :: SActivation 'Softmax
  SNoOp :: SActivation 'NoOp

activationVal :: SActivation a -> Activation
activationVal SReLU = ReLU
activationVal SSigmoid = Sigmoid
activationVal SSoftmax = Softmax
activationVal SNoOp = NoOp

instance Given (SActivation 'ReLU) where
  given = SReLU
  {-# INLINE given #-}

instance Given (SActivation 'Sigmoid) where
  given = SSigmoid
  {-# INLINE given #-}

instance Given (SActivation 'Softmax) where
  given = SSoftmax
  {-# INLINE given #-}

instance Given (SActivation 'NoOp) where
  given = SNoOp
  {-# INLINE given #-}

sActivation :: KnownActivation a => SActivation a
sActivation = given

deriving instance Eq (SActivation a)

deriving instance Ord (SActivation a)

deriving instance Show (SActivation a)

type KnownActivation a = Given (SActivation a)

data LayerKind = Aff | Lin | Act !Activation | BatN | LayN
  deriving (Show, Eq, Ord, Generic)
  deriving anyclass (NFData)

data SLayerKind (lk :: LayerKind) (i :: Nat) (o :: Nat) where
  SAff :: SLayerKind 'Aff i o
  SLin :: SLayerKind 'Lin i o
  SAct :: SActivation act -> SLayerKind ( 'Act act) i i
  SBatN :: SLayerKind 'BatN i i
  SLayN :: SLayerKind 'LayN i i

type KnownLayerKind lk i o = (KnownNat i, KnownNat o, Given (SLayerKind lk i o))

sLayerKind :: KnownLayerKind lk i o => SLayerKind lk i o
{-# INLINE sLayerKind #-}
sLayerKind = given

instance Given (SLayerKind 'Aff i o) where
  given = SAff
  {-# INLINE given #-}

instance Given (SLayerKind 'Lin i o) where
  given = SLin
  {-# INLINE given #-}

instance (Given (SActivation a), i ~ o) => Given (SLayerKind ( 'Act a) i o) where
  given = SAct given
  {-# INLINE given #-}

instance (i ~ o) => Given (SLayerKind 'BatN i o) where
  given = SBatN
  {-# INLINE given #-}

instance (i ~ o) => Given (SLayerKind 'LayN i o) where
  given = SLayN
  {-# INLINE given #-}

type Weights :: LayerKind -> Nat -> Nat -> TensorLike -> Type -> Type
data Weights l i o t a where
  AffineW :: !(AffineWeights i o tensor a) -> Weights 'Aff i o tensor a
  LinearW :: !(LinearWeights i o tensor a) -> Weights 'Lin i o tensor a
  ActivateW :: Weights ( 'Act act) i i tensor a
  BatchnormW ::
    !(BatchNormWeights i tensor a) ->
    Weights 'BatN i i tensor a
  LayernormW ::
    !(LayerNormWeights i tensor a) ->
    Weights 'LayN i i tensor a

deriving instance
  ( KnownNat i
  , KnownNat o
  , Show (MatrixOf t i o a)
  , Show (VectorOf t o a)
  ) =>
  Show (Weights l i o t a)

type AuxParams :: LayerKind -> Nat -> Nat -> TensorLike -> Type -> Type
data AuxParams l i o t a where
  AffineParams :: AuxParams 'Aff i o tensor a
  LinearParams :: AuxParams 'Lin i o tensor a
  ActivateParams :: !(SActivation act) -> AuxParams ( 'Act act) i i tensor a
  BatchnormParams :: !(BatchNormAuxParams i tensor a) -> AuxParams 'BatN i i tensor a
  LayernormParams :: AuxParams 'LayN i i tensor a

deriving instance
  (KnownNat i, KnownNat o, Show (VectorOf t i a)) =>
  Show (AuxParams l i o t a)

instance
  (KnownNat i, KnownNat o, A.Num a) =>
  Backprop (Weights l i o AccTensor a)
  where
  zero (AffineW aw) = AffineW $ zero aw
  zero (LinearW aw) = LinearW $ zero aw
  zero l@ActivateW = l
  zero (BatchnormW bnw) = BatchnormW (zero bnw)
  zero (LayernormW lnw) = LayernormW (zero lnw)
  {-# INLINE zero #-}
  one (AffineW aw) = AffineW $ one aw
  one (LinearW aw) = LinearW $ one aw
  one l@ActivateW = l
  one (BatchnormW bnw) = BatchnormW (one bnw)
  one (LayernormW lnw) = LayernormW (one lnw)
  {-# INLINE one #-}
  add (AffineW aw) (AffineW aw') = AffineW $ add aw aw'
  add (LinearW aw) (LinearW aw') = LinearW $ add aw aw'
  add l@ActivateW ActivateW {} = l
  add (BatchnormW bnw) (BatchnormW bnw') = BatchnormW (add bnw bnw')
  add (LayernormW lnw) (LayernormW lnw') = LayernormW $ add lnw lnw'
  {-# INLINE add #-}

instance
  (KnownNat o, A.Num a) =>
  Backprop (AuxParams l i o AccTensor a)
  where
  zero l@AffineParams = l
  zero l@LinearParams = l
  zero l@(ActivateParams _) = l
  zero (BatchnormParams bnw) = BatchnormParams (zero bnw)
  zero l@LayernormParams = l
  {-# INLINE zero #-}
  one l@AffineParams = l
  one l@LinearParams = l
  one l@(ActivateParams _) = l
  one (BatchnormParams bnw) = BatchnormParams (one bnw)
  one l@LayernormParams = l
  {-# INLINE one #-}
  add l@AffineParams AffineParams {} = l
  add l@LinearParams LinearParams {} = l
  add l@(ActivateParams _) ActivateParams {} = l
  add (BatchnormParams bnw) (BatchnormParams bnw') = BatchnormParams (add bnw bnw')
  add l@LayernormParams LayernormParams {} = l
  {-# INLINE add #-}

type family WeightsOf l i o a t = x | x -> l where
  WeightsOf 'Aff i o a t = AffineWeights i o a t
  WeightsOf 'Lin i o a t = LinearWeights i o a t
  WeightsOf ( 'Act act) i o a t = Proxy act
  WeightsOf 'BatN i i a t = BatchNormWeights i a t
  WeightsOf 'LayN i i a t = LayerNormWeights i a t

type family AuxParamsOf l i o a t = x | x -> l where
  AuxParamsOf 'BatN i i a t = BatchNormAuxParams i a t
  AuxParamsOf ( 'Act act) i i a t = SActivation act
  AuxParamsOf s i o a t = Proxy s

class HoistTensor h where
  runTensors :: A.Elt a => (forall x. Arrays x => Acc x -> x) -> h AccTensor a -> h RawTensor a
  useTensors :: A.Elt a => h RawTensor a -> h AccTensor a

instance (KnownNat i, KnownNat o) => HoistTensor (AffineWeights i o) where
  {-# SPECIALIZE instance
    (KnownNat i, KnownNat o) => HoistTensor (AffineWeights i o)
    #-}
  runTensors f aw =
    AffineWeights
      { scale = runTensor f $ aw ^. #scale
      , bias = runTensor f $ aw ^. #bias
      }
  {-# INLINE runTensors #-}
  useTensors aws =
    AffineWeights
      { scale = useTensor $ aws ^. #scale
      , bias = useTensor $ aws ^. #bias
      }
  {-# INLINE useTensors #-}

instance (KnownNat i, KnownNat o) => HoistTensor (LinearWeights i o) where
  {-# SPECIALIZE instance
    (KnownNat i, KnownNat o) => HoistTensor (LinearWeights i o)
    #-}
  runTensors f aw =
    LinearWeights {scale = runTensor f $ aw ^. #scale}
  {-# INLINE runTensors #-}
  useTensors = coerce useTensor
  {-# INLINE useTensors #-}

instance (KnownNat i) => HoistTensor (BatchNormWeights i) where
  {-# SPECIALIZE instance (KnownNat i) => HoistTensor (BatchNormWeights i) #-}
  runTensors f aw =
    BatchNormWeights
      { scale = runTensor f $ aw ^. #scale
      , shift = runTensor f $ aw ^. #shift
      }
  {-# INLINE runTensors #-}
  useTensors bnw =
    BatchNormWeights
      { scale = useTensor $ bnw ^. #scale
      , shift = useTensor $ bnw ^. #shift
      }
  {-# INLINE useTensors #-}

instance (KnownNat i) => HoistTensor (BatchNormAuxParams i) where
  {-# SPECIALIZE instance (KnownNat i) => HoistTensor (BatchNormAuxParams i) #-}
  runTensors f aw =
    BatchNormAuxParams
      { mean = runTensor f $ aw ^. #mean
      , variance = runTensor f $ aw ^. #variance
      }
  {-# INLINE runTensors #-}
  useTensors aw =
    BatchNormAuxParams
      { mean = useTensor $ aw ^. #mean
      , variance = useTensor $ aw ^. #variance
      }
  {-# INLINE useTensors #-}

instance (KnownNat i) => HoistTensor (LayerNormWeights i) where
  {-# SPECIALIZE instance (KnownNat i) => HoistTensor (LayerNormWeights i) #-}
  runTensors f aw =
    LayerNormWeights
      { scale = runTensor f $ aw ^. #scale
      , shift = runTensor f $ aw ^. #shift
      }
  {-# INLINE runTensors #-}
  useTensors bnw =
    LayerNormWeights
      { scale = useTensor $ bnw ^. #scale
      , shift = useTensor $ bnw ^. #shift
      }
  {-# INLINE useTensors #-}

instance (KnownNat i, KnownNat o) => HoistTensor (Weights l i o) where
  runTensors f (AffineW aw) = AffineW $ runTensors f aw
  runTensors f (LinearW lw) = LinearW $ runTensors f lw
  runTensors _ ActivateW = ActivateW
  runTensors f (BatchnormW bnw) = BatchnormW $ runTensors f bnw
  runTensors f (LayernormW lnw) = LayernormW $ runTensors f lnw
  {-# INLINE runTensors #-}
  useTensors (AffineW aw) = AffineW $ useTensors aw
  useTensors (LinearW lw) = LinearW $ useTensors lw
  useTensors ActivateW = ActivateW
  useTensors (BatchnormW bnw) = BatchnormW $ useTensors bnw
  useTensors (LayernormW lnw) = LayernormW $ useTensors lnw

instance (KnownNat o) => HoistTensor (AuxParams l i o) where
  runTensors _ AffineParams = AffineParams
  runTensors _ LinearParams = LinearParams
  runTensors _ (ActivateParams sa) = ActivateParams sa
  runTensors f (BatchnormParams bnap) = BatchnormParams $ runTensors f bnap
  runTensors _ LayernormParams = LayernormParams
  {-# INLINE runTensors #-}
  useTensors AffineParams = AffineParams
  useTensors LinearParams = LinearParams
  useTensors (ActivateParams sa) = ActivateParams sa
  useTensors (BatchnormParams bnap) = BatchnormParams $ useTensors bnap
  useTensors LayernormParams = LayernormParams

weightsIso :: forall l i o a t. KnownLayerKind l i o => Iso' (Weights l i o t a) (WeightsOf l i o t a)
{-# INLINE weightsIso #-}
weightsIso =
  iso
    ( \case
        (AffineW aw) -> aw
        (LinearW lw) -> lw
        ActivateW -> Proxy
        (BatchnormW bnw) -> bnw
        (LayernormW lnw) -> lnw
    )
    ( case sLayerKind @l @i @o of
        SAff -> AffineW
        SLin -> LinearW
        SAct {} -> const ActivateW
        SBatN -> BatchnormW
        SLayN -> LayernormW
    )

auxParamsIso :: forall l i o a t. KnownLayerKind l i o => Iso' (AuxParams l i o t a) (AuxParamsOf l i o t a)
auxParamsIso =
  iso
    ( \case
        AffineParams -> Proxy
        LinearParams -> Proxy
        (ActivateParams sa) -> sa
        (BatchnormParams bnap) -> bnap
        LayernormParams -> Proxy
    )
    ( case sLayerKind @l @i @o of
        SAff -> const AffineParams
        SLin -> const LinearParams
        SAct {} -> ActivateParams
        SBatN -> BatchnormParams
        SLayN -> const LayernormParams
    )

runLayer ::
  forall m l i o a s.
  ( Reifies s W
  , AB.Numeric a
  , A.Ord a
  , A.Floating a
  , KnownNat m
  , KnownNat i
  , KnownNat o
  , A.FromIntegral Int a
  , A.ToFloating Double a
  ) =>
  Pass ->
  AuxParams l i o AccTensor a ->
  BVar s (Weights l i o AccTensor a) ->
  BVar s (AccMatrix m i a) ->
  BVar s (AccMatrix m o a, AuxParams l i o AccTensor a)
{-# INLINEABLE runLayer #-}
runLayer pass =
  \case
    AffineParams -> \ws inp ->
      let wb = ws ^^. weightsIso
          w = wb ^^. #scale
          b = wb ^^. #bias
       in T2 (w !*! inp + duplicateAsCols b) (auto AffineParams)
    LinearParams -> \ws inp ->
      let wb = ws ^^. weightsIso
          w = wb ^^. #scale
       in T2 (w !*! inp) (auto LinearParams)
    l@(ActivateParams sa) -> \_ inp ->
      T2 (applyActivation sa inp) (auto l)
    l@(BatchnormParams bnps) -> \bnw ->
      let weights = bnw ^^. weightsIso
          mu = auto $ bnps ^. #mean
          sigma2 = auto $ bnps ^. #variance
          gamma = weights ^^. #scale
          beta = weights ^^. #shift
       in case pass of
            Train -> \x ->
              let !m = fromIntegral $ dimVal @m
                  batchMean = sumRows x /. m
                  xRel = x - duplicateAsCols batchMean
                  batchVar = sumRows (xRel * xRel) /. m
                  out = duplicateAsCols gamma * xRel / duplicateAsCols (sqrt $ batchVar + 1e-12) + duplicateAsCols beta
                  bRP =
                    auto l
                      & auxParamsIso . #mean .~~ batchMean
                      & auxParamsIso . #variance .~~ batchVar
               in T2 out bRP
            Eval -> \x ->
              let eps = 1e-12
                  normed =
                    (x - duplicateAsCols mu)
                      / sqrt (duplicateAsCols (sigma2 + eps))
               in T2 normed (auto l)
    l@LayernormParams -> \ps x ->
      let !i = fromIntegral $ dimVal @i
          ws = ps ^^. weightsIso
          mu = sumCols x /. i
          xRel = x - duplicateAsRows mu
          dev = sumCols (xRel * xRel) /. i
          xHat = (x - duplicateAsRows mu) / duplicateAsRows (sqrt (dev + 1e-12))
          x' = duplicateAsCols (ws ^^. #scale) * xHat + duplicateAsCols (ws ^^. #shift)
       in T2 x' (auto l)

applyActivation :: (KnownNat m, KnownNat i, A.Ord a, A.Floating a, Reifies s W, A.ToFloating Double a, A.FromIntegral Int a) => SActivation act -> BVar s (AccMatrix m i a) -> BVar s (AccMatrix m i a)
applyActivation SReLU = relu
applyActivation SSigmoid = sigmoid
applyActivation SSoftmax = softmax
applyActivation SNoOp = id

data Spec = L !LayerKind !Nat | Skip ![Spec]
  deriving (Generic)

type L = 'L

type Skip = 'Skip

type LayerLike = LayerKind -> Nat -> Nat -> TensorLike -> Type -> Type

infixr 9 :-, ::-

newtype LayerKindProxy l n m t a = LayerKindProxy {getLayerKind :: SLayerKind l n m}

data Void2 (a :: k) (b :: k2)

newtype NetworkShape i hs o = MkNetworkShape {unSkeleton :: Network LayerKindProxy i hs o Void2 Void}

toNetworkShape :: Network h i hs o t a -> NetworkShape i hs o
toNetworkShape = MkNetworkShape . go
  where
    go :: Network h i hs o t a -> Network LayerKindProxy i hs o Void2 Void
    go Output = Output
    go (_ :- net') = LayerKindProxy sLayerKind :- go net'
    go (net ::- net') = go net ::- go net'

pattern IsOutput :: () => (hs ~ '[], o ~ i) => NetworkShape i hs o
pattern IsOutput = MkNetworkShape Output

pattern IsCons ::
  () =>
  (hs ~ ( 'L l k : hs1), KnownLayerKind l i k) =>
  SLayerKind l i k ->
  NetworkShape k hs1 o ->
  NetworkShape i hs o
pattern IsCons pk rest <-
  MkNetworkShape (LayerKindProxy pk :- (MkNetworkShape -> rest))
  where
    IsCons pk (MkNetworkShape rest) = MkNetworkShape (LayerKindProxy pk :- rest)

pattern IsSkip ::
  () =>
  (hs ~ ( 'Skip (u : hs') : hs1), KnownNetwork i (u : hs') i) =>
  NetworkShape i (u : hs') i ->
  NetworkShape i hs1 o ->
  NetworkShape i hs o
pattern IsSkip inner rest <-
  MkNetworkShape ((MkNetworkShape -> inner) ::- (MkNetworkShape -> rest))
  where
    IsSkip (MkNetworkShape inner) (MkNetworkShape rest) =
      MkNetworkShape $ inner ::- rest

{-# COMPLETE IsOutput, IsCons, IsSkip :: NetworkShape #-}

type KnownNetwork i hs o =
  ( KnownNat i
  , KnownNetwork' i hs o
  , Given (NetworkShape i hs o)
  , KnownNat o
  )

type family KnownNetwork' i hs o where
  KnownNetwork' i '[] o = i ~ o
  KnownNetwork' i (L l k ': ls) o =
    (KnownLayerKind l i k, KnownNetwork k ls o)
  KnownNetwork' i (Skip (u ': ins) ': ls) o =
    (KnownNetwork i (u ': ins) i, KnownNetwork i ls o)

networkShape :: KnownNetwork i hs o => NetworkShape i hs o
networkShape = given

instance i ~ o => Given (NetworkShape i '[] o) where
  given = IsOutput
  {-# INLINE given #-}

instance
  (Given (NetworkShape k hs o), KnownLayerKind l i k) =>
  Given (NetworkShape i ( 'L l k ': hs) o)
  where
  given = IsCons sLayerKind given
  {-# INLINE given #-}

instance
  (Given (NetworkShape i hs o), KnownNetwork i (u ': hs') i) =>
  Given (NetworkShape i ( 'Skip (u ': hs') ': hs) o)
  where
  given = IsSkip given given
  {-# INLINE given #-}

instance
  (forall l j k. KnownLayerKind l j k => HoistTensor (h l j k)) =>
  HoistTensor (Network h i ls o)
  where
  runTensors _ Output = Output
  runTensors run (h :- net) = runTensors run h :- runTensors run net
  runTensors run (net ::- net') = runTensors run net ::- runTensors run net'
  {-# INLINE runTensors #-}
  useTensors Output = Output
  useTensors (h :- net') = useTensors h :- useTensors net'
  useTensors (net ::- net') = useTensors net ::- useTensors net'
  {-# INLINE useTensors #-}

instance HoistTensor (NeuralNetwork i ls o) where
  runTensors f NeuralNetwork {..} =
    NeuralNetwork
      { weightsNN = runTensors f weightsNN
      , auxParamsNN = runTensors f auxParamsNN
      }
  {-# INLINE runTensors #-}
  useTensors NeuralNetwork {..} =
    NeuralNetwork
      { weightsNN = useTensors weightsNN
      , auxParamsNN = useTensors auxParamsNN
      }
  {-# INLINE useTensors #-}

type Network :: LayerLike -> Nat -> [Spec] -> Nat -> TensorLike -> Type -> Type
data Network h i hs o t a where
  Output :: Network h i '[] i t a
  (:-) ::
    (KnownLayerKind l i k) =>
    !(h l i k t a) ->
    !(Network h k hs o t a) ->
    Network h i ( 'L l k ': hs) o t a
  (::-) ::
    (KnownNetwork i (u ': hs') i) =>
    !(Network h i (u ': hs') i t a) ->
    !(Network h i hs o t a) ->
    Network h i ( 'Skip (u ': hs') ': hs) o t a

deriving instance
  (forall l x y. KnownLayerKind l x y => Show (h l x y t a)) =>
  Show (Network h i hs o t a)

zipNetworkWith ::
  forall h t g u k v i ls o a b c.
  ( forall l x y.
    KnownLayerKind l x y =>
    h l x y t a ->
    g l x y u b ->
    k l x y v c
  ) ->
  Network h i ls o t a ->
  Network g i ls o u b ->
  Network k i ls o v c
{-# INLINE zipNetworkWith #-}
zipNetworkWith f = go
  where
    {-# INLINEABLE go #-}
    go ::
      Network h n hs m t a ->
      Network g n hs m u b ->
      Network k n hs m v c
    go Output Output = Output
    go (h :- net) (g :- net') = f h g :- go net net'
    go (blk ::- net) (blk' ::- net') = go blk blk' ::- go net net'

instance
  ( A.Num a
  , forall l x y. (KnownLayerKind l x y => Backprop (h l x y AccTensor a))
  ) =>
  Backprop (Network h i hs o AccTensor a)
  where
  zero Output = Output
  zero (we :- net') = zero we :- zero net'
  zero (net' ::- net2) = zero net' ::- zero net2
  {-# INLINE zero #-}
  one Output = Output
  one (we :- net') = one we :- one net'
  one (net' ::- net2) = one net' ::- one net2
  {-# INLINE one #-}
  add Output Output = Output
  add (we :- net2) (we' :- net) = add we we' :- add net2 net
  add (net2 ::- net3) (net ::- net4) = add net2 net ::- add net3 net4
  {-# INLINE add #-}

consIso :: (KnownLayerKind l i k) => Iso' (Network h i ( 'L l k ': hs) o t a) (h l i k t a, Network h k hs o t a)
consIso = iso (\(hlikta :- net') -> (hlikta, net')) (uncurry (:-))

skipIso :: (KnownNetwork i (l ': ls) i) => Iso' (Network h i ( 'Skip (l ': ls) ': hs) o t a) (Network h i (l ': ls) i t a, Network h i hs o t a)
skipIso = iso (\(hlikta ::- net') -> (hlikta, net')) (uncurry (::-))

runNN ::
  forall m i hs o a s.
  ( Reifies s W
  , KnownNat m
  , KnownNat i
  , KnownNat o
  , A.Floating a
  , AB.Numeric a
  , A.Ord a
  , A.FromIntegral Int a
  , A.ToFloating Double a
  ) =>
  Pass ->
  Network AuxParams i hs o AccTensor a ->
  BVar s (Network Weights i hs o AccTensor a) ->
  BVar s (AccMatrix m i a) ->
  BVar s (AccMatrix m o a, Network AuxParams i hs o AccTensor a)
runNN pass = go
  where
    go ::
      (KnownNat x, KnownNat y) =>
      Network AuxParams x ls y AccTensor a ->
      BVar s (Network Weights x ls y AccTensor a) ->
      BVar s (AccMatrix m x a) ->
      BVar s (AccMatrix m y a, Network AuxParams x ls y AccTensor a)
    go Output _ = \xs -> T2 xs (auto Output)
    go (p :- net') ws = \xs ->
      let wTl = ws ^^. consIso
          w = wTl ^^. _1
          tl = wTl ^^. _2
          xp' = runLayer pass p w xs
          x' = xp' ^^. _1
          p' = xp' ^^. _2
          ups' = go net' tl x'
       in T2 (ups' ^^. _1) (isoVar2 (:-) (\(h :- t) -> (h, t)) p' $ ups' ^^. _2)
    go (net' ::- net2) ws = \xs ->
      let wTl = ws ^^. skipIso
          w = wTl ^^. _1
          tl = wTl ^^. _2
          xp' = go net' w xs
          x = (xp' ^^. _1) + xs
          p' = xp' ^^. _2
          ups' = go net2 tl x
       in T2
            (ups' ^^. _1)
            (isoVar2 (::-) (\(h ::- t) -> (h, t)) p' (ups' ^^. _2))

type LossFunction m o a =
  forall s. Reifies s W => BVar s (AccMatrix m o a) -> BVar s (AccMatrix m o a) -> BVar s (AccScalar a)

gradNN ::
  (A.Floating a, KnownNat m, KnownNat i, KnownNat o, AB.Numeric a, A.Ord a, A.FromIntegral Int a, A.ToFloating Double a) =>
  LossFunction m o a ->
  (AccMatrix m i a, AccMatrix m o a) ->
  Network AuxParams i ls o AccTensor a ->
  Network Weights i ls o AccTensor a ->
  (Network Weights i ls o AccTensor a, Network AuxParams i ls o AccTensor a)
gradNN loss (inps, oups) recPs =
  Bi.bimap ($ (1, zero recPs)) snd . swap
    . backpropWith
      ( \net ->
          let ysRecs = runNN Train recPs net (auto inps)
           in T2 (loss (ysRecs ^^. _1) (auto oups)) (ysRecs ^^. _2)
      )

data NeuralNetwork i ls o t a = NeuralNetwork
  { weightsNN :: !(Network Weights i ls o t a)
  , auxParamsNN :: !(Network AuxParams i ls o t a)
  }
  deriving (Generic)

deriving instance
  ( forall n. KnownNat n => Show (VectorOf t n a)
  , forall n m. (KnownNat n, KnownNat m) => Show (MatrixOf t n m a)
  ) =>
  Show (NeuralNetwork i ls o t a)

evalNNA ::
  ( KnownNat m
  , KnownNat i
  , KnownNat o
  , AB.Numeric a
  , A.Ord a
  , A.Floating a
  , A.FromIntegral Int a
  , A.ToFloating Double a
  ) =>
  NeuralNetwork i ls o AccTensor a ->
  AccMatrix m i a ->
  AccMatrix m o a
{-# INLINEABLE evalNNA #-}
evalNNA NeuralNetwork {..} =
  evalBP2
    (fmap (viewVar _1) . runNN Eval auxParamsNN)
    weightsNN

evalNNWith ::
  ( KnownNat m
  , KnownNat i
  , KnownNat o
  , AB.Numeric a
  , A.Ord a
  , A.Floating a
  , A.FromIntegral Int a
  , A.ToFloating Double a
  ) =>
  (forall x y. (Arrays x, Arrays y) => (Acc x -> Acc y) -> x -> y) ->
  NeuralNetwork i ls o RawTensor a ->
  RawMatrix m i a ->
  RawMatrix m o a
{-# INLINE evalNNWith #-}
evalNNWith runner = runTensor1 runner . evalNNA . useTensors

trainGDA ::
  ( KnownNat m
  , AB.Numeric a
  , A.Ord a
  , A.Floating a
  , KnownNetwork i ls o
  , A.FromIntegral Int a
  , A.ToFloating Double a
  ) =>
  -- | Learning rate (dt)
  AccScalar a ->
  -- | Dumping factor for batchnorm
  AccScalar a ->
  -- | Iteration
  Int ->
  LossFunction m o a ->
  (AccMatrix m i a, AccMatrix m o a) ->
  NeuralNetwork i ls o AccTensor a ->
  NeuralNetwork i ls o AccTensor a
{-# INLINEABLE trainGDA #-}
trainGDA dt alpha n loss dataSet = last . take (n + 1) . iterate' step
  where
    step NeuralNetwork {..} =
      let (dW, ps') = gradNN loss dataSet auxParamsNN weightsNN
       in NeuralNetwork
            { weightsNN = weightsNN ^+^ dt *^ dW
            , auxParamsNN = zipNetworkWith (updateAuxParams alpha) ps' auxParamsNN
            }

trainGDWith ::
  ( KnownNat m
  , KnownNetwork i ls o
  , AB.Numeric a
  , A.Ord a
  , A.Floating a
  , A.FromIntegral Int a
  , A.ToFloating Double a
  ) =>
  (forall arrays. Arrays arrays => Acc arrays -> arrays) ->
  -- | Learning rate (dt)
  a ->
  -- | Dumping factor for batchnorm
  a ->
  -- | Iteration
  Int ->
  LossFunction m o a ->
  (RawMatrix m i a, RawMatrix m o a) ->
  NeuralNetwork i ls o RawTensor a ->
  NeuralNetwork i ls o RawTensor a
{-# INLINEABLE trainGDWith #-}
trainGDWith runner dt alpha n loss (ins, ous) =
  runTensors runner
    . trainGDA (asScalar dt) (asScalar alpha) n loss (useTensor ins, useTensor ous)
    . useTensors

updateAuxParams ::
  (KnownNat i, A.Num a, A.FromIntegral Int a) =>
  AccScalar a ->
  AuxParams l i o AccTensor a ->
  AuxParams l i o AccTensor a ->
  AuxParams l i o AccTensor a
updateAuxParams _ AffineParams AffineParams = AffineParams
updateAuxParams _ LinearParams LinearParams = LinearParams
updateAuxParams _ l@(ActivateParams _) ActivateParams {} = l
updateAuxParams alpha (BatchnormParams bnap) (BatchnormParams bnap') =
  BatchnormParams $
    alpha *^ bnap ^+^ (1 - alpha) *^ bnap'
updateAuxParams _ LayernormParams LayernormParams = LayernormParams

traverseNetwork ::
  forall h g t t' a b i ls o f.
  Applicative f =>
  ( forall l x y.
    KnownLayerKind l x y =>
    h l x y t a ->
    f (g l x y t' b)
  ) ->
  Network h i ls o t a ->
  f (Network g i ls o t' b)
{-# INLINE traverseNetwork #-}
traverseNetwork f = go
  where
    {-# INLINEABLE go #-}
    go :: Network h x hs y t a -> f (Network g x hs y t' b)
    go Output = pure Output
    go (h :- net') = (:-) <$> f h <*> go net'
    go (blk ::- net') = (::-) <$> go blk <*> go net'

mapNetwork ::
  forall h g t t' a b i ls o.
  ( forall l x y.
    KnownLayerKind l x y =>
    h l x y t a ->
    g l x y t' b
  ) ->
  Network h i ls o t a ->
  Network g i ls o t' b
{-# INLINE mapNetwork #-}
mapNetwork f = go
  where
    {-# INLINEABLE go #-}
    go :: Network h x hs y t a -> Network g x hs y t' b
    go Output = Output
    go (h :- net') = f h :- go net'
    go (blk ::- net') = go blk ::- go net'

traverseNetworkWithTail ::
  forall h g t t' a b i ls o f.
  Applicative f =>
  ( forall l x y l' z.
    (KnownLayerKind l x y, KnownLayerKind l' y z) =>
    h l x y t a ->
    Maybe (h l' y z t a) ->
    f (g l x y t' b)
  ) ->
  Network h i ls o t a ->
  f (Network g i ls o t' b)
{-# INLINE traverseNetworkWithTail #-}
traverseNetworkWithTail f = go
  where
    {-# INLINEABLE go #-}
    go :: forall x hs y. Network h x hs y t a -> f (Network g x hs y t' b)
    go Output = pure Output
    go (h :- Output) = (:- Output) <$> f h (Nothing @(h ( 'Act 'NoOp) y y t a))
    go (h :- (g :- net')) = (:-) <$> f h (Just g) <*> go (g :- net')
    go ((h :: h l x k t a) :- net') =
      (:-)
        <$> f h (Nothing @(h ( 'Act 'NoOp) k k t a))
        <*> go net'
    go (blk ::- net') = (::-) <$> go blk <*> go net'

randomNN ::
  forall i ls o a g m r.
  (RandomGenM g r m, KnownNetwork i ls o, Fractional a, A.Elt a) =>
  g ->
  m (NeuralNetwork i ls o RawTensor a)
randomNN g = do
  let !(MkNetworkShape sh) = networkShape @i @ls @o
  weightsNN <- traverseNetworkWithTail goW sh
  auxParamsNN <- traverseNetwork goP sh
  pure NeuralNetwork {..}
  where
    mayAct :: LayerKindProxy l' y z t w -> Maybe Activation
    mayAct (LayerKindProxy (SAct sa)) = Just (activationVal sa)
    mayAct LayerKindProxy {} = Nothing

    weightFactor = maybe 1.0 (\case ReLU -> 2.0; _ -> 1.0) . (mayAct =<<)

    goW ::
      forall l x y l' z t w.
      (KnownNat x, KnownLayerKind l' y z) =>
      LayerKindProxy l x y t w ->
      Maybe (LayerKindProxy l' y z t w) ->
      m (Weights l x y RawTensor a)
    goW (LayerKindProxy SAff) down = do
      scale <-
        replicateRawA $
          realToFrac
            <$> normal 0.0 (sqrt $ weightFactor down / fromIntegral (dimVal @x)) g
      let bias = repeatRaw 0
      pure $ AffineW AffineWeights {..}
    goW (LayerKindProxy SLin) down = do
      scale <-
        replicateRawA $
          realToFrac
            <$> normal 0.0 (sqrt $ weightFactor down / fromIntegral (dimVal @x)) g
      pure $ LinearW LinearWeights {..}
    goW (LayerKindProxy (SAct _)) _ = pure ActivateW
    goW (LayerKindProxy SBatN) _ =
      pure $ BatchnormW BatchNormWeights {scale = repeatRaw 1, shift = repeatRaw 0}
    goW (LayerKindProxy SLayN) _ =
      pure $
        LayernormW
          LayerNormWeights
            { scale = repeatRaw 1
            , shift = repeatRaw 0
            }
    goP ::
      forall l x y t z.
      (KnownNat y) =>
      LayerKindProxy l x y t z ->
      m (AuxParams l x y RawTensor a)
    goP (LayerKindProxy SAff) = pure AffineParams
    goP (LayerKindProxy SLin) = pure LinearParams
    goP (LayerKindProxy (SAct sa)) = pure $ ActivateParams sa
    goP (LayerKindProxy SBatN) =
      pure $
        BatchnormParams
          BatchNormAuxParams {mean = repeatRaw 0, variance = repeatRaw 1}
    goP (LayerKindProxy SLayN) = pure LayernormParams

crossEntropy :: (KnownNat m, KnownNat o, A.Floating a, A.FromIntegral Int a, A.ToFloating Double a) => LossFunction m o a
{-# INLINE crossEntropy #-}
crossEntropy yPred yTruth = sums $ yTruth * log yPred
