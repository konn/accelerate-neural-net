{-# LANGUAGE ApplicativeDo #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}

module Main (main) where

import Control.Applicative (optional)
import Control.Arrow ((>>>))
import qualified Control.Foldl as L
import Control.Monad (when)
import qualified Data.Array.Accelerate as A
import qualified Data.Array.Accelerate.IO.Data.Vector.Unboxed as VA
import qualified Data.Array.Accelerate.Interpreter as AI
import qualified Data.Array.Accelerate.LLVM.PTX as GPU
import Data.Format.MNIST
import Data.Function ((&))
import Data.Functor.Of (Of)
import qualified Data.Heap as H
import Data.Massiv.Array (MonadUnliftIO)
import qualified Data.Massiv.Array as M
import qualified Data.Massiv.Array as MA
import qualified Data.Massiv.Vector as MV
import qualified Data.Massiv.Vector as VM
import Data.Proxy (Proxy)
import qualified Data.Vector as V
import qualified Data.Vector.Unboxed as U
import DeepLearning.Accelerate
import GHC.Generics (Generic)
import GHC.TypeNats
import Numeric.Linear.Accelerate
import Numeric.Natural (Natural)
import qualified Options.Applicative as Opts
import RIO (MonadTrans (..), NFData (..), PrimMonad, fromMaybe, liftIO)
import RIO.FilePath ((</>))
import qualified Streaming as SS
import qualified Streaming.ByteString as Q
import qualified Streaming.Prelude as S
import System.Random (getStdGen)
import System.Random.Stateful (RandomGenM, UniformRange (..), freezeGen, newAtomicGenM)
import Text.Printf
import UnliftIO.Resource (MonadResource, runResourceT)

data TrainOpts = TrainOpts
  { trainPath, testPath :: !FilePath
  , epoch :: !Int
  , batchSize :: !Int
  , outputInterval :: !Int
  , timeStep :: !Double
  , dumping :: !Double
  , digitsDir :: !(Maybe FilePath)
  }
  deriving (Show, Eq, Ord, Generic)

main :: IO ()
main = do
  trOpts@TrainOpts {..} <- Opts.execParser trainOptsP
  print trOpts
  gen <- getStdGen
  glbGen <- newAtomicGenM gen

  nn0 <- randomNN @Pixels @BatchedNet @10 @Float glbGen
  seed <- freezeGen glbGen
  runResourceT $ do
    (testInfo, st) <- loadDataSetStream testPath
    liftIO $ print testInfo
    acc <-
      st & SS.chunksOf batchSize
        & S.mapped (L.impurely S.foldM $ someCasesL testInfo)
        & S.map (toAccuracyStat nn0)
        & L.purely S.fold_ (L.foldMap id accuracy)
    liftIO $ putStrLn $ printf "Initial accuracy %f%%" (acc * 100)
  pure ()

data AccuracyStatistics = AS {count :: !Int, correct :: !Int}
  deriving (Show, Eq, Ord, Generic)

instance Semigroup AccuracyStatistics where
  AS l r <> AS l' r' = AS (l + l') (r + r')
  {-# INLINE (<>) #-}

instance Monoid AccuracyStatistics where
  mempty = AS 0 0
  {-# INLINE mempty #-}

accuracy :: AccuracyStatistics -> Double
{-# INLINE accuracy #-}
accuracy = (/) <$> fromIntegral . correct <*> fromIntegral . count

toAccuracyStat :: NeuralNetwork 784 hs 10 RawTensor Float -> SomeCases 784 10 -> AccuracyStatistics
toAccuracyStat nn (MkSomeCases inps _ exps) =
  let preds =
        M.map fst $
          M.ifoldlWithin
            M.Dim2
            ( \(M.Ix2 i _) (idx, w) w' ->
                if w < w'
                  then (toEnum i, w')
                  else (idx, w)
            )
            (D0, -1 / 0)
            $ fromRawAccMatrix $ evalNNWith GPU.run1 nn inps
   in M.foldMono (\p -> AS {count = 1, correct = if p then 1 else 0}) $
        M.zipWith (==) preds exps

fromRawAccMatrix :: forall i o a. (KnownNat i, KnownNat o, VA.Unbox a) => RawMatrix i o a -> M.Matrix M.U a
fromRawAccMatrix = M.resize' (M.Sz2 (dimVal @o) (dimVal @i)) . VM.fromUnboxedVector M.Par . VA.toUnboxed . getRawTensor

type BatchedNet =
  '[ L 'Lin 300
   , L 'BatN 300
   , L ( 'Act 'ReLU) 300
   , L 'Lin 50
   , L 'BatN 50
   , L ( 'Act 'ReLU) 50
   , L 'Lin 10
   , L ( 'Act 'Softmax) 10
   ]

type AImage = A.Vector Float

toAImage :: Image -> AImage
toAImage =
  VA.fromUnboxed . M.toUnboxedVector . M.computeP
    . M.map (subtract 0.5 . (/ 255) . realToFrac)

labelsFile, imagesFile :: FilePath
labelsFile = "labels.mnist"
imagesFile = "images.mnist"

batchFold ::
  Monad m =>
  Int ->
  L.FoldM m a b ->
  S.Stream (Of a) m r ->
  S.Stream (Of b) m r
batchFold size fld =
  SS.chunksOf size
    >>> S.mapped (L.impurely S.foldM fld)

type Pixels = 28 * 28

data FoldMap a b where
  FoldMap :: (w -> w -> w) -> !w -> (a -> w) -> (w -> b) -> FoldMap a b

pooledFoldMap :: MonadUnliftIO m => Int -> FoldMap a b -> S.Stream (Of a) m r -> m b
pooledFoldMap size (FoldMap reduce zero mapper extract) =
  SS.chunksOf size
    >>> S.mapped (L.purely S.fold (L.Fold (\x -> reduce x . mapper) zero id))
    >>> L.purely S.fold_ (L.Fold reduce zero extract)

toFold :: FoldMap a b -> L.Fold a b
{-# INLINE toFold #-}
toFold (FoldMap reduce zero to from) = L.Fold (\x -> reduce x . to) zero from

someCasesL ::
  PrimMonad m =>
  ImageFileHeader ->
  L.FoldM m (Image, Digit) (SomeCases Pixels 10)
someCasesL ImageFileHeader {..} =
  mkSomeCase
    <$> L.generalize L.genericLength
    <*> L.premapM (pure . fst) L.vectorM
    <*> L.premapM (pure . snd) L.vectorM
  where
    !pixels = fromIntegral $ rowCount * columnCount
    mkSomeCase :: Natural -> V.Vector Image -> U.Vector Digit -> SomeCases Pixels 10
    mkSomeCase 0 _ ds =
      MkSomeCases @0
        (runTensor AI.run $ repeated 0)
        (runTensor AI.run $ repeated 0)
        (MV.fromUnboxedVector M.Par ds)
    mkSomeCase m ins outs =
      case (someNatVal m, someNatVal $ fromIntegral pixels) of
        (SomeNat (_ :: Proxy m), SomeNat (_ :: Proxy i)) ->
          MkSomeCases @m
            ( toRawTensor' "input" $
                massivToAccel $
                  M.compute $
                    M.concat' 1 $
                      M.fromBoxedVector $
                        V.map (M.resize' (M.Sz2 pixels 1) . M.map ((0.5 -) . (/ 255) . realToFrac)) ins
            )
            ( toRawTensor' "output" $
                massivToAccel $
                  M.computeP $
                    M.concat' 1 $
                      M.map (M.resize' (M.Sz2 10 1) . digitVector) $
                        MV.fromUnboxedVector MV.Par outs
            )
            (M.fromUnboxedVector M.Par outs)

toRawTensor' ::
  forall dims a.
  (KnownDims dims, Show (A.CoSliceShape (A.CoSliceShape (ToShape dims)))) =>
  String ->
  A.Array (ToShape dims) a ->
  RawTensor dims a
toRawTensor' lab arr =
  fromMaybe
    ( error $
        "Dimension mismatched(" <> lab <> "): (exp, got) = "
          <> show (theShape @dims, A.arrayShape arr)
    )
    $ toRawTensor arr

massivToAccel :: M.Matrix M.U Float -> A.Array A.DIM2 Float
massivToAccel mmat =
  GPU.run
    . A.reshape (A.constant $ toMatShape $ M.size mmat)
    . A.use
    . VA.fromUnboxed
    $ M.toUnboxedVector mmat

type ToMassivShape sh = M.Ix sh

type family ToMassivShape' a where
  ToMassivShape' A.Z = 0
  ToMassivShape' (f A.:. Int) = 1 + ToMassivShape' f

toMatShape :: MA.Sz MA.Ix2 -> A.DIM2
toMatShape (M.Sz2 l r) = A.Z A.:. l A.:. r

digitVector :: Digit -> M.Vector M.D Float
digitVector d = M.generate M.Par (M.Sz1 10) $ \(M.Ix1 i) ->
  if i == fromEnum d then 1.0 else 0.0

data SomeCases i o where
  MkSomeCases ::
    (KnownNat m) =>
    RawMatrix m i Float ->
    RawMatrix m o Float ->
    M.Vector M.U Digit ->
    SomeCases i o

instance (KnownNat i, KnownNat o) => NFData (SomeCases i o) where
  rnf (MkSomeCases m m2 ds) = rnf m `seq` rnf m2 `seq` rnf ds

deriving instance (KnownNat i, KnownNat o) => Show (SomeCases i o)

loadDataSetStream ::
  (MonadUnliftIO m, MonadResource m) =>
  FilePath ->
  m (ImageFileHeader, S.Stream (Of (Image, Digit)) m ())
loadDataSetStream dir = do
  (imgHeader@ImageFileHeader {..}, imgStream) <-
    parseImageFileS $ Q.readFile (dir </> imagesFile)
  (LabelFileHeader {..}, lblStream) <-
    parseLabelFileS $ Q.readFile (dir </> labelsFile)
  when (imageCount /= labelCount) $
    error $
      "Image and label count unmatched! (imgs, lbls) "
        <> show (imageCount, labelCount)
  pure (imgHeader, S.zip imgStream lblStream)

trainOptsP :: Opts.ParserInfo TrainOpts
trainOptsP = Opts.info (Opts.helper <*> p) mempty
  where
    p = do
      trainPath <-
        Opts.strOption $
          Opts.long "train" <> Opts.help "The path to the training dataset"
            <> Opts.showDefault
            <> Opts.value
              ("data" </> "mnist" </> "train")
      testPath <-
        Opts.strOption $
          Opts.long "test" <> Opts.help "The path to the training dataset"
            <> Opts.showDefault
            <> Opts.value
              ("data" </> "mnist" </> "test")
      epoch <-
        Opts.option Opts.auto $
          Opts.long "epoch" <> Opts.short 'n'
            <> Opts.value 10
            <> Opts.showDefault
            <> Opts.help "Number of epochs"
      timeStep <-
        Opts.option Opts.auto $
          Opts.long "time-step"
            <> Opts.long "dt"
            <> Opts.long "learning-rate"
            <> Opts.short 'g'
            <> Opts.value 0.1
            <> Opts.metavar "DT"
            <> Opts.showDefault
            <> Opts.help "Time step a.k.a. learning rate"
      dumping <-
        Opts.option Opts.auto $
          Opts.long "alpha"
            <> Opts.long "dumping-factor"
            <> Opts.value 0.1
            <> Opts.help "dumping factor for moving average used in batchnorm layer"
            <> Opts.showDefault
      batchSize <-
        Opts.option Opts.auto $
          Opts.long "batch"
            <> Opts.short 'b'
            <> Opts.value 100
            <> Opts.showDefault
            <> Opts.help "Mini batch size"
      outputInterval <-
        Opts.option Opts.auto $
          Opts.long "interval"
            <> Opts.short 'I'
            <> Opts.value 1
            <> Opts.showDefault
            <> Opts.help "Output interval"
      digitsDir <-
        optional $
          Opts.strOption $
            Opts.long "digits"
              <> Opts.help
                "When specified, guesses the digit in the \
                \given directory at each output timing"
      pure TrainOpts {..}

-- | A variant using priority queue
shuffleBuffered' ::
  (RandomGenM g r m) =>
  g ->
  Int ->
  S.Stream (Of a) m x ->
  S.Stream (Of a) m ()
shuffleBuffered' g n =
  SS.hoist lift
    >>> L.impurely S.foldM_ (shuffleStreamL g n)

shuffleStreamL ::
  (RandomGenM g r m) =>
  g ->
  Int ->
  L.FoldM (S.Stream (Of a) m) a ()
shuffleStreamL g n = L.FoldM step (pure H.empty) (S.map H.payload . S.each)
  where
    step !h !a = do
      w <- lift $ uniformRM (0.0 :: Double, 1.0) g
      let h' = H.insert H.Entry {H.priority = w, H.payload = a} h
          (h'', mout)
            | H.size h' <= n = (h', Nothing)
            | Just (over, rest) <- H.uncons h' = (rest, Just $ H.payload over)
            | otherwise = (h', Nothing)
      S.each mout
      pure h''