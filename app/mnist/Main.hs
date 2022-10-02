{-# LANGUAGE ApplicativeDo #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}
{-# OPTIONS_GHC -Wno-orphans #-}

module Main (main) where

import Control.Applicative (optional)
import Control.Arrow ((>>>))
import qualified Control.Foldl as L
import Control.Monad (when, (>=>))
import Control.Monad.IO.Class (MonadIO)
import Control.Monad.Trans.Resource (MonadResource, runResourceT)
import qualified Data.Array.Accelerate as A
import Data.Array.Accelerate.Data.Monoid
import qualified Data.Array.Accelerate.IO.Data.ByteString as ABS
import qualified Data.Array.Accelerate.LLVM.Native as CPU
import qualified Data.Array.Accelerate.LLVM.PTX as GPU
import Data.Attoparsec.ByteString (Parser)
import qualified Data.Attoparsec.ByteString.Streaming as AQ
import qualified Data.Bifunctor as Bi
import qualified Data.ByteString as BS
import Data.Format.MNIST
import Data.Function ((&))
import Data.Functor.Of (Of ((:>)))
import qualified Data.Heap as H
import Data.Proxy (Proxy)
import Data.Strict.Tuple (Pair (..))
import qualified Data.Vector.Fusion.Bundle.Monadic as B
import Data.Word (Word8)
import DeepLearning.Accelerate
import GHC.Generics (Generic)
import GHC.Stack (HasCallStack)
import GHC.TypeNats
import Numeric.Linear.Accelerate
import qualified Options.Applicative as Opts
import RIO (MonadTrans (..), MonadUnliftIO, NFData (..), evaluateDeep, fromMaybe, liftIO, throwIO)
import RIO.FilePath ((</>))
import qualified Streaming as SS
import qualified Streaming.ByteString as Q
import qualified Streaming.Prelude as S
import System.Random (RandomGen (split), getStdGen)
import System.Random.Stateful (AtomicGen (..), RandomGenM, UniformRange (..), newAtomicGenM, thawGen)
import Text.Printf

data TrainOpts = TrainOpts
  { trainPath, testPath :: !FilePath
  , epoch :: !Int
  , batchSize :: !Int
  , outputInterval :: !Int
  , timeStep :: !Float
  , dumping :: !Float
  , digitsDir :: !(Maybe FilePath)
  , backend :: !Backend
  }
  deriving (Show, Eq, Ord, Generic)

data Backend = GPU | CPU
  deriving (Show, Read, Eq, Ord, Generic)

runBackend ::
  Backend ->
  (forall arrays. A.Arrays arrays => A.Acc arrays -> arrays)
runBackend GPU = GPU.run
runBackend CPU = CPU.run

runBackend1 ::
  Backend ->
  (forall arrays arrays'. (A.Arrays arrays, A.Arrays arrays') => (A.Acc arrays -> A.Acc arrays') -> arrays -> arrays')
runBackend1 GPU = GPU.run1
runBackend1 CPU = CPU.run1

main :: IO ()
main = do
  trOpts@TrainOpts {..} <- Opts.execParser trainOptsP
  print trOpts
  gen <- getStdGen
  let (nnGen, g') = split gen
      seed = AtomicGen g'

  nn0 <- randomNN @Pixels @BatchedNet @10 @Float =<< newAtomicGenM nnGen
  !acc <- calcAccuracy backend batchSize testPath nn0
  liftIO $ putStrLn $ printf "Initial accuracy %.10f%%" (acc * 100)
  hdr@ImageFileHeader {..} <-
    evaluateDeep =<< runResourceT (fst <$> loadDataSetStream trainPath)
  let (!numBat0, over) = fromIntegral imageCount `quotRem` batchSize
      numBatches
        | over == 0 = numBat0
        | otherwise = numBat0 + 1
      (!epochGroup, !r) = epoch `quotRem` outputInterval
      !shuffWindow = batchSize * max 1 (numBatches `quot` 100)
      lo
        | r == 0 = B.empty
        | otherwise = B.singleton r
  putStrLn $ printf "Training with %d cases for %d epochs, divided into %d mini-batches, each of size %d" imageCount epoch numBatches batchSize
  B.foldlM'
    ( \(cur :!: net) eps -> do
        let !cur' = cur + eps
        putStrLn $ printf "Epoch %d..%d started." cur cur'
        !net' <-
          B.foldlM'
            ( \curNN () -> do
                g <- thawGen seed
                runResourceT $ do
                  (_, st) <- loadDataSetStream trainPath
                  st
                    & shuffleBuffered' g shuffWindow
                    & chunksOfImages backend hdr batchSize
                    & S.fold_ (flip $ trainMNIST trOpts) curNN id
            )
            net
            (B.replicate eps ())
        putStrLn $ printf "Epoch %d Done." cur'
        acc' <- calcAccuracy backend batchSize testPath net'
        putStrLn $ printf "Test Accuracy at Epoch %d: %.10f%%" cur' (acc' * 100)
        pure $! (cur + eps) :!: net'
    )
    (0 :!: nn0)
    (B.replicate epochGroup outputInterval B.++ lo)
  pure ()

trainMNIST ::
  KnownNetwork 784 ns 10 =>
  TrainOpts ->
  SomeCases 784 10 ->
  NeuralNetwork 784 ns 10 RawTensor Float ->
  NeuralNetwork 784 ns 10 RawTensor Float
trainMNIST TrainOpts {..} (MkSomeCases inps exps _) =
  trainGDWith (runBackend backend) timeStep dumping 1 crossEntropy (inps, exps)

calcAccuracy :: Backend -> Int -> FilePath -> NeuralNetwork 784 BatchedNet 10 RawTensor Float -> IO Double
calcAccuracy mode size fp nn = runResourceT $ do
  (testInfo, st) <- loadDataSetStream fp
  st & chunksOfImages mode testInfo size
    & S.map (toAccuracyStat mode nn)
    & L.purely S.fold_ (L.foldMap id accuracy)

data AccuracyStatistics = AS {count :: !(Sum Int), correct :: !(Sum Int)}
  deriving (Show, Eq, Ord, Generic)
  deriving anyclass (A.Elt)

instance Semigroup AccuracyStatistics where
  AS l r <> AS l' r' = AS (l + l') (r + r')
  {-# INLINE (<>) #-}

instance Monoid AccuracyStatistics where
  mempty = AS 0 0
  {-# INLINE mempty #-}

accuracy :: AccuracyStatistics -> Double
{-# INLINE accuracy #-}
accuracy = (/) <$> fromIntegral . getSum . correct <*> fromIntegral . getSum . count

toAccuracyStat :: Backend -> NeuralNetwork 784 hs 10 RawTensor Float -> SomeCases 784 10 -> AccuracyStatistics
toAccuracyStat mode nn (MkSomeCases inps _ exps) =
  let preds =
        A.map (\(A.T2 (A.I2 i _) _) -> A.fromIntegral i) $
          A.fold1
            (\l@(A.T2 _ w) r@(A.T2 _ w') -> w' A.> w A.? (r, l))
            $ A.indexed $
              getAccTensor $ evalNNA (useTensors nn) (useTensor inps)
   in uncurry AS $
        flip A.indexArray A.Z $
          runBackend mode $
            A.foldAll (<>) mempty $
              A.zipWith
                (\p e -> p A.== e A.? (A.constant (Sum 1, Sum 1), A.constant (Sum 1, Sum 0)))
                preds
                $ A.use $ getRawTensor exps

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

labelsFile, imagesFile :: FilePath
labelsFile = "labels.mnist"
imagesFile = "images.mnist"

type Pixels = 28 * 28

toRawTensor' ::
  forall dims a.
  (HasCallStack, KnownDims dims, Show (A.CoSliceShape (A.CoSliceShape (ToShape dims)))) =>
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

data SomeCases i o where
  MkSomeCases ::
    (KnownNat m) =>
    RawMatrix m i Float ->
    RawMatrix m o Float ->
    RawVector m Word8 ->
    SomeCases i o

instance (KnownNat i, KnownNat o) => NFData (SomeCases i o) where
  rnf (MkSomeCases m m2 ds) = rnf m `seq` rnf m2 `seq` rnf ds

deriving instance (KnownNat i, KnownNat o) => Show (SomeCases i o)

loadDataSetStream ::
  (MonadUnliftIO m, MonadResource m) =>
  FilePath ->
  m (ImageFileHeader, S.Stream (Of (RawImage, Digit)) m ())
loadDataSetStream dir = do
  (imgHeader@ImageFileHeader {..}, imgStream) <-
    parseImageFileA $ Q.readFile (dir </> imagesFile)
  (LabelFileHeader {..}, lblStream) <-
    parseLabelFileS $ Q.readFile (dir </> labelsFile)
  when (imageCount /= labelCount) $
    error $
      "Image and label count unmatched! (imgs, lbls) "
        <> show (imageCount, labelCount)
  pure (imgHeader, S.zip imgStream lblStream)

newtype RawImage = RawImage {getRawImage :: BS.ByteString}

parseImageFileA :: MonadIO m => Q.ByteStream m r -> m (ImageFileHeader, S.Stream (Of RawImage) m r)
parseImageFileA = do
  parseE imageFileHeaderP >&> \(hdr@ImageFileHeader {..}, st) ->
    let pixels = fromIntegral $ columnCount * rowCount
        imgs =
          S.unfoldr
            ( Q.splitAt pixels >>> Q.toStrict >>> fmap \(bs :> st') ->
                Right (RawImage bs, st')
            )
            st
     in (hdr, imgs)

parseE :: MonadIO m => Parser a -> Q.ByteStream m x -> m (a, Q.ByteStream m x)
parseE p =
  AQ.parse p >=> \case
    (Left err, _) -> liftIO $ throwIO $ ParseError err
    (Right hdr, rest) -> pure (hdr, rest)

chunksOfImages ::
  (Monad m, KnownNat pixels) =>
  Backend ->
  ImageFileHeader ->
  Int ->
  S.Stream (Of (RawImage, Digit)) m r ->
  S.Stream (Of (SomeCases pixels 10)) m r
chunksOfImages mode ImageFileHeader {..} chunk =
  let !pixels = fromIntegral $ columnCount * rowCount
   in SS.chunksOf chunk
        >>> S.mapped
          ( S.map (Bi.bimap getRawImage getDigit)
              >>> S.store S.length
              >>> S.unzip
              >>> Q.fromChunks
              >>> Q.toStrict
              >>> Q.pack
              >>> Q.toStrict
              >&> assocOf
              >>> Bi.first \(digits :> images :> sz) ->
                ( runBackend1
                    mode
                    (A.map ((0.5 -) . (/ 255) . A.toFloating @Word8 @Float))
                    (ABS.fromByteStrings (A.Z A.:. pixels A.:. sz) images)
                , ABS.fromByteStrings (A.Z A.:. sz) digits
                )
          )
        >>> S.map (uncurry $ mkCases mode)

mkCases :: (HasCallStack, KnownNat pixels) => Backend -> A.Array A.DIM2 Float -> A.Array A.DIM1 Word8 -> SomeCases pixels 10
mkCases mode images labels =
  let (A.Z A.:. _ A.:. m) = A.arrayShape images
   in case someNatVal (fromIntegral m) of
        SomeNat (_ :: Proxy m) ->
          MkSomeCases @m
            (toRawTensor' "images" images)
            (toRawTensor' "labels" $ toAccelDigitVector mode labels)
            (toRawTensor' "rawDigits" labels)

toAccelDigitVector :: Backend -> A.Array A.DIM1 Word8 -> A.Array A.DIM2 Float
toAccelDigitVector mode =
  runBackend1 mode $
    A.imap (\(A.I2 i _) w8 -> i A.== A.fromIntegral w8 A.? (1.0, 0.0))
      . A.replicate (A.constant $ A.Z A.:. (10 :: Int) A.:. A.All)

infixr 1 >&>

(>&>) :: Functor f => (a -> f b) -> (b -> c) -> a -> f c
{-# INLINE (>&>) #-}
(>&>) = (. fmap) . (>>>)

assocOf :: Of a (Of b1 (Of b2 b3)) -> Of (Of a (Of b1 b2)) b3
assocOf (a :> b :> c :> d) = (a :> b :> c) :> d

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
      backend <-
        Opts.option Opts.auto $
          Opts.long "backend" <> Opts.value GPU <> Opts.showDefault
            <> Opts.help "The backend for the Accelerate"
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
