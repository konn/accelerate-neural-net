{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Main (main) where

import Data.Array.Accelerate hiding (generate)
import Data.Array.Accelerate.LLVM.PTX
import Numeric.Linear.Accelerate
import Numeric.Linear.Accelerate.Forward
import Prelude hiding (fromIntegral)

main :: IO ()
main = do
  let mat23 :: AccMatrix 256 128 Float
      mat23 = generate $ \(I2 i j) -> fromIntegral i * 128 + fromIntegral j
      mat35 :: AccMatrix 128 10240 Float
      mat35 = generate $ \(I2 i j) -> fromIntegral i * 64 + fromIntegral j
      vec :: AccVector 256 Float
      vec = generate $ \(I1 i) -> fromIntegral i
  print $ runTensor run mat23
  print $ runTensor run vec
  print $ runTensor run $ mat23 !* vec
  print $ runTensor run $ mat35 !*! mat23 !* vec
  print $ runTensor run mat35
  print $ runTensor run $ mat35 !*! mat23
  print $ runTensor run $ sumRows $ mat35 !*! mat23
  print $ runTensor run $ sumCols $ mat35 !*! mat23
  print $ runTensor run $ duplicateAsRows @3 $ 2 .* vec
  print $ runTensor run $ duplicateAsCols @3 $ 2 .* vec
