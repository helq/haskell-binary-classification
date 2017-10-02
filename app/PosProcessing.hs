{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import Data.List ((!!), zip4)
import Control.Monad (forM_)
import Data.Semigroup ((<>))

main :: IO ()
main = do
  input <- readFile "kfold-val0.14/final-results.txt"
  let trainCEs    = sumEvery6 $ (read . (!!2) . words) <$> lines input
      validCEs    = sumEvery6 $ (read . (!!4) . words) <$> lines input
      totalEpochs = sumEvery6 $ (read . (!!1) . words) <$> lines input

  putStrLn $ "hidden-layer-size" <> "\t"
          <> "total_epochs" <> "\t"
          <> "training-classification-error" <> "\t"
          <> "validation-classification-error"

  forM_ (zip4 [(1::Int)..] totalEpochs trainCEs validCEs) $ \(sizeHidden, epochs, trainCE, validCE) -> --do
    putStrLn $ show sizeHidden <> "\t"
            <> show epochs <> "\t"
            <> show trainCE <> "\t"
            <> show validCE <> "\t"

sumEvery6 :: [Double] -> [Double]
sumEvery6 [] = []
sumEvery6 xs =
  let (oneKfold, rest) = splitAt 6 xs
   in (sum oneKfold / 6) : sumEvery6 rest
