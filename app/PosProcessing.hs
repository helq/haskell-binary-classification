{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import Data.List ((!!))
import Control.Monad (forM_)
import Data.Semigroup ((<>))

main :: IO ()
main = do
  input <- readFile "kfold-val0.14/results.txt"
  let (trainCEs :: [Double]) = (read . (!!4) . words) <$> lines input

  putStrLn "hidden-layer-size\ttraining-classification-error"
  forM_ (zip [(1::Int)..] $ sumEvery6 trainCEs) $ \(sizeHidden, trainCE) -> --do
    putStrLn $ show sizeHidden <> "\t" <> show trainCE

sumEvery6 :: [Double] -> [Double]
sumEvery6 [] = []
sumEvery6 xs =
  let (oneKfold, rest) = splitAt 6 xs
   in (sum oneKfold / 6) : sumEvery6 rest
