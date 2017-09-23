module Shuffle (
  shuffle
) where

import Control.Monad (forM, forM_)
import Control.Monad.Random (RandomGen, Rand, getRandomR)
import Data.Array.ST (runSTArray)
import GHC.Arr (thawSTArray, writeSTArray, elems, listArray, readSTArray)

-- copied from Haskell Wiki https://wiki.haskell.org/Random_shuffle

shuffle :: RandomGen g => [a] -> Rand g [a]
shuffle xs = do
    let l = length xs
    rands <- forM [0..(l-2)] $ \i -> getRandomR (i, l-1)
    let ar_ = runSTArray $ do
          ar <- thawSTArray $ listArray (0, l-1) xs
          forM_ (zip [0..] rands) $ \(i, j) -> do
              vi <- readSTArray ar i
              vj <- readSTArray ar j
              writeSTArray ar j vi
              writeSTArray ar i vj
          return ar
    return (elems ar_)
