{-# LANGUAGE DataKinds         #-}
{-# LANGUAGE BangPatterns      #-}
{-# LANGUAGE RankNTypes        #-}
{-# LANGUAGE FlexibleContexts  #-}
{-# LANGUAGE TypeFamilies      #-}

module GrenadeExtras (
  epochTraining,
  binaryNetError,
  normalize
) where

import           Shuffle (shuffle)
import           Control.Monad.Random (MonadRandom)

import           Data.List (foldl')

import           Lens.Micro (over, both)--Lens', set)

import           Data.Singletons (SingI)
import           Data.Singletons.Prelude (Head, Last)

import           Numeric.LinearAlgebra.Static (extract, R, ℝ)
import           Numeric.LinearAlgebra.Data (Vector, (!))

import           GHC.TypeLits (KnownNat)

import           Grenade (Network, Shape(D1), LearningParameters(..), train, runNet, S(S1D))


epochTraining :: (SingI (Last shapes), MonadRandom m) =>
                 Network layers shapes
                 -> LearningParameters
                 -> [(S (Head shapes), S (Last shapes))]
                 -> Int
                 -> m [Network layers shapes]
epochTraining net0 rate input_data epochs =

    foldMeOutList net0 [1..epochs::Int] $ \net _-> do
      shuffledInput <- shuffle input_data
      -- traning net (an epoch) with the input shuffled
      let newNet = foldl' trainEach net shuffledInput
      return newNet

  where
    -- TODO: add training by batches
    trainEach !network (i,o) = train rate network i o
    --foldForM zero list operation = foldM operation zero list

    foldMeOutList :: Monad m => a -> [b] -> (a -> b -> m a) -> m [a]
    foldMeOutList z xs_ op = f' z xs_
      where f' _    []     = return []
            f' zero (x:xs) = do
              zero' <- op zero x
              rec   <- f' zero' xs
              return $ zero':rec


binaryNetError :: (Last shapes ~ 'D1 1, Foldable t) =>
  Network layers shapes -> t (S (Head shapes), S ('D1 1)) -> (Double, Double)

binaryNetError net test = (fromIntegral errors / fromIntegral total, distance)
  where
    total, errors :: Integer
    {-(total, errors) = foldl' step (0,0) test-}
    distance :: Double
    (total, errors, distance) = foldl' step (0,0,0) test

    {-step (t, e) song = (t+1, e')-}
    step (t, e, d) song = (t+1, e', d')
      where
        (label, netOut) = valueFromDataAndNet song
        -- the predictions from the network come with numbers between 0 and 1,
        -- everything above .5 is considered 1 and below 0
        e' = if (label > 0.5) == (netOut > 0.5)
                then e
                else e+1
        d' = d + abs (label - netOut)

--    valueFromDataAndNet :: SongSD -> (Double, Double)
    valueFromDataAndNet (input, label) = over both sd1toDouble (label, runNet net input)
      where
        sd1toDouble :: S ('D1 1) -> Double
        sd1toDouble (S1D r) = (extract :: R 1 -> Vector ℝ) r ! 0

-- taken from https://en.wikipedia.org/wiki/Normalization_(statistics)
-- normalization method: Student's t-statistic
normalize :: KnownNat n => [R n] -> [R n]
normalize features = fmap (\x -> (x-mean)/stdDeviation ) features
  where
    len = length features

    --mean :: R n
    mean = sum features / fromIntegral len

    --stdDeviation :: R n
    stdDeviation = sqrt $ (sum . fmap (\x-> (x-mean)^(2::Int)) $ features)
                          / fromIntegral (len-1)
