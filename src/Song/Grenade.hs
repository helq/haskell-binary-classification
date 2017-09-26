{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeFamilies #-}

module Song.Grenade (
  SongSD
, song2SD
, normalize
) where

import Song (Song(..))
import Grenade (S(S1D), Shape(..))
import Numeric.LinearAlgebra.Static (R, konst, vector, (#), (&))
--import GHC.TypeLits (KnownNat)

type SongSD = (S ('D1 30), S ('D1 1))

song2SD :: (String -> Double) -> Song -> SongSD
song2SD tag Song{..} = (S1D features, S1D . konst $ tag _genre)
  where
    features = ((konst _loudness::R 1)
             & _tempo
             & fromIntegral _timeSignature
             & fromIntegral _key
             & fromIntegral _mode
             & _duration
               )
             # (vector _avgTimbre::R 12)
             # (vector _varTimbre::R 12)

-- taken from https://en.wikipedia.org/wiki/Normalization_(statistics)
-- normalization method: Student's t-statistic
normalize :: [SongSD] -> [SongSD]
normalize songs = fmap (\(S1D x, o) -> (S1D ((x-mean)/stdDeviation), o) ) songs
  where
    n = length songs

    mean :: R 30
    mean = sum (fmap (\(S1D x, _)->x) songs) / fromIntegral n

    stdDeviation :: R 30
    stdDeviation = sqrt $ (sum . fmap (\(S1D x, _)-> (x-mean)^(2::Int)) $ songs)
                          / fromIntegral (n-1)

    --getRn :: KnownNat n => S ('D1 n) -> R n
    --getRn (S1D r) = r
