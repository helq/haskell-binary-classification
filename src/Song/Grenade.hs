{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RecordWildCards #-}

module Song.Grenade (
  SongSD
, song2SD
) where

import Song (Song(..))
import Grenade (S(S1D), Shape(..))
import Numeric.LinearAlgebra.Static (R, konst, vector, (#), (&))

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
