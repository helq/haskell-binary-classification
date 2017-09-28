{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeFamilies #-}

module Song.Grenade (
  SongSD
, song2SD
, song2Rn
, song2TupleRn
) where

import Song (Song(..))
import Grenade (S(S1D), Shape(..))
import Numeric.LinearAlgebra.Static (R, konst, vector, (#), (&))
import Control.Arrow ((***), first)

type SongSD = (S ('D1 30), S ('D1 1))

song2SD :: (String -> Double) -> Song -> SongSD
song2SD tag = (S1D *** S1D) . song2Rn tag

song2Rn :: (String -> Double) -> Song -> (R 30, R 1)
song2Rn tag song = first (uncurry (#)) $ song2TupleRn tag song

song2TupleRn :: (String -> Double) -> Song -> ((R 3, R 27), R 1)
song2TupleRn tag Song{..} = ((discreteFeats, floatFeats), konst $ tag _genre)
  where
    discreteFeats = konst (fromIntegral _timeSignature)
                  & fromIntegral _key
                  & fromIntegral _mode

    floatFeats = (konst _loudness & _tempo & _duration)
               # (vector _avgTimbre::R 12)
               # (vector _varTimbre::R 12)
