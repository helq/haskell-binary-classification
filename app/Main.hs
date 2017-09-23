{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DataKinds #-}

module Main where

{-import           Lens.Micro (Lens')--, set)-}
import           Lens.Micro.Extras (view)
import qualified Data.Text.IO as Text (readFile)
import           Data.Text (lines)
import           Data.Maybe (mapMaybe)
import           Data.List (foldl')
import           Control.Monad (foldM)
import           Control.Monad.Random (MonadRandom, evalRandIO)

import           Grenade (Network, FullyConnected, Tanh, Relu, Logit, Shape(D1)
                         , LearningParameters(..), randomNetwork, train)

import           Song (Song(..), line2song, genre)
import           Song.Grenade (SongSD, song2SD)
import           Shuffle (shuffle)

type FFNet = Network '[ FullyConnected 30 40, Tanh, FullyConnected 40 10, Relu, FullyConnected 10 1, Logit ]
                     '[ 'D1 30, 'D1 40, 'D1 40, 'D1 10, 'D1 10, 'D1 1, 'D1 1]

randomNet :: MonadRandom m => m FFNet
randomNet = randomNetwork

trainNet :: FFNet -> LearningParameters -> [SongSD] -> Int -> IO FFNet
trainNet net0 rate input_data epochs =

    foldForM net0 [1..epochs] $ \net _-> do
      shuffledInput <- evalRandIO (shuffle input_data)
      return . foldl' trainEach net $ shuffledInput

  where
    -- TODO: add training by batches
    trainEach :: FFNet -> SongSD -> FFNet
    trainEach !network (i,o) = train rate network i o
    foldForM zero list operation = foldM operation zero list


-- Reading data and running code

filenameDataset :: String
filenameDataset = "/home/helq/Experimenting/haskell/IHaskell/my_notebooks/ML_class/msd_genre_dataset.txt"

getSongs :: String -> IO [Song]
getSongs filename = do
  --putStrLn "Reading data file ..." -- <- this doesn't work by the lazy nature of haskell
  file <- Text.readFile filename
  let listLines = drop 10 $ Data.Text.lines file
      all_songs = mapMaybe line2song listLines
  return $ filter inRightGenre all_songs

 where inRightGenre song = let g = view genre song
                            in g == "dance and electronica" || g == "jazz and blues"

tagSong :: String -> Double
tagSong "dance and electronica" = 1
tagSong _                       = 0

main :: IO ()
main = do
  -- Loading songs
  songs <- fmap (song2SD tagSong) <$> getSongs filenameDataset
  -- Creating random model
  net0 <- randomNet
  -- Shuffling songs
  --putStrLn "Shuffling songs ..." -- <- this doesn't work by the lazy nature of haskell
  shuffledSongs <- evalRandIO $ shuffle songs
  --print (head shuffledSongs) -- <- this actually works, this is printed only when all the previous steps have been performed
  -- Training Net
  net <- trainNet net0 (LearningParameters 0.01 0.9 0.0005) shuffledSongs 10
  -- Showing results of training net
  print net
