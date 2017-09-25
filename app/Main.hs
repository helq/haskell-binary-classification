{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}

module Main where

import           Lens.Micro (over, both)--Lens', set)
import           Lens.Micro.Extras (view)
import qualified Data.Text.IO as Text (readFile)
import           Data.Text (lines)
import           Data.Maybe (mapMaybe)

import           Data.List (foldl')
import           Control.Monad (foldM)
import           Control.Monad.Random (MonadRandom, evalRandIO)

import           Data.Semigroup ((<>))
import           Options.Applicative (Parser, option, auto, optional, long, short, value
                                     , strOption, helper, idm, execParser, info, (<**>))
import qualified Data.ByteString as B
import           Data.Serialize (runPut, put, runGet, get, Get)

import           Numeric.LinearAlgebra.Static (extract, R, ℝ)
import           Numeric.LinearAlgebra.Data (Vector, (!))

import           Grenade (Network, FullyConnected, Tanh, Relu, Logit, Shape(D1)
                         , LearningParameters(..), randomNetwork, train, runNet, S(S1D))

import           Song (Song(..), line2song, genre)
import           Song.Grenade (SongSD, song2SD)
import           Shuffle (shuffle)

type FFNet = Network '[ FullyConnected 30 100, Tanh, FullyConnected 100 1, Logit ]
                     '[ 'D1 30, 'D1 100, 'D1 100, 'D1 1, 'D1 1 ]

randomNet :: MonadRandom m => m FFNet
randomNet = randomNetwork

trainNet :: FFNet -> LearningParameters -> [SongSD] -> [SongSD] -> Int -> IO FFNet
trainNet net0 rate input_data test_data epochs =

    foldForM net0 [1..epochs] $ \net _-> do
      shuffledInput <- evalRandIO (shuffle input_data)
      let newNet = foldl' trainEach net shuffledInput
      netScore newNet input_data test_data
      return newNet

  where
    -- TODO: add training by batches
    trainEach :: FFNet -> SongSD -> FFNet
    trainEach !network (i,o) = train rate network i o
    foldForM zero list operation = foldM operation zero list

netLoad :: FilePath -> IO FFNet
netLoad modelPath = do
  modelData <- B.readFile modelPath
  either fail return $ runGet (get :: Get FFNet) modelData

netError :: FFNet -> [SongSD] -> (Double, Double)
netError net test = (fromIntegral errors / fromIntegral total, distance)
  where
    total, errors :: Integer
    distance :: Double
    (total, errors, distance) = foldl' step (0,0,0) test

    step (t, e, d) song = (t+1, e', d')
      where
        (label, netOut) = valueFromDataAndNet song
        -- the predictions from the network come with numbers between 0 and 1,
        -- everything above .5 is considered 1 and below 0
        e' = if (label > 0.5) == (netOut > 0.5)
                then e
                else e+1
        d' = d + abs (label - netOut)

    valueFromDataAndNet :: SongSD -> (Double, Double)
    valueFromDataAndNet (input, label) = over both sd1toDouble (label, runNet net input)
      where
        sd1toDouble :: S ('D1 1) -> Double
        sd1toDouble (S1D r) = (extract :: R 1 -> Vector ℝ) r ! 0

netScore :: FFNet -> [SongSD] -> [SongSD] -> IO ()
netScore net trainSet test = putStrLn $ "Training error: " <> show (netError net trainSet) <> "\tTesting error: " <> show (netError net test)

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

labelSong :: String -> Double
labelSong "dance and electronica" = 1
labelSong _                       = 0

data ModelsParameters = ModelsParameters Float Int LearningParameters (Maybe FilePath) (Maybe FilePath)

modelsParameters :: Parser ModelsParameters
modelsParameters =
  ModelsParameters <$> option auto (long "test-set" <> short 't' <> value 0.10)
                   <*> option auto (long "epochs"   <> short 'e' <> value 40)
                   <*> (LearningParameters
                       <$> option auto (long "train_rate" <> short 'r' <> value 0.01)
                       <*> option auto (long "momentum" <> value 0.9)
                       <*> option auto (long "l2" <> value 0.0005)
                       )
                   <*> optional (strOption (long "load"))
                   <*> optional (strOption (long "save"))

main :: IO ()
main = do
  ModelsParameters testPerc epochs rate load save <- execParser (info (modelsParameters <**> helper) idm)

  net0 <- case load of
    Just loadFile -> netLoad loadFile
    Nothing       -> randomNet

  -- Loading songs
  songs <- fmap (song2SD labelSong) <$> getSongs filenameDataset
  let n = length songs
  -- Shuffling songs
  --putStrLn "Shuffling songs ..." -- <- this doesn't work by the lazy nature of haskell
  shuffledSongs <- evalRandIO $ shuffle songs
  --print (head shuffledSongs) -- <- this actually works, this is printed only when all the previous steps have been performed
  let (testSet, trainSet) = splitAt (round $ fromIntegral n * testPerc) shuffledSongs

  putStrLn $ "Test Size: " <> show (length testSet)
  netScore net0 trainSet testSet
  -- Training Net
  net <- trainNet net0 rate trainSet testSet epochs
  -- Showing results of training net
  --print net
  netScore net trainSet testSet

  case save of
    Just saveFile -> B.writeFile saveFile $ runPut (put net)
    Nothing       -> return ()
