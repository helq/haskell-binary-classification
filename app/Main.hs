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
import           Control.Monad (foldM, forM_)
import           Control.Monad.Random (MonadRandom, evalRandIO, evalRand)
import           System.Random (mkStdGen)

import           Data.Semigroup ((<>))
import           Options.Applicative (Parser, option, auto, optional, long, short, value
                                     , strOption, helper, idm, execParser, info, (<**>))
import qualified Data.ByteString as B
import           Data.Serialize (runPut, put, runGet, get, Get)

import           Numeric.LinearAlgebra.Static (extract, R, ℝ)
import           Numeric.LinearAlgebra.Data (Vector, (!))

import           Grenade (Network, FullyConnected, Tanh, Logit, Shape(D1), Relu
                         , LearningParameters(..), randomNetwork, train, runNet, S(S1D))

import           Song (Song(..), line2song, genre)
import           Song.Grenade (SongSD, song2SD, normalize)
import           Shuffle (shuffle)

--type FFNet = Network '[ FullyConnected 30 1, Logit ]
--                     '[ 'D1 30, 'D1 1, 'D1 1 ]
type FFNet = Network '[ FullyConnected 30 40, Tanh, FullyConnected 40 1, Logit ]
                     '[ 'D1 30, 'D1 40, 'D1 40, 'D1 1, 'D1 1 ]
--type FFNet = Network '[ FullyConnected 30 100, Tanh, FullyConnected 100 40, Tanh, FullyConnected 40 1, Logit ]
--                     '[ 'D1 30, 'D1 100, 'D1 100, 'D1 40, 'D1 40, 'D1 1, 'D1 1 ]
--type FFNet = Network '[ FullyConnected 30 100, Tanh, FullyConnected 100 40, Tanh, FullyConnected 40 20, Relu, FullyConnected 20 1, Logit ]
--                     '[ 'D1 30, 'D1 100, 'D1 100, 'D1 40, 'D1 40, 'D1 20, 'D1 20, 'D1 1, 'D1 1 ]

randomNet :: MonadRandom m => m FFNet
randomNet = randomNetwork

trainNet :: FFNet -> LearningParameters -> [SongSD] -> Int -> IO [FFNet]
trainNet net0 rate input_data epochs =

    foldMeOutList net0 [1..epochs] $ \net _-> do
      shuffledInput <- evalRandIO (shuffle input_data)
      -- traning net (an epoch) with the input shuffled
      let newNet = foldl' trainEach net shuffledInput
      return newNet

  where
    -- TODO: add training by batches
    trainEach :: FFNet -> SongSD -> FFNet
    trainEach !network (i,o) = train rate network i o
    --foldForM zero list operation = foldM operation zero list

    foldMeOutList :: Monad m => a -> [b] -> (a -> b -> m a) -> m [a]
    foldMeOutList z xs_ op = f' z xs_
      where f' zero []     = return [zero]
            f' zero (x:xs) = do
              zero' <- op zero x
              rec   <- f' zero' xs
              return $ zero':rec

netLoad :: FilePath -> IO FFNet
netLoad modelPath = do
  modelData <- B.readFile modelPath
  either fail return $ runGet (get :: Get FFNet) modelData

netError :: FFNet -> [SongSD] -> Double
netError net test = fromIntegral errors / fromIntegral total
--netError net test = (fromIntegral errors / fromIntegral total, distance)
  where
    total, errors :: Integer
    (total, errors) = foldl' step (0,0) test
    --distance :: Double
    --(total, errors, distance) = foldl' step (0,0,0) test

    step (t, e) song = (t+1, e')
    --step (t, e, d) song = (t+1, e', d')
      where
        (label, netOut) = valueFromDataAndNet song
        -- the predictions from the network come with numbers between 0 and 1,
        -- everything above .5 is considered 1 and below 0
        e' = if (label > 0.5) == (netOut > 0.5)
                then e
                else e+1
        --d' = d + abs (label - netOut)

    valueFromDataAndNet :: SongSD -> (Double, Double)
    valueFromDataAndNet (input, label) = over both sd1toDouble (label, runNet net input)
      where
        sd1toDouble :: S ('D1 1) -> Double
        sd1toDouble (S1D r) = (extract :: R 1 -> Vector ℝ) r ! 0

netScore :: [SongSD] -> [SongSD] -> FFNet -> IO ()
netScore trainSet test net = putStrLn $ "Training error: " <> show (netError net trainSet) <> "\tTesting error: " <> show (netError net test)

saveScores :: FilePath -> [(Double, Double)] -> IO ()
saveScores logsPath scores = writeFile logsPath (headScores<>"\n"<>scoresStr) -- TODO: catch IO Errors!
  where
    scoresStr :: String
    scoresStr = mconcat $ fmap (\(tr,te)-> show tr <> "\t" <> show te <> "\n") scores

    headScores = "train_error\ttest_error"

-- Reading data and running code

filenameDataset :: String
filenameDataset = "/home/helq/Experimenting/haskell/IHaskell/my_notebooks/ML_class/msd_genre_dataset.txt"

getSongs :: String -> IO [Song]
getSongs filename = do
  --putStrLn "Reading data file ..." -- <- this doesn't work by the lazy nature of haskell
  file <- Text.readFile filename
  --let listLines = drop 10 $ Data.Text.lines file
  let listLines = Data.Text.lines file
      all_songs = mapMaybe line2song listLines
  return $ filter inRightGenre all_songs

 where inRightGenre song = let g = view genre song
                            in g == "dance and electronica" || g == "jazz and blues"

labelSong :: String -> Double
labelSong "dance and electronica" = 1
labelSong _                       = 0

data ModelsParameters =
  ModelsParameters
    Float -- Size of test set
    Int   -- Number of epochs for training
    Bool  -- Normalize data
    LearningParameters -- Parameters for Gradient Descent
    (Maybe FilePath) -- Load path
    (Maybe FilePath) -- Save path
    (Maybe FilePath) -- Logs path

modelsParameters :: Parser ModelsParameters
modelsParameters =
  ModelsParameters <$> option auto (long "test-set" <> short 't' <> value 0.15)
                   <*> option auto (long "epochs"   <> short 'e' <> value 40)
                   <*> option auto (long "normalize" <> value True)
                   <*> (LearningParameters
                       <$> option auto (long "train_rate" <> short 'r' <> value 0.01)
                       <*> option auto (long "momentum" <> value 0.9)
                       <*> option auto (long "l2" <> value 0.0005)
                       )
                   <*> optional (strOption (long "load"))
                   <*> optional (strOption (long "save"))
                   <*> optional (strOption (long "logs"))

main :: IO ()
main = do
  ModelsParameters testPerc epochs norm rate load save logs <- execParser (info (modelsParameters <**> helper) idm)

  net0 <- case load of
    Just loadFile -> netLoad loadFile
    Nothing       -> randomNet

  let normFun = if norm then normalize else id

  -- Loading songs
  songs <- normFun . fmap (song2SD labelSong) <$> getSongs filenameDataset

  let n = length songs
      -- Shuffling songs
      shuffledSongs       = evalRand (shuffle songs) (mkStdGen 487239842)
      (testSet, trainSet) = splitAt (round $ fromIntegral n * testPerc) shuffledSongs

  {-
   -case net0 of
   -  x@(FullyConnected (FullyConnected' i o) _) :~> _ -> do
   -    let subnet = x :~> NNil :: Network '[FullyConnected 30 1] '[ 'D1 30, 'D1 1 ]
   -    print $ runNet subnet (S1D 0) -- with this we get the biases
   -    print i -- biases for each output neuron
   -    print o -- weights between neurons
   -}

  -- Pretraining scores
  --print $ head shuffledSongs
  putStrLn $ "Test Size: " <> show (length testSet)
  --netScore trainSet testSet net0

  -- Training Net
  nets <- trainNet net0 rate trainSet epochs
  let net = last nets
      netsScores = fmap (\n->(netError n trainSet, netError n testSet)) (net0:nets)

  forM_ netsScores $ \(trainError, testError) ->
    putStrLn $ "Training error: " <> show trainError <> "\tTesting error: " <> show testError

  case logs of
    Just logsPath -> saveScores logsPath netsScores
    Nothing       -> return ()

  -- Showing results of training net
  netScore trainSet testSet net

  case save of
    Just saveFile -> B.writeFile saveFile $ runPut (put net)
    Nothing       -> return ()
