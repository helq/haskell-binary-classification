{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DataKinds         #-}
{-# LANGUAGE TypeFamilies      #-}
{-# LANGUAGE RecordWildCards   #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}

module Main where

import           Lens.Micro.Extras (view)
import qualified Data.Text.IO as Text (readFile)
import           Data.Text (lines)
import           Data.Maybe (mapMaybe, fromMaybe)
import           Data.List (zip4) --, foldl')

import           Control.Monad (forM_)
import           Control.Monad.Random (evalRand)
import           System.Random (mkStdGen)
import qualified System.Random as SR (split)

import           Data.Semigroup ((<>))
import           Options.Applicative (Parser, option, auto, optional, long, short, value
                                     , strOption, helper, idm, execParser, info, (<**>))
import qualified Data.ByteString as B
import           Data.Serialize (Serialize(..), runPut, put, runGet, get) --, Get

import           Numeric.LinearAlgebra.Static ((#), (&), R, unwrap, konst) --, zipWithVector)
import           Numeric.LinearAlgebra (toList)

--import           GHC.TypeLits
import           Data.Singletons.TypeLits (KnownNat)

import           Grenade (Network, FullyConnected, Tanh, Logit, Shape(D1), --Relu,
                         LearningParameters(LearningParameters), randomNetwork, S(S1D))

import           GrenadeExtras (binaryNetError, trainOnBatchesEpochs, normalize, hotvector)
--import           GrenadeExtras.Zip (Zip)
--import           GrenadeExtras.Orphan()
--import           GrenadeExtras.GradNorm

import           Song (Song, line2song, genre)
import           Song.Grenade (song2TupleRn) --, SongSD)
import           Shuffle (shuffle)

--import           Debug.Trace (trace)

--type FFNet = Network '[ FullyConnected 75 1, Logit ]
--                     '[ 'D1 75, 'D1 1, 'D1 1 ]
type FFNet n = Network '[ FullyConnected 75 n, Tanh, FullyConnected n 1, Logit ]
                       '[ 'D1 75, 'D1 n, 'D1 n, 'D1 1, 'D1 1 ]
--type FFNet n = Network '[ FullyConnected 30 n, Tanh, FullyConnected n 1, Logit ]
--                     '[ 'D1 30, 'D1 n, 'D1 n, 'D1 1, 'D1 1 ]

--instance KnownNat n => Serialize (FFNet n)

--type FFNetForDiscrete = Network '[FullyConnected 21 10] '[ 'D1 21, 'D1 10 ]
----type FFNetForDiscrete = Network '[FullyConnected 21 14, FullyConnected 14 10] '[ 'D1 21, 'D1 14, 'D1 10 ]
--type FFNet = Network '[ Zip ('D1 21) ('D1 10) FFNetForDiscrete ('D1 54) ('D1 90) (FullyConnected 54 90),
--                        Tanh, FullyConnected 100 35,
--                        Tanh, FullyConnected 35 1,
--                        Logit ]

--                     '[ 'D1 75, 'D1 100, 'D1 100, 'D1 35, 'D1 35, 'D1 1, 'D1 1 ]
--type FFNet = Network '[ FullyConnected 75 300, Tanh, FullyConnected 300 140, Tanh, FullyConnected 140 1, Logit ]
--                     '[ 'D1 75, 'D1 300, 'D1 300, 'D1 140, 'D1 140, 'D1 1, 'D1 1 ]
--type FFNet = Network '[ FullyConnected 75 100, Tanh, FullyConnected 100 40, Tanh, FullyConnected 40 20, Relu, FullyConnected 20 1, Logit ]
--                     '[ 'D1 75, 'D1 100, 'D1 100, 'D1 40, 'D1 40, 'D1 20, 'D1 20, 'D1 1, 'D1 1 ]

netLoad :: KnownNat n => FilePath -> IO (Network '[ FullyConnected 75 n, Tanh, FullyConnected n 1, Logit ]
                                                 '[ 'D1 75, 'D1 n, 'D1 n, 'D1 1, 'D1 1 ])
netLoad modelPath = do
  modelData <- B.readFile modelPath
  either fail return $ runGet get modelData

--createKFoldPartitions :: Int -> [a] -> [([a],[a])]
--createKFoldPartitions size = takeWhile  . splitInSizes
--  where
--    splitInSizes :: [a] -> [[a]]
--    splitInSizes [] = []
--    splitInSizes xs =
--      let (start, finish) = splitAt size xs
--       in start : splitInSizes finish

saveScores :: FilePath -> [NetScore] -> IO ()
saveScores logsPath scores = writeFile logsPath (headScores<>"\n"<>scoresStr) -- TODO: catch IO Errors!
  where
    scoresStr :: String
    scoresStr = mconcat $ fmap netScore2String scores

    netScore2String NetScore{..} = show trainClassError <> "\t"
                                <> show trainingError   <> "\t"
                                <> show testClassError  <> "\t"
                                <> show gradientNorm    <> "\n"

    headScores = "train_classification_error\ttrain_error\ttest_classification_error\tgradient_norm"

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

data StoppingCondition = StoppingCondition { -- stop training if either:
  maxEpochs           :: Int,           -- maximum total epochs has been reached
  gradientSmallerthan :: (Double, Int), -- gradient is smaller than `d` for at least `i` consecutive epochs
  stuckInARange       :: (Double, Int)  -- or, the error hasn't changed more than `d` in the last `i` consecutive epochs
  }

data ModelsParameters =
  ModelsParameters
    Float -- Size of test set
    StoppingCondition
    Int   -- Size of batch
    Bool  -- Normalize data
    LearningParameters -- Parameters for Gradient Descent
    (Maybe FilePath) -- Load path
    (Maybe FilePath) -- Save path
    (Maybe FilePath) -- Logs path

modelsParameters :: Parser ModelsParameters
modelsParameters =
  ModelsParameters <$> option auto (long "test-set"   <> short 't' <> value 0.15)
                   <*> ( StoppingCondition
                         <$> option auto (long "epochs" <> short 'e' <> value 300)
                         <*> option auto (long "gradient-smaller" <> short 'g' <> value (0.09, 2))
                         <*> option auto (long "max-error-change" <> value (0.004, 10))
                       )
                   <*> option auto (long "batch-size" <> short 'b' <> value 40)
                   <*> option auto (long "normalize"  <> value True)
                   <*> (LearningParameters
                       <$> option auto (long "train_rate" <> short 'r' <> value 0.01)
                       <*> option auto (long "momentum" <> value 0.9)
                       <*> option auto (long "l2" <> value 0) -- 0.0005
                       )
                   <*> optional (strOption (long "load"))
                   <*> optional (strOption (long "save"))
                   <*> optional (strOption (long "logs"))

addLog :: KnownNat n => R n -> R n
addLog a = signum a * log (abs a+1)

discreteToOneHotVector :: R 3 -> R 21
discreteToOneHotVector featDiscre = (fromMaybe (konst 0) $ hotvector (round a)::R 8)
                                  # (fromMaybe (konst 0) $ hotvector (round b)::R 12)
                                  & c
  where [a,b,c] = toList $ unwrap featDiscre

takeWhileCond :: ([a] -> [b]) -> (b -> Bool) -> [a] -> [a]
takeWhileCond toCond stopCond xs = fmap snd . takeWhile (stopCond . fst) $ zip (toCond xs) xs

data NetScore = NetScore {
  trainClassError :: Double, -- Classification training error
  trainingError   :: Double, -- (optimization) training error
  testClassError  :: Double, -- Classification testing error
  gradientNorm    :: Double  -- accumulated norm of the gradients on an epoch
} deriving (Show)

netScore trainSet testSet (gradNorm, nn) =
  let (trainCE, trainE) = binaryNetError nn trainSet
      (testCE, _)       = binaryNetError nn testSet
   in
      NetScore { trainClassError = trainCE,
                 trainingError   = trainE,
                 testClassError  = testCE,
                 gradientNorm    = gradNorm }

takeWhileCondFunc :: Show a => StoppingCondition -> [(a, NetScore)] -> [(a, NetScore)]
takeWhileCondFunc (StoppingCondition epochs (grad, giters) (stuck, siters)) =
 -- trace ("hi" <> show (head (takeWhileCond maxErrorChangeInSIters (<stuck) xs))) .
  take (epochs+1) . takeWhileCond consecutiveSmallerThanGrad (<giters)
                  . takeWhileCond maxErrorChangeInSIters (>stuck)
  where
    consecutiveSmallerThanGrad :: [(a, NetScore)] -> [Int]
    consecutiveSmallerThanGrad = (0:) . countConsecutive (0::Int) . fmap (trainClassError . snd)
      where
        countConsecutive _    []     = []
        countConsecutive cond (x:xs) =
          let condVal = if x<grad then cond+1 else 0
           in condVal : countConsecutive condVal xs

    maxErrorChangeInSIters :: [(a, NetScore)] -> [Double]
    maxErrorChangeInSIters = (replicate siters (stuck+1) <>) . fmap findMaxChange . groupOnSIters . fmap (trainClassError . snd)
      where groupOnSIters :: [a] -> [[a]]
            groupOnSIters []         = []
            groupOnSIters xs@(_:xs') = take siters xs : groupOnSIters xs'
            findMaxChange :: [Double] -> Double
            findMaxChange xs = maximum xs - minimum xs

main :: IO ()
main = do
  ModelsParameters testPerc stopCond batchSize norm rate load save logs <- execParser (info (modelsParameters <**> helper) idm)

  let (seedNet, (seedShuffle, seedTraining)) = SR.split <$> SR.split (mkStdGen 487239842)

  (net0 :: FFNet 2) <- case load of
     Just loadFile -> netLoad loadFile
     Nothing       -> return $ evalRand randomNetwork seedNet

  let normFun :: KnownNat n => [R n] -> [R n]
      normFun = if norm then normalize else id

  -- Loading songs
  songsRaw <- fmap (song2TupleRn labelSong) <$> getSongs filenameDataset
  let songsDiscrete :: [R 3]
      songsDiscrete = fmap (fst . fst) songsRaw
      songsDisOneHotVector :: [R 21]
      songsDisOneHotVector = fmap discreteToOneHotVector songsDiscrete
      --songsFloat    :: [R 27]
      songsFloat    = fmap (snd . fst) songsRaw
      --songsLog      :: [R 27]
      songsLog      = fmap addLog songsFloat
      --songsLabel    :: [R 1]
      songsLabel    = fmap snd songsRaw
      --songs :: [(S ('D1 57), S ('D1 1))]
      songs = (\(l1,l2,l3,o)->(S1D (l1#l2#l3), S1D o)) <$> zip4 songsDisOneHotVector (normFun songsFloat) (normFun songsLog) songsLabel

  --print . toList . unwrap $ foldl1' (zipWithVector max) songsDiscrete
  --print . toList . unwrap $ foldl1' (zipWithVector min) songsDiscrete

  let n = length songsRaw
      -- Shuffling songs
      shuffledSongs       = evalRand (shuffle songs) seedShuffle
      (testSet, trainSet) = splitAt (round $ fromIntegral n * testPerc) shuffledSongs

  {-
   -case net0 of
   -  x@(FullyConnected (FullyConnected' i o) _) :~> _ -> do
   -    let subnet = x :~> NNil :: Network '[FullyConnected 75 1] '[ 'D1 75, 'D1 1 ]
   -    --print $ runNet subnet (S1D 0) -- with this we get the biases
   -    --print . sqrt . normSquared $ net0
   -    print i -- biases for each output neuron
   -    print o -- weights between neurons
   -}

  -- Pretraining scores
  --print $ head shuffledSongs
  putStrLn $ "Test Size: " <> show (length testSet)
  putStrLn $ "Batch Size: " <> show batchSize

  -- Training Net
  let netsInf = (read "Infinity", net0) : evalRand (trainOnBatchesEpochs net0 rate trainSet batchSize) seedTraining
      netsScoresInf = fmap (netScore trainSet testSet) netsInf

      (nets, netsScores) = unzip . takeWhileCondFunc stopCond $ zip (fmap snd netsInf) netsScoresInf
      net = last nets

  putStrLn $ "Epoch"
           <> "\tTraining classification error"
           <> "\tTraining error"
           <> "\tGradNorm"
           <> "\tTesting error"
  -- Showing results of training net
  forM_ (zip [(0::Integer)..] netsScores) $ \(epoch, NetScore trainCE trainE testE gradNorm) ->
    putStrLn $ show epoch
             <> "\t" <> show trainCE
             <> "\t" <> show trainE
             <> "\t" <> show gradNorm
             <> "\t" <> show testE

  case logs of
    Just logsPath -> saveScores logsPath netsScores
    Nothing       -> return ()

  case save of
    Just saveFile -> B.writeFile saveFile $ runPut (put net)
    Nothing       -> return ()
