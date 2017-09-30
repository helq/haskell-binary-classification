{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DataKinds         #-}
{-# LANGUAGE TypeFamilies      #-}

module Main where

import           Lens.Micro.Extras (view)
import qualified Data.Text.IO as Text (readFile)
import           Data.Text (lines)
import           Data.Maybe (mapMaybe, fromMaybe)
import           Data.List (zip4) --, foldl1')

import           Control.Arrow ((***))

import           Control.Monad (forM_)
import           Control.Monad.Random (evalRand)
import           System.Random (mkStdGen)
import qualified System.Random as SR (split)

import           Data.Semigroup ((<>))
import           Options.Applicative (Parser, option, auto, optional, long, short, value
                                     , strOption, helper, idm, execParser, info, (<**>))
import qualified Data.ByteString as B
import           Data.Serialize (runPut, put, runGet, get) --, Get

import           Numeric.LinearAlgebra.Static ((#), (&), R, unwrap, konst) --, zipWithVector)
import           Numeric.LinearAlgebra (toList)

--import           GHC.TypeLits
import           Data.Singletons.TypeLits (KnownNat)

import           Grenade (Network, FullyConnected, Tanh, Logit, Shape(D1), -- Relu,
                         LearningParameters(LearningParameters), randomNetwork, S(S1D))

import           GrenadeExtras (binaryNetError, epochTraining, trainOnBatchesEpochs, normalize, hotvector)
import           GrenadeExtras.Zip (Zip)
import           GrenadeExtras.Orphan()

import           Song (Song, line2song, genre)
import           Song.Grenade (song2TupleRn) --, SongSD)
import           Shuffle (shuffle)

--type FFNet = Network '[ FullyConnected 75 1, Logit ]
--                     '[ 'D1 75, 'D1 1, 'D1 1 ]
type FFNet = Network '[ FullyConnected 75 16, Tanh, FullyConnected 16 1, Logit ]
                     '[ 'D1 75, 'D1 16, 'D1 16, 'D1 1, 'D1 1 ]
--type FFNet n = Network '[ FullyConnected 30 n, Tanh, FullyConnected n 1, Logit ]
--                     '[ 'D1 30, 'D1 n, 'D1 n, 'D1 1, 'D1 1 ]

--type FFNetForDiscrete = Network '[FullyConnected 21 10] '[ 'D1 21, 'D1 10 ]
----type FFNetForDiscrete = Network '[FullyConnected 21 14, FullyConnected 14 10] '[ 'D1 21, 'D1 14, 'D1 10 ]
--type FFNet = Network '[ Zip ('D1 21) ('D1 10) FFNetForDiscrete ('D1 54) ('D1 90) (FullyConnected 54 90),
--                        Tanh, FullyConnected 100 35,
--                        Tanh, FullyConnected 35 1,
--                        Logit ]

--                     '[ 'D1 75, 'D1 100, 'D1 100, 'D1 35, 'D1 35, 'D1 1, 'D1 1 ]
--type FFNet = Network '[ FullyConnected 75 300, Tanh, FullyConnected 300 140, Tanh, FullyConnected 140 1, Logit ]
--                     '[ 'D1 75, 'D1 300, 'D1 300, 'D1 140, 'D1 140, 'D1 1, 'D1 1 ]
--type FFNet = Network '[ FullyConnected 60 100, Tanh, FullyConnected 100 40, Tanh, FullyConnected 40 20, Relu, FullyConnected 20 1, Logit ]
--                     '[ 'D1 60, 'D1 100, 'D1 100, 'D1 40, 'D1 40, 'D1 20, 'D1 20, 'D1 1, 'D1 1 ]

--netLoad :: KnownNat n => FilePath -> IO (FFNet n)
netLoad :: FilePath -> IO FFNet
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
    Int   -- Size of batch
    Bool  -- Normalize data
    LearningParameters -- Parameters for Gradient Descent
    (Maybe FilePath) -- Load path
    (Maybe FilePath) -- Save path
    (Maybe FilePath) -- Logs path

modelsParameters :: Parser ModelsParameters
modelsParameters =
  ModelsParameters <$> option auto (long "test-set"   <> short 't' <> value 0.15)
                   <*> option auto (long "epochs"     <> short 'e' <> value 40)
                   <*> option auto (long "batch-size" <> short 'b' <> value 40)
                   <*> option auto (long "normalize"  <> value True)
                   <*> (LearningParameters
                       <$> option auto (long "train_rate" <> short 'r' <> value 0.01)
                       <*> option auto (long "momentum" <> value 0.9)
                       <*> option auto (long "l2" <> value 0.0005)
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

main :: IO ()
main = do
  ModelsParameters testPerc epochs batchSize norm rate load save logs <- execParser (info (modelsParameters <**> helper) idm)

  let (seedNet, (seedShuffle, seedTraining)) = SR.split <$> SR.split (mkStdGen 487239842)

  net0 <- case load of
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
   -    let subnet = x :~> NNil :: Network '[FullyConnected 30 1] '[ 'D1 30, 'D1 1 ]
   -    print $ runNet subnet (S1D 0) -- with this we get the biases
   -    print i -- biases for each output neuron
   -    print o -- weights between neurons
   -}

  -- Pretraining scores
  --print $ head shuffledSongs
  putStrLn $ "Test Size: " <> show (length testSet)
  putStrLn $ "Batch Size: " <> show batchSize

  -- Training Net
  let nets = take epochs $ evalRand (if batchSize > 1
                                     then trainOnBatchesEpochs net0 rate trainSet batchSize
                                     else epochTraining net0 rate trainSet)
                                    seedTraining
      net = last nets
      netsScores = fmap (\nn->(binaryNetError nn trainSet, binaryNetError nn testSet)) (net0:nets)

  -- Showing results of training net
  forM_ (zip [(0::Integer)..] netsScores) $ \(epoch, (trainError, testError)) ->
    putStrLn $ "Epoch " <> show epoch <> "\tTraining error: " <> show trainError <> "\tTesting error: " <> show testError

  case logs of
    Just logsPath -> saveScores logsPath $ fmap (fst *** fst) netsScores
    Nothing       -> return ()

  case save of
    Just saveFile -> B.writeFile saveFile $ runPut (put net)
    Nothing       -> return ()
