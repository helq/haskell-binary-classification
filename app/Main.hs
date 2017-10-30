{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DataKinds         #-}
{-# LANGUAGE TypeFamilies      #-}
{-# LANGUAGE RecordWildCards   #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs             #-}
{-# LANGUAGE FlexibleContexts  #-}

module Main where

import           Lens.Micro.Extras (view)
import qualified Data.Text.IO as Text (readFile)
import           Data.Text (lines)
import           Data.Maybe (mapMaybe, fromMaybe)
import           Data.List (zip4) --, foldl')

import           Control.Monad (forM_)
import           Control.Monad.Random (evalRand)
import           System.Random (mkStdGen, RandomGen)
import qualified System.Random as SR (split)

import           Data.Semigroup ((<>))
import           Options.Applicative (Parser, option, auto, optional, long, short, value,
                                      strOption, helper, idm, execParser, info, (<**>), (<|>),
                                      flag')
import qualified Data.ByteString as B
import           Data.Serialize (Serialize, runPut, put, runGet, get) --, Get
import           System.Directory (createDirectoryIfMissing)

import           Numeric.LinearAlgebra.Static ((#), (&), R, unwrap, konst) --, zipWithVector)
import           Numeric.LinearAlgebra (toList)

--import           GHC.TypeLits
import           Data.Singletons (Sing, SomeSing(SomeSing), toSing)
import           Data.Singletons.TypeLits (KnownNat, Nat, Sing(SNat))
import           Data.Singletons.Prelude.List (Head, Last)

import           Grenade (Network,  FullyConnected, Tanh, Logit, Shape(D1), --Relu,
                         LearningParameters(LearningParameters), randomNetwork, S(S1D),
                         Gradients)
--import           Grenade (Network((:~>), NNil),  FullyConnected(FullyConnected), FullyConnected'(FullyConnected'), Tanh, Logit, Shape(D1), runNet, Relu,
--                         LearningParameters(LearningParameters), randomNetwork, S(S1D))

import           GrenadeExtras (binaryNetError, trainOnBatchesEpochs, normalize, hotvector)
import           GrenadeExtras.Zip (Zip)
--import           GrenadeExtras.Orphan()
import           GrenadeExtras.GradNorm (GradNorm)

import           Song (Song, line2song, genre)
import           Song.Grenade (song2TupleRn) --, SongSD)
import           Shuffle (shuffle) -- TODO: use the "standard" shuffleM http://hackage.haskell.org/package/random-shuffle

--import           Debug.Trace (trace)

type LogisticRegression = Network '[ FullyConnected 75 1, Logit ]
                                  '[ 'D1 75, 'D1 1, 'D1 1 ]

type OneHiddenLayer n = Network '[ FullyConnected 75 n, Tanh, FullyConnected n 1, Logit ]
                                '[ 'D1 75, 'D1 n, 'D1 n, 'D1 1, 'D1 1 ]

type DiscreteSepContinuousNet = Network '[ Zip ('D1 21) ('D1 2) (FullyConnected 21 2) ('D1 54) ('D1 2) (FullyConnected 54 2),
                                           Tanh, FullyConnected 4 1,
                                           Logit ]
                                        '[ 'D1 75, 'D1 4, 'D1 4, 'D1 1, 'D1 1 ]

type HugeNetwork = Network '[ FullyConnected 75 40, Tanh, FullyConnected 40 10, Tanh, FullyConnected 10 3, Tanh, FullyConnected 3 1, Logit ]
                           '[ 'D1 75, 'D1 40, 'D1 40, 'D1 10, 'D1 10, 'D1 3, 'D1 3, 'D1 1, 'D1 1 ]

--type FFNet = Network '[ FullyConnected 75 100, Tanh, FullyConnected 100 40, Tanh, FullyConnected 40 20, Relu, FullyConnected 20 1, Logit ]
--                     '[ 'D1 75, 'D1 100, 'D1 100, 'D1 40, 'D1 40, 'D1 20, 'D1 20, 'D1 1, 'D1 1 ]

--netLoad :: KnownNat n => FilePath -> IO (OneHiddenLayer n)
netLoad :: Serialize b => FilePath -> IO b
netLoad modelPath = do
  modelData <- B.readFile modelPath
  either fail return $ runGet get modelData

createKFoldPartitions :: Int -> [a] -> (Int, [([a],[a])])
createKFoldPartitions size ks = (parts, take parts . separateTrainAndValidLists . splitInSizes $ ks)
  where
    len = length ks
    parts = floor $ fromIntegral len / (fromIntegral size :: Double)

    splitInSizes :: [a] -> [[a]]
    splitInSizes [] = []
    splitInSizes xs =
      let (start, finish) = splitAt size xs
       in start : splitInSizes finish

    separateTrainAndValidLists :: [[a]] -> [([a], [a])]
    separateTrainAndValidLists = fmap (\(t,v)->(t [], v)) . separate' ([]++)

    separate' :: ([a]->[a]) -> [[a]] -> [([a]->[a], [a])]
    separate' acc [x]    = [(acc, x)]
    separate' acc (x:xs) = (acc . (concat xs++), x) : separate' (acc . (x++)) xs
    separate' _   []     = error "this should never happen, the size of the partition is too big to even hold an element"

saveScores :: FilePath -> [NetScore] -> IO ()
saveScores logsPath scores = writeFile logsPath (headScores<>"\n"<>scoresStr) -- TODO: catch IO Errors!
  where
    scoresStr :: String
    scoresStr = mconcat . fmap netScore2String $ zip [(0::Int)..] scores

    netScore2String (i, NetScore{..}) = show i <> "\t"
                                     <> show trainClassError <> "\t"
                                     <> show trainingError   <> "\t"
                                     <> show testClassError  <> "\t"
                                     <> show gradientNorm    <> "\n"

    headScores = "epoch\ttrain_classification_error\ttrain_error\ttest_classification_error\tgradient_norm"

-- Reading data and running code

filenameDataset :: String
filenameDataset = "./msd_genre_dataset.txt"

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

data TypeOfModelToTrain = LogitRegModel | OneHiddenLayerModel Integer | ArbitraryNNModel Int

data ModelsParameters =
  ModelsParameters
    Float -- Size of test set
    Int   -- Size of batch
    Bool  -- Normalize data
    StoppingCondition
    LearningParameters -- Parameters for Gradient Descent
    (Maybe FilePath) -- Load path
    (Maybe FilePath) -- Save path
    (Maybe FilePath) -- Logs path
    (Maybe Double)   -- K-fold size of validation set
    TypeOfModelToTrain

modelsParameters :: Parser ModelsParameters
modelsParameters =
  ModelsParameters <$> option auto (long "test-set"   <> short 't' <> value 0.15)
                   <*> option auto (long "batch-size" <> short 'b' <> value 40)
                   <*> option auto (long "normalize"  <> value True)
                   <*> ( StoppingCondition
                         <$> option auto (long "epochs" <> short 'e' <> value 300)
                         <*> option auto (long "gradient-smaller" <> short 'g' <> value (0.085, 3))
                         <*> option auto (long "max-error-change" <> value (0.004, 15))
                       )
                   <*> (LearningParameters
                       <$> option auto (long "train_rate" <> short 'r' <> value 0.01)
                       <*> option auto (long "momentum" <> value 0.9)
                       <*> option auto (long "l2" <> value 0) -- 0.0005
                       )
                   <*> optional (strOption (long "load"))
                   <*> optional (strOption (long "save"))
                   <*> optional (strOption (long "logs"))
                   <*> optional (option auto (long "k-fold-val-size"))
                   <*> (   flag' LogitRegModel (long "logit-reg")
                       <|> (OneHiddenLayerModel <$> option auto (long "one-hidden-layer" <> short 'h'))
                       <|> (ArbitraryNNModel    <$> option auto (long "arbitrary-nn"))
                       )

addLog :: KnownNat n => R n -> R n
addLog a = signum a * log (abs a+1)

discreteToOneHotVector :: R 3 -> R 21
discreteToOneHotVector featDiscre = (fromMaybe (konst 0) $ hotvector (round a)::R 8)
                                  # (fromMaybe (konst 0) $ hotvector (round b)::R 12)
                                  & c
  where (a,b,c) = case toList $ unwrap featDiscre of
                    [a',b',c'] -> (a',b',c')
                    _          -> error $  "This is weird, there should be only three discrete "
                                        <> "features (R 3), aka. this never happens but the compiler "
                                        <> "asks me to give an exahustive list for pattern matching, doh'"

takeWhileCond :: ([a] -> [b]) -> (b -> Bool) -> [a] -> [a]
takeWhileCond toCond stopCond xs = fmap snd . takeWhile (stopCond . fst) $ zip (toCond xs) xs

data NetScore = NetScore {
  trainClassError :: Double, -- Classification training error
  trainingError   :: Double, -- (optimization) training error
  testClassError  :: Double, -- Classification testing error
  gradientNorm    :: Double  -- accumulated norm of the gradients on an epoch
} deriving (Show)

netScore :: (Last shapes ~ 'D1 1, Foldable t)
         => t (S (Head shapes), S ('D1 1))
         -> t (S (Head shapes), S ('D1 1))
         -> (Double, Network layers shapes)
         -> NetScore
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

loadModel (Just loadFile) _       = netLoad loadFile
loadModel Nothing         seedNet = return $ evalRand randomNetwork seedNet

main :: IO ()
main = do
  mparams@(ModelsParameters _ _ _ _ _ load _ _ _ modelType) <- execParser (info (modelsParameters <**> helper) idm)

  let (seedNet, randSeed) = SR.split (mkStdGen 487239842)

  case modelType of
    LogitRegModel ->
      (loadModel load seedNet :: IO LogisticRegression) >>= mainWithRandmNet mparams randSeed

    OneHiddenLayerModel sizeHidden ->
      let (singSizeHidden :: SomeSing Nat) = toSing sizeHidden
      in case singSizeHidden of
           SomeSing (SNat :: Sing n) ->
             (loadModel load seedNet :: IO (OneHiddenLayer n)) >>=
             mainWithRandmNet mparams randSeed

    ArbitraryNNModel arbModelNum ->
      case arbModelNum of
        1 -> (loadModel load seedNet :: IO DiscreteSepContinuousNet) >>= mainWithRandmNet mparams randSeed
        2 -> (loadModel load seedNet :: IO HugeNetwork) >>= mainWithRandmNet mparams randSeed
        _ -> putStrLn $ "Sorry but there is no arbitrary neural net number " <> show arbModelNum


mainWithRandmNet :: (Head shapes ~ 'D1 75,
                     Last shapes ~ 'D1 1,
                     Show (Network layers shapes),
                     Serialize (Network layers shapes),
                     GradNorm (Gradients layers),
                     Num (Gradients layers),
                     RandomGen g)
                  => ModelsParameters -> g -> Network layers shapes -> IO ()
mainWithRandmNet mparams randSeed net0 = do
  let (seedShuffle, seedTraining) = SR.split randSeed
      ModelsParameters testPerc batchSize norm stopCond rate _ save logs kfoldValPer modelType = mparams
      modelName = case modelType of
                    LogitRegModel                  -> fromMaybe "logitReg"  logs
                    OneHiddenLayerModel sizeHidden -> fromMaybe "hidden"    logs <> "-" <> show sizeHidden
                    ArbitraryNNModel arbModelNum   ->
                      fromMaybe "arbitrary" logs <> "-" <> case arbModelNum of
                                                             1 -> "separated_discrete_continuous"
                                                             2 -> "huge_network"
                                                             _ -> error $ "There is no arbitrary Neural Net model #" <> show arbModelNum

  putStrLn $ "Model name: " <> modelName

  let normFun :: KnownNat m => [R m] -> [R m]
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

  let sizeSongs = length songsRaw
      -- Shuffling songs
      shuffledSongs       = evalRand (shuffle songs) seedShuffle
      (testSet, trainSet) = splitAt (round $ fromIntegral sizeSongs * testPerc) shuffledSongs

  {-
   -case net0 of
   -  x@(FullyConnected (FullyConnected' i o) _) :~> _ -> do
   -    let subnet = x :~> NNil :: Network '[FullyConnected 75 n] '[ 'D1 75, 'D1 n ]
   -    print $ runNet subnet (S1D 0) -- with this we get the biases
   -    --print . sqrt . normSquared $ net0
   -    print i -- biases for each output neuron
   -    print o -- weights between neurons
   -}

  -- Pretraining scores
  --print $ head shuffledSongs
  putStrLn $ "Total Size: " <> show sizeSongs
  putStrLn $ "Test Size:  " <> show (length testSet)
  putStrLn $ "Batch Size: " <> show batchSize

  case kfoldValPer of
    Nothing -> trainNet net0 trainSet testSet rate batchSize seedTraining stopCond logs save
    Just valPer -> do
      let valSize             = floor $ fromIntegral sizeSongs * valPer
          (parts, kfoldSets)  = createKFoldPartitions valSize trainSet
      putStrLn $ "Validation Size: " <> show valSize
      putStrLn $ "Total parts for kfold: " <> show parts

      forM_ (zip [(1::Int)..] kfoldSets) $ \(kfoldNum, (trainSet', valSet)) -> do
        let dirSaveNet = "kfold-val" <> show valPer <> "/" <> modelName <> "/" <> "kfoldnet" <> "-" <> show kfoldNum
            logsKfoldN = dirSaveNet <> ".txt"
            saveKfoldN = dirSaveNet <> "/" <> fromMaybe "nnet-hidden" save
        putStrLn ""
        createDirectoryIfMissing True dirSaveNet -- TODO: catch exceptions
        putStrLn $ "Results to be saved in " <> logsKfoldN
        putStrLn $ "Nets to be saved in " <> saveKfoldN
        trainNet net0 trainSet' valSet rate batchSize seedTraining stopCond (Just logsKfoldN) (Just saveKfoldN)


trainNet :: (Last shapes ~ 'D1 1,
             RandomGen g,
             Num (Gradients layers),
             GradNorm (Gradients layers),
             Show (Network layers shapes),
             Serialize (Network layers shapes))
         => Network layers shapes
         -> [(S (Head shapes), S (Last shapes))]
         -> [(S (Head shapes), S ('D1 1))]
         -> LearningParameters
         -> Int
         -> g
         -> StoppingCondition
         -> Maybe FilePath
         -> Maybe FilePath
         -> IO ()
trainNet net0 trainSet testSet rate batchSize seedTraining stopCond logs save = do
  -- Training Net
  let netsInf = (read "Infinity", net0) : evalRand (trainOnBatchesEpochs net0 rate trainSet batchSize) seedTraining
      netsScoresInf = fmap (netScore trainSet testSet) netsInf

      (nets, netsScores) = unzip . takeWhileCondFunc stopCond $ zip (fmap snd netsInf) netsScoresInf
      --net = last nets

  putStrLn $ "Epoch"
           <> "\tTraining classification error"
           <> "\tTraining error"
           <> "\tTesting error"
           <> "\tGradNorm"

  -- Showing results of training net
  forM_ (zip3 [(0::Integer)..] netsScores nets) $ \(epoch, NetScore trainCE trainE testE gradNorm, currentNet) -> do
    putStrLn $ show epoch
             <> "\t" <> show trainCE
             <> "\t" <> show trainE
             <> "\t" <> show testE
             <> "\t" <> show gradNorm
    -- saving current trained net
    case save of
      Just saveFile -> let saveFileBin = saveFile <> "-e_" <> show epoch
                                                  <> "-trainCE_" <> show trainCE
                                                  <> "-testE_"   <> show testE
                                                  <> ".bin"
                        in B.writeFile saveFileBin $ runPut (put currentNet)
      Nothing -> return ()

  case logs of
    Just logsPath -> saveScores logsPath netsScores
    Nothing       -> return ()

  --case save of
  --  Just saveFile -> B.writeFile saveFile $ runPut (put net)
  --  Nothing       -> return ()
