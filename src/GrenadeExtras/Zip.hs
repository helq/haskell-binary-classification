{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE ScopedTypeVariables   #-}

module GrenadeExtras.Zip (
  Zip (..),
) where

import           Data.Serialize

--import           Data.Singletons
import           GHC.TypeLits

import           Grenade.Core

import           Numeric.LinearAlgebra.Static ((#), split, R) -- row, (===), splitRows, unrow

data Zip :: Shape -> Shape -> * -> Shape -> Shape -> * -> * where
  Zip :: x -> y -> Zip ix ox x iy oy y

instance (Show x, Show y) => Show (Zip ix ox x iy oy y) where
  show (Zip x y) = "Zip\n" ++ show x ++ "\n" ++ show y

instance (UpdateLayer x, UpdateLayer y) => UpdateLayer (Zip ix ox x iy oy y) where
  type Gradient (Zip ix ox x iy oy y) = (Gradient x, Gradient y)
  runUpdate lr (Zip x y) (x', y') = Zip (runUpdate lr x x') (runUpdate lr y y')
  createRandom = Zip <$> createRandom <*> createRandom

-- instance ( SingI i
--          , Layer x i ('D1 o)
--          , Layer y i ('D1 o)
--          ) => Layer (Concat ('D1 o) x ('D1 o) y) i ('D2 2 o) where
--   type Tape (Concat ('D1 o) x ('D1 o) y) i ('D2 2 o) = (Tape x i ('D1 o), Tape y i ('D1 o))
--
--   runForwards (Concat x y) input =
--     let (xT, xOut :: S ('D1 o)) = runForwards x input
--         (yT, yOut :: S ('D1 o)) = runForwards y input
--     in case (xOut, yOut) of
--         (S1D xOut', S1D yOut') ->
--             ((xT, yT), S2D (row xOut' === row yOut'))
--
--   runBackwards (Concat x y) (xTape, yTape) (S2D o) =
--     let (ox, oy) = splitRows o
--         (x', xB :: S i) = runBackwards x xTape (S1D $ unrow ox)
--         (y', yB :: S i) = runBackwards y yTape (S1D $ unrow oy)
--     in  ((x', y'), xB + yB)

instance ( Layer x ('D1 p) ('D1 m)
         , Layer y ('D1 q) ('D1 n)
         , KnownNat r
         , KnownNat p
         , KnownNat q
         , r ~ (p + q)
         , q ~ (r - p)
         , (p <=? r) ~ 'True
         , KnownNat o
         , KnownNat m
         , KnownNat n
         , o ~ (m + n)
         , n ~ (o - m)
         , (m <=? o) ~ 'True
         ) => Layer (Zip ('D1 p) ('D1 m) x ('D1 q) ('D1 n) y) ('D1 r) ('D1 o) where
   type Tape (Zip ('D1 p) ('D1 m) x ('D1 q) ('D1 n) y) ('D1 r) ('D1 o) = (Tape x ('D1 p) ('D1 m), Tape y ('D1 q) ('D1 n))

   runForwards (Zip x y) (S1D input) =
     let (inputX :: R p, inputY :: R q) = split input
         (xT, xOut :: S ('D1 m)) = runForwards x (S1D inputX)
         (yT, yOut :: S ('D1 n)) = runForwards y (S1D inputY)
     in case (xOut, yOut) of
         (S1D xOut', S1D yOut') ->
             ((xT, yT), S1D (xOut' # yOut'))

   runBackwards (Zip x y) (xTape, yTape) (S1D o) =
     let (ox :: R m , oy :: R n) = split o
         (x', xB :: S ('D1 p)) = runBackwards x xTape (S1D ox)
         (y', yB :: S ('D1 q)) = runBackwards y yTape (S1D oy)
      in case (xB, yB) of
           (S1D xB', S1D yB') ->
             ((x', y'), S1D (xB' # yB'))

-- instance ( SingI i
--          , Layer x i ('D3 rows cols m)
--          , Layer y i ('D3 rows cols n)
--          , KnownNat (rows * n)
--          , KnownNat (rows * m)
--          , KnownNat (rows * o)
--          , KnownNat o
--          , KnownNat m
--          , KnownNat n
--          , ((rows * m) + (rows * n)) ~ (rows * o)
--          , ((rows * o) - (rows * m)) ~ (rows * n)
--          , ((rows * m) <=? (rows * o)) ~ 'True
--          ) => Layer (Concat ('D3 rows cols m) x ('D3 rows cols n) y) i ('D3 rows cols o) where
--   type Tape (Concat ('D3 rows cols m) x ('D3 rows cols n) y) i ('D3 rows cols o) = (Tape x i ('D3 rows cols m), Tape y i ('D3 rows cols n))
--
--   runForwards (Concat x y) input =
--     let (xT, xOut :: S ('D3 rows cols m)) = runForwards x input
--         (yT, yOut :: S ('D3 rows cols n)) = runForwards y input
--     in case (xOut, yOut) of
--         (S3D xOut', S3D yOut') ->
--             ((xT, yT), S3D (xOut' === yOut'))
--
--   runBackwards (Concat x y) (xTape, yTape) (S3D o) =
--     let (ox, oy) = splitRows o
--         (x', xB :: S i) = runBackwards x xTape (S3D ox :: S ('D3 rows cols m))
--         (y', yB :: S i) = runBackwards y yTape (S3D oy :: S ('D3 rows cols n))
--     in  ((x', y'), xB + yB)

instance (Serialize a, Serialize b) => Serialize (Zip sia sa a sib sb b) where
  put (Zip a b) = put a *> put b
  get = Zip <$> get <*> get
