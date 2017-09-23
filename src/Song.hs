{-# LANGUAGE TemplateHaskell #-}

module Song
    ( Song(..),
      genre, trackId, artistName, title, loudness, tempo, timeSignature, key, mode, duration, avgTimbre, varTimbre
    , line2song
    ) where

import           Lens.Micro.TH (makeLenses)

import           Data.Text (Text)
import           Text.Megaparsec (Parsec, Dec, count, sepBy, some, noneOf, char, parseMaybe)
import qualified Text.Megaparsec.Lexer as L (float, integer, signed)
import           Control.Monad (void)

data Song = Song
  {
    _genre          :: String,
    _trackId        :: String,
    _artistName     :: String,
    _title          :: String,

    _loudness       :: Double,
    _tempo          :: Double,
    _timeSignature  :: Integer,
    _key            :: Integer,
    _mode           :: Integer,
    _duration       :: Double,
    _avgTimbre      :: [Double], -- there are only 12 avg_timbre numbers
    _varTimbre      :: [Double]  -- there are only 12 var_timbre numbers
  } deriving (Eq, Show)

makeLenses ''Song

line2song :: Text -> Maybe Song
line2song = parseMaybe songParser

songParser :: Parsec Dec Text Song
songParser = Song <$> (cell <* comma)
                  <*> (cell <* comma)
                  <*> (cell <* comma)
                  <*> (cell <* comma)
                  <*> (double <* comma)
                  <*> (double <* comma)
                  <*> (integer <* comma)
                  <*> (integer <* comma)
                  <*> (integer <* comma)
                  <*> (double <* comma)
                  <*> count 12 (double <* comma)
                  <*> sepBy double comma -- there should be 12 in here

  where
    --cell :: Parsec Dec Text [Char]
    cell = some (noneOf [','])
    --comma :: Parsec Dec Text ()
    --comma = char ',' >> return ()
    comma = void (char ',')
    --double :: Parsec Dec Text Double
    double  = L.signed (return ()) L.float
    integer = L.signed (return ()) L.integer
