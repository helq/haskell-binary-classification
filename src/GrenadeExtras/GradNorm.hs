{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE AllowAmbiguousTypes   #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE FlexibleContexts      #-}

module GrenadeExtras.GradNorm (
  GradNorm(..)
) where

import Grenade (FullyConnected'(FullyConnected'), FullyConnected(FullyConnected), Tanh,
                Logit, Gradient, Relu, Gradients(GNil, (:/>)))
import Numeric.LinearAlgebra.Static ((<.>), toColumns)
import GHC.TypeLits (KnownNat)

class GradNorm x where
  normSquared :: x -> Double

instance (KnownNat i, KnownNat o) => GradNorm (FullyConnected' i o) where
  normSquared (FullyConnected' r l) = r <.> r + (sum . fmap (\c->c<.>c) . toColumns $ l)

instance (KnownNat i, KnownNat o) => GradNorm (FullyConnected i o) where
  normSquared (FullyConnected w m) = normSquared w + normSquared m

instance GradNorm () where
  normSquared _ = 0

instance GradNorm Logit where
  normSquared _ = 0

instance GradNorm Tanh where
  normSquared _ = 0

instance GradNorm Relu where
  normSquared _ = 0

instance GradNorm (Gradients '[]) where
  normSquared GNil = 0

instance (GradNorm l, GradNorm (Gradient l), GradNorm (Gradients ls)) => GradNorm (Gradients (l ': ls)) where
  normSquared (grad :/> grest) = normSquared grad + normSquared grest

instance (GradNorm a, GradNorm b) => GradNorm (a, b) where
  normSquared (a, b) = normSquared a + normSquared b
