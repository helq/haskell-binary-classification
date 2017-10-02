{-# LANGUAGE DataKinds         #-}
{-# LANGUAGE TypeOperators     #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts  #-}
{-# LANGUAGE TypeFamilies      #-}

module GrenadeExtras.OrphanNum () where

import GHC.TypeLits (KnownNat)

import Grenade (Gradient, Gradients(GNil, (:/>)), FullyConnected(FullyConnected), FullyConnected'(FullyConnected'),
                Logit(Logit), Tanh(Tanh), Relu(Relu), UpdateLayer)

instance (KnownNat i, KnownNat o) => Num (FullyConnected' i o) where
  (FullyConnected' r1 l1) + (FullyConnected' r2 l2) = FullyConnected' (r1+r2) (l1+l2)
  (FullyConnected' r1 l1) - (FullyConnected' r2 l2) = FullyConnected' (r1-r2) (l1-l2)
  (FullyConnected' r1 l1) * (FullyConnected' r2 l2) = FullyConnected' (r1*r2) (l1*l2)
  abs    (FullyConnected' r l) = FullyConnected' (abs r) (abs l)
  signum (FullyConnected' r l) = FullyConnected' (signum r) (signum l)
  negate (FullyConnected' r l) = FullyConnected' (negate r) (negate l)
  fromInteger i = FullyConnected' (fromInteger i) (fromInteger i)

instance (KnownNat i, KnownNat o) => Num (FullyConnected i o) where
  (FullyConnected w1 m1) + (FullyConnected w2 m2) = FullyConnected (w1+w2) (m1+m2)
  (FullyConnected w1 m1) - (FullyConnected w2 m2) = FullyConnected (w1-w2) (m1-m2)
  (FullyConnected w1 m1) * (FullyConnected w2 m2) = FullyConnected (w1*w2) (m1*m2)
  abs    (FullyConnected w m) = FullyConnected (abs w) (abs m)
  signum (FullyConnected w m) = FullyConnected (signum w) (signum m)
  negate (FullyConnected w m) = FullyConnected (negate w) (negate m)
  fromInteger i = FullyConnected (fromInteger i) (fromInteger i)

instance Num () where
  () + () = ()
  () - () = ()
  () * () = ()
  abs () = ()
  signum () = ()
  negate () = ()
  fromInteger _ = ()

instance (Num a, Num b) => Num (a, b) where
  (w1, m1) + (w2, m2) = (w1+w2, m1+m2)
  (w1, m1) - (w2, m2) = (w1-w2, m1-m2)
  (w1, m1) * (w2, m2) = (w1*w2, m1*m2)
  abs    (w, m) = (abs w,         abs m)
  signum (w, m) = (signum w,      signum m)
  negate (w, m) = (negate w,      negate m)
  fromInteger i = (fromInteger i, fromInteger i)

instance Num Logit where
  Logit + Logit = Logit
  Logit - Logit = Logit
  Logit * Logit = Logit
  abs Logit = Logit
  signum Logit = Logit
  negate Logit = Logit
  fromInteger _ = Logit

instance Num Tanh where
  Tanh + Tanh = Tanh
  Tanh - Tanh = Tanh
  Tanh * Tanh = Tanh
  abs Tanh = Tanh
  signum Tanh = Tanh
  negate Tanh = Tanh
  fromInteger _ = Tanh

instance Num Relu where
  Relu + Relu = Relu
  Relu - Relu = Relu
  Relu * Relu = Relu
  abs Relu = Relu
  signum Relu = Relu
  negate Relu = Relu
  fromInteger _ = Relu

instance Num (Gradients '[]) where
  GNil + GNil = GNil
  GNil - GNil = GNil
  GNil * GNil = GNil
  abs GNil = GNil
  signum GNil = GNil
  negate GNil = GNil
  fromInteger _ = GNil

instance (Num x, Num (Gradients xs), Num (Gradient x), UpdateLayer x) => Num (Gradients (x ': xs)) where
  (l1 :/> ls1) + (l2 :/> ls2) = (l1+l2) :/> (ls1+ls2)
  (l1 :/> ls1) - (l2 :/> ls2) = (l1-l2) :/> (ls1-ls2)
  (l1 :/> ls1) * (l2 :/> ls2) = (l1*l2) :/> (ls1*ls2)
  abs    (l :/> ls) = abs    l :/> abs    ls
  signum (l :/> ls) = signum l :/> signum ls
  negate (l :/> ls) = negate l :/> negate ls
  fromInteger i = fromInteger i :/> fromInteger i
