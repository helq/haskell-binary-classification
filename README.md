# Binary Classification Problem in Haskell (using Grenade) #


This repo solves a homework planned for my class on machine learning. The homework was
about applying Logistic Regression and Neural Nets to solve a "simple" binary
classification problem. For more info on what is really this code doing please refer to
my [homework report](./ml_hw1.pdf)

To reproduce the results given in the homework please follow the instructions below.

## Prerequisites ##

<!--TODO: talk about how to install stack-->
Install Stack (haskell) and exec `stack build`.

## Reproducing results ##

<!--TODO: add details on what are the lines of code below-->

### Testing a Neural Net without normalizing data before feeding it to the NN ##

This may take several hours and the classification training error doesn't change, but the
training error (i.e., the error used in the optimization process underlying the learning)
does get smaller from 3714.88 to 3664.00 after 10000 (10k) iterations.

```bash
mkdir -p training/hidden-30-no-preprocessing
stack exec -- homework1-exe \
               -b 100 \                              # batch size
               --l2 0 \                              # no regularization
               -g '(0.085,10)' \                     # stop when classification error on training data goes below 8.5% for 3 consecutive epochs
               -e 10000 \                            # total number of epochs
               --max-error-change '(0.004, 10000)' \ # (this is a vacuum statement here)
               -h 30 \                               # using neural net with a hidden layer of size 30
               --normalize False \                   # skip normalizing step of the data <- this makes the process agonizingly slow
               --save training/hidden-30-no-preprocessing/nnet-30-hidden \
               --logs training/hidden-30-no-preprocessing.txt
```

### Logistic Regression ##

```bash
mkdir -p training/logistic{,2}
stack exec -- homework1-exe --logit-reg -b 1000 -e 1000 --max-error-change '(0.004, 30)' --save training/logistic/logistic --logs training/logistic.txt
                                        #^ batch size   #^ stop if the change in the classification error in the last 100 epochs was smaller than 0.4%
                                                #^ 1000 epochs tops

stack exec -- homework1-exe --logit-reg -b 10000 -e 1000 --max-error-change '(0.004, 30)' --load training/logistic/logistic-e_1000*.bin --save training/logistic2/logistic --logs training/logistic2.txt
                                        #^ maximum batch size, aka. use all data to train
```

### Neural Nets with a hidden layer of arbitrary size ##

Performing k-fold cross validation:

```bash
for i in {1..100}; do
  stack exec -- homework1-exe \
                   -b 100 \                           # batch size
                   --l2 0 \                           # no regularization
                   -g '(0.085,3)' \                   # look above (Testing a Neural Net ...)
                   -e 300 \                           # force stop in epoch 300
                   --max-error-change '(0.004, 15)' \ # stop if the maximum classification training error change in the last 15 epochs has been less than 0.4$
                   -h $i \                            # using neural net with a hidden layer of size 30
                   --k-fold-val-size 0.14             # size of the partition to perform k-fold cross validation
done
```

#### Post processing ####

```bash
for i in {1..100}; do
  f="kfold-val0.14/hidden-$i/kfoldnet-";
  for file in "$f"*.txt; do echo -n "$file "; tail -n 1 "$file"; done;
done > kfold-val0.14/final-results.txt

stack exec -- postprocessing > kfold-val0.14/results-to-plot.txt
```

## Arbitrary Neural Networks ##

## PS ##

If you, Professor, are reading this, it means that you really cared about knowing what we
were doing to solve the problem and are asking yourself, problably, why I've wrote this,
well, I really don't know, I just wanted to acknowledge the possibility of you being here.
Anyway, have a nice day, sir.
