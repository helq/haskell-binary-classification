# Binary Classification Problem in Haskell (using Grenade) #

This repo solves a machine learning learning homework problem proposed in the class
IELE4014 (Uniandes).

The task was about applying Logistic Regression and Neural Nets to solve a "simple" binary
classification problem. For more info on what is really this code doing please refer to my
[homework report](./ml_hw1.pdf) (The code to create the original report can be found
[in this repo](https://github.com/helq/report-hw1-IELE4014))

To **replicate** the results given in the report please follow the instructions below.
Thanks to [**@cesandovalp**](https://github.com/cesandovalp) for noticing the code below
replicates the experiments, not simply reproduces them. What is replicability you think,
well [Paper: Replicability is not Reproducibility (by Chris Drummond)](http://cogprints.org/7691/7/ICMLws09.pdf).

## Prerequisites ##

Install [Stack](https://docs.haskellstack.org/) (Haskell package manager) and run in
console:

```
stack setup # this may take quite a while depending on your system
stack build
```

Download a curated version of Million Song Dataset from
<https://labrosa.ee.columbia.edu/millionsong/blog/11-2-28-deriving-genre-dataset> and
uncompress it in this folder.

## Replicating results ##

Running the lines below in bash should **replicate** the results I got in the report (above).
Some minors adjustments are necessary to make all lines executable (I thought that
adding #'s at the end of lines finishing with \ would not be a problem, but apparently
they need to be removed :S, I leave them as documentation but take care).

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
mkdir -p training/logistic{-slow,1,2}
stack exec -- homework1-exe --logit-reg -b 10000 -e 700 --max-error-change '(-0.1, 1)' --save training/logistic-slow/logistic --logs training/logistic-slow.txt
                                        #^ batch size of 10k with train set of size 8k means that there is no batching, all data is used at once
                                                         #^ disabling stoping condition on small improvements in training error

stack exec -- homework1-exe --logit-reg -b 1000 -e 30 --max-error-change '(0.004, 30)' --save training/logistic1/logistic1 --logs training/logistic1.txt
                                        #^ batch size   #^ stop if the change in the classification error in the last 100 epochs was smaller than 0.4%
                                                #^ 1000 epochs tops

stack exec -- homework1-exe --logit-reg -b 10000 -e 300 --max-error-change '(0.002, 30)' --load training/logistic1/logistic1-e_30*.bin --save training/logistic2/logistic2 --logs training/logistic2.txt
                                        #^ maximum batch size (no batching), aka. use all data to train
```

### Neural Nets with a hidden layer of arbitrary size ##

Performing k-fold cross validation:

```bash
for i in {1..100}; do
  stack exec -- homework1-exe \
                   -b 100 \
                   --l2 0 \
                   -g '(0.085,3)' \
                   -e 300 \
                   --max-error-change '(0.004, 15)' \
                   -h $i \
                   --k-fold-val-size 0.14
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

#### Training selected model (2 neurons in hidden layer) ####

```bash
mkdir -p training/hidden-2
stack exec -- homework1-exe -b 100 --l2 0 -g '(0.085,3)' -e 300 --max-error-change '(0.004, 15)' -h 2 \
                            --save training/hidden-2/hidden-2 --logs training/hidden-2.txt
```


## Arbitrary Neural Networks ##

```bash
mkdir -p training/{huge,separated_discrete_continuous}
stack exec -- homework1-exe -b 100 --l2 0 --arbitrary-nn 1 \
                    --save training/separated_discrete_continuous/separated_discrete_continuous \
                    --logs training/separated_discrete_continuous.txt
stack exec -- homework1-exe -b 100 --l2 0 -g '(0.085,3)' --max-error-change '(0.004, 15)' --arbitrary-nn 2 --save training/huge/huge --logs training/huge.txt
```

## Plots ##

To plot is necessary to have installed gnuplot. Once you have it installed run in
terminal:

```bash
cp training/logistic{1,}.txt
tail -n +3 training/logistic2.txt >> training/logistic.txt
gnuplot gnuplotscripts/im0*.gn
```

## PS ##

If you, Professor, are reading this, it means that you really cared about knowing what we
were doing to solve the problem and are asking yourself, problably, why I've wrote this,
well, I really don't know, I just wanted to acknowledge the possibility of you being here.
Anyway, have a nice day, sir.
