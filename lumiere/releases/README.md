# Releases

Welcome to Lumiére releases! This directory contains all the model releass for Lumiére over
it's lifetime, essentially serving as an archive of previous model versions. More of a
library, maybe?

## A Release

A Lumiére release includes 4 main parts:

1. A self contained Python module containing the source code of the model.
2. A checkpoint file containing the associated model weights.
3. A config file outlining the model's training parameters.
4. A model card outlining the model's performance and any additional info about the model.
5. The tokenizer to be used with the model.

> Note: Why These?
> Each release is intended to provide absolute clarity and reproduceability of a given
> version of Lumiére. Currently, the training pipeline is not included in the release
> but can be easily referenced from the code.

### Reproduceability, Not Production Grade

Since each release is tailored for understandability, many of the features that would be
present in a production codebase (logging, validation, exception handling etc...) are not
included. The code in each release isn't intended to be deployed as-is. The main goal is
to provide a clean canvas for understanding the model and getting hands on with
experimenting.
