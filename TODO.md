# Todo:

### Research Question: Does utilizing a post-training pruning scheme from one model as a pre-training scheme on the same model architecture trained on different data facilitate or hinder classification accuracy in neural network training?

### Objective: Investigate whether utilizing a post-training pruning scheme as a pre-training scheme on the same model architecture but with different training data can enhance performance and efficiency in deep neural network training.

- [x] 1) train net model to desired accuracy (99%)
- [x] 2) next, find a suitable pruning method for net models AFTER training
    - [x] what is the highest sparsity we can reach and still acheive 96% accuracy?
    - [x] once achieved, save at what indices we pruned.
- [x] 3) before training, apply the pruning scheme from step 2. then train. Do we see a similar level of accuracy? 
- [x] 4) before training, apply random pruning of the same sparsity level and train. is difference of accuracies in step 3 and 4 significant?
- [x] 5) If we do, research further. If not, what may have caused this to not work?

### --------------------------------------------------------------------

- [x] 5) repeat step 3, but train on FashionMNIST instead of MNIST
- [x] 6) Is the accuracy acceptable?
