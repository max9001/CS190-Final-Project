
# Todo:

### Research Question: Does utilizing a post-training pruning scheme from one model as a pre-training scheme on the same model architecture trained on different data facilitate or hinder performance improvement and efficiency gains in deep neural network training?

- [x] 1) train net model to desired accuracy
- [ ] 2) randomly prune before training on same model and same data, see if noticeable difference in accuracy
- [ ] 3) next, find a suitable pruning method for net models AFTER training
- [ ] 4) apply this scheme to the same model and train again. Do we see improvements over step 2?



- [ ] 4) save the pruning scheme from previous method, and repeat step 2
- [ ] 5) we want to check if our results from step 2 and step 4 are different.
- [ ] 6) if step 4s accuracy is statistically significantly better than step 2, we proved out hypothesis and we should explore this idea further. if not, we should explore what we may have done wrong\
- [ ] 7) use successful pruning scheme on same model, DIFFERENT data, and see if accuracy is acceptable