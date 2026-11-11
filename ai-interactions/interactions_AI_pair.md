we are working right now with our label mismatch, so let's consider it a bit later
right now we want to perform at least 3 UMAPs:
1) for each age group perform UMAP to look at the clusters of types from datasets
2) take the same umap, but add labels with model predictions, so we could see what changed
3) on the same umap mark cells with the wrong type

is there anything else we need to do here? 
also give us the code for it pls

Do you have any ideas how we can adjust and match the Labels?

i noticed that you suggested to make batch check. why did you decide to do it?

don't you think it's better to make the umaps interactive? why did you decide to make it just straight pictures? 
since we have a lot of cells, the visuals look quite messy. maybe we could make the UMAPs interactive?
-> added phase1_umaps.ipynb

we are also thinking of playing with gene filtering. let’s say, we want to perform the classification, but with the consideration of only top-50% expressed genes or top-10% or something else. what else would you suggest here to look at different model behaviour?