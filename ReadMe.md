# Large-scale Visual Relationship Understanding

![alt text](https://github.com/pranoyr/lvrd/blob/master/results/Examples.PNG)
<p align="center">Example results from the VG80K dataset.</p>

## Download the VRD Dataet
Download the data VRD dataset from [here](https://cs.stanford.edu/people/ranjaykrishna/vrd/).


## Pretrained Word Vectors
Download pretrained embeddings from [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit).
Put it in the data/wordvectors folder. Folder structure is shown below.
    
 
### Try on VRD 

```
+ data 
    + VRD
        - json_dataset
        - sg_dataset
        
    + wordvectors  
        - GoogleNews-vectors-negative300.bin  
           
```

## Train
```
python train_large_scale.py
```

## Note 
* All the weights will be saved to the snapshots folder 
* To resume Training from any checkpoint, Use
```
--weight_path <path-to-model> 
```

## Inference
```
python inference.py
```

## References
* https://github.com/facebookresearch/Large-Scale-VRD

## License
This project is licensed under the MIT License 