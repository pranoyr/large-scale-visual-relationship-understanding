# Large-scale Visual Relationship Understanding

![alt text](https://github.com/pranoyr/lvrd/blob/master/results/Examples.PNG)
<p align="center">Example results from the VG80K dataset.</p>

Thie Repository is an implementation of the paper https://arxiv.org/pdf/1804.10660.pdf


## Download Pre-trained Model
Download the model from [here](https://drive.google.com/file/d/1uNisRSwajVLIj-56okui7q5EskBZ_OE6/view?usp=sharing).


## Download the VRD Dataet
Download the data VRD dataset from [here](https://cs.stanford.edu/people/ranjaykrishna/vrd/).


## Pretrained Word Vectors
Download pretrained embeddings from [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit).
Put it in the data/wordvectors folder. Folder structure is shown below.
    
    

### Folder Structure
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

### Note 
* All the weights will be saved to the snapshots folder 
* To resume Training from any checkpoint, Use
```
--weight_path <path-to-model> 
```

## Inference
```
python inference.py --input_image <path-to-image> --weight_path <path-to-weight-file>
```

## To Do
Visual Genome Dataset, VGG BackBone, Resnet 152

## Results
![alt text](https://github.com/pranoyr/large-scale-visual-relationship-understanding/blob/master/results/1.jpg)

## References
* https://github.com/facebookresearch/Large-Scale-VRD
* https://arxiv.org/pdf/1804.10660.pdf

## License
This project is licensed under the MIT License 