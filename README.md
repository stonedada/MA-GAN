# MA-GAN
The codes for the work "Resolution enhancement of cell microscopic images based on Mask-Assisted GAN"
<a id="system"></a>
# System requirements
## Hardware requirements
Inference can be run on a computer without a GPU in reasonable time, requiring only that the RAM if sufficient for the size of the model and the loaded image. 
Use the parameter ```gpu-ids=-1``` if the computer has no GPU, ```batch_size=1``` to avoid filling the RAM, and ```crop_size``` to run the inference on smaller crops if the available RAM is insufficient for the complete image.

For training MA-GAN, a GPU is necessary to reach reasonable computing times.

## Software requirements
### OS requirements
The source code was tested on Ubuntu 16.04, Ubuntu 20.04 and CentOS Linux 7.5.1804.

### Python dependencies
The source code was tested on Python 3.7. All required libraries are listed in the 'requirements.txt' file.

<a id="installation"></a>
# Installation
Clone this repository, then move to its directory:

```
git clone https://github.com/FLClab/TA-GAN.git
cd TA-GAN/
```


<a id="installation"></a>
# Documentation

## Pseudocode for the training algorithm
        opt, opt_val                      # Training options, validation options
        dataset = create_dataset(opt)     # Create a dataset given training options
        dataval = create_dataset(opt_val) # Create a dataset given validation options
        model = create_model(opt)         # Create a model given training options
        while (stopping criterion not satisfied)
            for data in dataset           # Iterate over all batches of data
                model.set_input(data)     # Unpack batch of data from dataset and apply preprocessing
                model.calculate_loss()    # Compute loss functions
                model.get_gradients()     # Get gradients
                model.optimize_parameters() # Update network weights
            update_learning_rate()        # Decrease learning rate if decaying lr is selected in options
        


## Training and testing on your own images

**A new model MUST be trained for every change in biological context or imaging parameter. You can not apply the models we provide on your images if their acquisition parameters differ in any way.** We strongly believe the method we introduce is applicable to any context, if trained properly. To train a model on a new set of images, you need the following:
- An extensive set of images that covers everything you could expect to see in your test images. The model won't learn to generate structures it has never encountered in training. 
- Segmentation annotations for all the training images
- Computational resources (GPU)
- A model adapted to your specific use case. The general MA-GAN model can be used here, but you need to specify the number of input and output channels for the generator and the segmentation network to fit your use case (default = 1).
- A dataloader adapted to your images. We provide a custom dataloader that is heavily commented so that you can easily modify it to fit your needs.

When everything is ready, run the training:
```
python3 train.py --dataroot=<DATASET> --model=<MODEL> --dataset_mode=<DATALOADER> 
```
The default hyperparameters might not lead to the best results. You should play with the following hyperparameters:
- ```--niter``` (number of iterations)
- ```--niter_decay``` (number of iterations with decreasing learning rate)
- ```--batch_size``` (rule of thumb, use the largest batch size that fits on your GPU)
- ```--netG``` (architecture of the generator)
- ```--netS``` (architecture of the segmentation network)

The description and default values for all hyperparameters can be consulted in options/base_options, train_options and test_options.

