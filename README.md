# Final project for Human-computer interaction course
## Cross modal retrieval - Can we retrieve a text which describes an image?
This work is focused on finding of intermodal correspondences between textual and visual modalities with incorporation gaze
information as a regularization technique.

<p align="center"><img src="/intro_image.png" width="500px"/></p>

### Datasets
* MS-coco: download and place under "dataset" folder.
* Salicon: download and place under "dataset" folder.
* run `python utils.py`

Or download from [here](https://drive.google.com/file/d/1hE0zkUsGYb1iHXstQL8Z1swwH_un1gc5/view?usp=sharing).
### Training
Depends on which sampling algorithm,
 * batch sampling: `python train_two_encoders.py`
 * queue sampling: `python train_queue.py`
 * rerank sampling: `python train_rerank.py`
 
Note: training expects high GPU memory usage, so either use a single GPU with more than 10GB or two GPUs. If it is the former case, change argument `multi` to `0`.
### Inference
See `inference_with_two_enc.py`.

### Visualization
You can vizualize the results via Tensorboard, specify the path to the logs

```
tensorboard --logdir=path/to/the/log
```
### Examples
Examples are available here: `debug_post_training.ipynb` 
Set the name(timeline) of your pre-trained model:

```
timeline = "20200130-130202"
````
### Models
We provide a sample model pre-trained with the queue sampling algorithim (queue size 0.3)
Download from [here](https://drive.google.com/file/d/1KVhywaF1U17OJgHqNFOkba113WUTGogi/view?usp=sharing).
