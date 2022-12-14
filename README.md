# BlueNoise
Code repository for the paper:

<b>Fast Blue Noise Generation via Unsupervised Learning</b><br>
[Daniele Giunchi](https://scholar.google.com/citations?user=04u9QAIAAAAJ)\*,
[Alejandro Sztrajman](https://asztr.github.io)\*,
[Anthony Steed](https://wp.cs.ucl.ac.uk/anthonysteed/)<br>
<i>International Joint Conference on Neural Networks</i> (IJCNN), 2022.

### [Project Page](https://asztr.github.io/publications/ijcnn2022/ijcnn2022.html) | [Paper](https://asztr.github.io/publications/ijcnn2022/ijcnn2022-preprint.pdf)

![ijcnn2022-teaser](https://user-images.githubusercontent.com/10238412/188176041-5dc7b7ed-41bf-468a-95f3-67875e009505.jpg)

### Training
Running the script `bn_train.py` will train the blue noise neural network model and save it as `model128.h5` and `model128.json`, where 128 indicates
the resolution of the square grayscale blue noise masks generated by the network.

### Prediction
Run the following line to generate a blue noise texture using the model `model128.h5`:
```
python bn_predict.py model128.h5 --cpu
```
This will create an output file `pred128.png` with the grayscale blue noise mask.

### Dithering
Use the script `dither.py` to perform dithering of an image with our generated blue noise:
```
python dither.py --images "img/meadow1.png" --noises "pred128.png" --bits 1
```
This uses the blue noise mask in `pred128.png` to dither the image `meadow1.png`, compressing it to a single bit per color channel,
outputting the file `dither_meadow1_pred128_bits4.png`.
<br><br>
<div align="center">
    <img src="img/meadow1.png" width="200px"> <img src="img/dither_meadow1_bluenoise128_bits2.png" width="200px"><br>
    <b>Left:</b> original image. <b>Right:</b> dithering with 2 bits per channel.
</div>

### BibTeX
If you find our work useful, please cite:
```
@article{giunchi2022bluenoise,
    author={Daniele Giunchi and Alejandro Sztrajman and Anthony Steed},
    title = {Fast Blue-Noise Generation via Unsupervised Learning},
    booktitle = {International Joint Conference on Neural Networks},
    year = {2022}
}
```
