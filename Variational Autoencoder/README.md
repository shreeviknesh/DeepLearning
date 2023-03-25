# Variational AutoEncoders
- Implements the paper [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf) by Kingma et al.
- Inspired by [Aladdin Persson](https://www.youtube.com/watch?v=VELQT1-hILo&t=2002s)'s YouTube video.


## Generated Outputs
### Zero
![label_0_image0](outputs/0/0.png)
![label_0_image1](outputs/0/1.png)
![label_0_image2](outputs/0/2.png)
![label_0_image3](outputs/0/3.png)
![label_0_image4](outputs/0/4.png)
![label_0_image5](outputs/0/5.png)
![label_0_image6](outputs/0/6.png)
![label_0_image7](outputs/0/7.png)
![label_0_image8](outputs/0/8.png)
![label_0_image9](outputs/0/9.png)

### One
![label_1_image0](outputs/1/0.png)
![label_1_image1](outputs/1/1.png)
![label_1_image2](outputs/1/2.png)
![label_1_image3](outputs/1/3.png)
![label_1_image4](outputs/1/4.png)
![label_1_image5](outputs/1/5.png)
![label_1_image6](outputs/1/6.png)
![label_1_image7](outputs/1/7.png)
![label_1_image8](outputs/1/8.png)
![label_1_image9](outputs/1/9.png)

### Two
![label_2_image0](outputs/2/0.png)
![label_2_image1](outputs/2/1.png)
![label_2_image2](outputs/2/2.png)
![label_2_image3](outputs/2/3.png)
![label_2_image4](outputs/2/4.png)
![label_2_image5](outputs/2/5.png)
![label_2_image6](outputs/2/6.png)
![label_2_image7](outputs/2/7.png)
![label_2_image8](outputs/2/8.png)
![label_2_image9](outputs/2/9.png)

### Three
![label_3_image0](outputs/3/0.png)
![label_3_image1](outputs/3/1.png)
![label_3_image2](outputs/3/2.png)
![label_3_image3](outputs/3/3.png)
![label_3_image4](outputs/3/4.png)
![label_3_image5](outputs/3/5.png)
![label_3_image6](outputs/3/6.png)
![label_3_image7](outputs/3/7.png)
![label_3_image8](outputs/3/8.png)
![label_3_image9](outputs/3/9.png)

### Four
![label_4_image0](outputs/4/0.png)
![label_4_image1](outputs/4/1.png)
![label_4_image2](outputs/4/2.png)
![label_4_image3](outputs/4/3.png)
![label_4_image4](outputs/4/4.png)
![label_4_image5](outputs/4/5.png)
![label_4_image6](outputs/4/6.png)
![label_4_image7](outputs/4/7.png)
![label_4_image8](outputs/4/8.png)
![label_4_image9](outputs/4/9.png)

### Five
![label_5_image0](outputs/5/0.png)
![label_5_image1](outputs/5/1.png)
![label_5_image2](outputs/5/2.png)
![label_5_image3](outputs/5/3.png)
![label_5_image4](outputs/5/4.png)
![label_5_image5](outputs/5/5.png)
![label_5_image6](outputs/5/6.png)
![label_5_image7](outputs/5/7.png)
![label_5_image8](outputs/5/8.png)
![label_5_image9](outputs/5/9.png)

### Six
![label_6_image0](outputs/6/0.png)
![label_6_image1](outputs/6/1.png)
![label_6_image2](outputs/6/2.png)
![label_6_image3](outputs/6/3.png)
![label_6_image4](outputs/6/4.png)
![label_6_image5](outputs/6/5.png)
![label_6_image6](outputs/6/6.png)
![label_6_image7](outputs/6/7.png)
![label_6_image8](outputs/6/8.png)
![label_6_image9](outputs/6/9.png)

### Seven
![label_7_image0](outputs/7/0.png)
![label_7_image1](outputs/7/1.png)
![label_7_image2](outputs/7/2.png)
![label_7_image3](outputs/7/3.png)
![label_7_image4](outputs/7/4.png)
![label_7_image5](outputs/7/5.png)
![label_7_image6](outputs/7/6.png)
![label_7_image7](outputs/7/7.png)
![label_7_image8](outputs/7/8.png)
![label_7_image9](outputs/7/9.png)

### Eight
![label_8_image0](outputs/8/0.png)
![label_8_image1](outputs/8/1.png)
![label_8_image2](outputs/8/2.png)
![label_8_image3](outputs/8/3.png)
![label_8_image4](outputs/8/4.png)
![label_8_image5](outputs/8/5.png)
![label_8_image6](outputs/8/6.png)
![label_8_image7](outputs/8/7.png)
![label_8_image8](outputs/8/8.png)
![label_8_image9](outputs/8/9.png)

### Nine
![label_9_image0](outputs/9/0.png)
![label_9_image1](outputs/9/1.png)
![label_9_image2](outputs/9/2.png)
![label_9_image3](outputs/9/3.png)
![label_9_image4](outputs/9/4.png)
![label_9_image5](outputs/9/5.png)
![label_9_image6](outputs/9/6.png)
![label_9_image7](outputs/9/7.png)
![label_9_image8](outputs/9/8.png)
![label_9_image9](outputs/9/9.png)

## Notes:
- [TODO] As we can see from the outputs, the alignments of all the outputs are the same, this is because there is some high bias & low variance in the inputs & the model does not learn the general shape of the digits. Instead, it learns the digits along with their orientations. This can be overcome by including some transforms while training the data.