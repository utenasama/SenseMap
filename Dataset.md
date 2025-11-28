**Hint**: any commits that could affect the result should test on the benchmarks first and record experimental data!

We choose the following dataset to evaluate:

#### 1.[ETH3D](https://www.eth3d.net/)
- **Tens** of images;
- **Indoors** & **Outdoors**;
- Ground truth **Poses** & **Scans**;
- Format of the *COLMAP*;

We choose five of them:

1.1 pipes - 14 images indoors

![pipes](https://www.eth3d.net/img/dslr_pipes.jpg)

1.2 meadow - 15 images outdoors

![meadow](https://www.eth3d.net/img/dslr_meadow.jpg)

1.3 relief - 31 images indoors

![relief](https://www.eth3d.net/img/dslr_relief.jpg)

1.4 courtyard - 38 images outdoors

![courtyard](https://www.eth3d.net/img/dslr_courtyard.jpg)

1.5 facade - 76 images outdoors

![facade](https://www.eth3d.net/img/dslr_facade.jpg)

For details, see https://www.eth3d.net/.

#### 2.[Tanks and Temples](https://www.tanksandtemples.org/)
- **Hundreds** of images;
- **Indoors** & **Outdoors**;
- Ground truth **Scans**;
- *COLMAP* Poses;

2.1 Caterpillar - 383 images objects

![caterpillar1](img_README/caterpillar1.jpg)

2.2 Ignatius - 263 images objects

![Ignatius1](img_README/Ignatius1.jpg)

2.3 Truck - 251 images objects

![Truck1](img_README/Truck1.jpg)

2.4 Barn - 384 images Outdoors

![Barn1](img_README/Barn1.jpg)

2.5 Church - 600 images Indoors

![Church](img_README/Church1.jpg)

2.6 Meetingroom - 371 images Indoors

![Meetingroom](img_README/Meetingroom1.jpg)

2.7 Courthouse - 1106 images Outdoors

![Courthouse](img_README/Courthouse1.jpg)

For details, see https://www.tanksandtemples.org/download/.

#### 3.[ICL-NUIM](http://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html)

Choose the Living Room Dataset

- **Thousands** of images;
- **Indoor**
- Ground truth **Poses** & **Surfaces**;

Four video for one living room:

3.1 video0 - 1509 images Indoors

3.2 video1 - 966 images Indoors

3.3 video2 - 882 images Indoors

3.4 video3 - 1241 images Indoors

![living room](img_README/room3.png)

For details, see http://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html.

#### 4.[KITTI](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)

- **Large-scale** **outdoors**;
- Ground truth **Poses**;

![KITTI](http://www.cvlibs.net/datasets/kitti/images/header_odometry.jpg)

For details, see http://www.cvlibs.net/datasets/kitti/eval_odometry.php.