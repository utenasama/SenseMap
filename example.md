# Example

## SfM
```
cd SenseMap/bin
Run test_all_configure_outside workspace/sfm.yaml
```

上述命令执行完毕，workspace下会生成一些结果文件

cameras.bin为相机模型文件

features.bin为特征文件

scene_graph.bin为场景图文件

scene_graph.png为场景图的可视化表示

0/为存放SfM结果的目录

0/cameras.bin为相机模型文件

0/images.bin为注册的图像数据文件

0/points3D.bin为地图点文件

## MVS
```
Run test_patch_match workspace/mvs.yaml
```

上述命令执行完毕，workspace/0下新增一个dense目录，存放稠密重建相关结果

```
Run test_fusion workspace/mvs.yaml
```

上述命令执行完毕，workspace/0/dense下生成两个新的文件

fused.ply场景稠密点云文件

fused.ply.vis场景稠密点云可视信息

其中fused.ply可以通过meshlab软件打开

## 配置文件相关

### sfm.yaml

image_path: 图像路径

workspace_path: 工作目录

single_camera: 所有图像是否共享一个相机参数，通常设置为0，表示每张图像的相机参数不同

在图像分辨率大小一致的情况下，也可设置为1

camera_model: 相机模型

- SIMPLE_PINHOLE: 简单针孔相机模型，只有一个焦距、一个主点坐标（f, cx, cy）
- PINHOLE: 针孔相机模型，两个焦距，一个主点坐标(fx, fy, cx, cy)
- SIMPLE_RADIAL: 带一个焦距、一个畸变参数的简单相机模型(f, cx, cy, k)
- RADIAL: 带一个焦距、两个畸变参数的相机模型(fx, fy, cx, cy, k1, k2)
- SPHERICAL: 全景相机模型

### mvs.yaml

image_path: 图像路径

workspace_path: 工作目录

image_type: 图像类型，一般有perspective、panorama和rgbd三种，普通手机拍摄的图片为perspective, 全景相机为panorama

max_image_size: 最大重建图像分辨率
