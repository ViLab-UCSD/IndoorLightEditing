# Physically-based Editing of Indoor Scene Lighting from a Single Image

Zhengqin Li, Jia Shi, Sai Bi, Rui Zhu, Kalyan Sunkavalli, Miloš Hašan, Zexiang Xu, Ravi Ramamoorthi, Manmohan Chandraker

![](Images/teaser.png)

## Related links:
* [Trained models](https://drive.google.com/drive/folders/1jIaDIKKf3R_EpeMrxobA_HOMWO8D3W0C?usp=sharing)
* [Object insertion](https://github.com/lzqsd/VirtualObjectInsertion)
* [Dataset](https://ucsd-openrooms.github.io/)
* [Arxiv](https://arxiv.org/abs/2205.09343)

## Dependencies
We highly recommend using Anaconda to manage python packages. Required dependencies include:
* [pytorch](https://pytorch.org/)
* [pytorch3D](https://pytorch3d.org/)
* [OpenCV](https://opencv.org/)
* [Open3D](http://www.open3d.org/)
* [Nvidia Optix 5.11 - 6.0](https://developer.nvidia.com/designworks/optix/downloads/legacy)

## Train and test models on the OpenRooms dataset

![](Images/pipeline.png)

1. Download the [OpenRooms dataset](https://ucsd-openrooms.github.io/). 
2. Compile Optix-based shadow renderer with python binding. 
      * Go to [OptixRendererShadow](OptixRendererShadow) directory. Compile the code following this [link](https://github.com/lzqsd/OptixRenderer.git). 
3. Modify the pytorch3D code to support RMSE chamfer distance loss.
      * Go to [chamfer.py](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/loss/chamfer.py). 
      * Add flag `isRMSE = False` to function `chamfer_distance`
      * Modify function `chamfer_distance` by adding lines below after the definition of `cham_x` and `cham_y`
      ```python
      if isRMSE == True:
          cham_x = torch.sqrt(cham_x + 1e-6)
          cham_y = torch.sqrt(cham_y + 1e-6)
      ```
4. Train models 
     * Train the material prediction network. 
     ```python
     python trainBRDF.py           # Train material prediction.
     ```
     * Train light source prediction networks
     ```python
     python trainVisLamp.py        # Train visible lamp prediction.
     python trainVisWindow.py      # Train visible window prediction.
     python trainInvLamp.py        # Train invisible lamp prediction.
     python trainInvWindow.py      # Train invisible window prediction.
     ```
     * Train the neural renderer
     ```python
     python trainShadowDepth.py --isGradLoss      # Train shadow prediction.
     python trainDirectIndirect.py                # Train indirect illumination prediction.
     python trainPerpixelLighting.py              # Train perpixel lighting prediction.
     ```
5. Test models
     * Test the material prediction network
     ```python
     python testBRDF.py            # Test BRDF prediction. Results in Table 5 in the supp.
     ```
     * Test light source prediction networks
     ```python
     python testVisLamp.py         # Test visible lamp prediction. Results in Table 3 in the main paper.
     python testVisWindow.py       # Test visible window prediction. Results in Table 3 in the main paper. 
     python testInvLamp.py         # Test invisible lamp prediction. Results in Table 3 in the main paper.
     python testInvWindow.py       # Test invisible window prediction. Results in Table 3 in the main paper.
     ```
     * Test the neural renderer
     ```python
     python testShadowDepth.py --isGradLoss       # Test shadow prediction. Results in Table 2 in the main paper. 
     python testDirectIndirect.py                 # Test indirect illumination prediction. 
     python testPerpixelLighting.py               # Test perpixel lighting prediction. 
     python testFull.py                           # Test the whole neural renderer with predicted light sources. Results in Table 4 in the main paper. 
     ```
    
## Scene editing applications on real images
