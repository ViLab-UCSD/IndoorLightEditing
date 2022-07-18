import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class renderDirecLighting:
    def __init__(self,
            fov=57.95,
            isCuda = True, sampleNum=10 ):
        self.fov = fov / 180.0 * np.pi
        self.isCuda = isCuda
        self.sampleNum = sampleNum
        self.F0 = 0.05

    def depthToPoint(self, depth ):
        height, width = depth.size(2), depth.size(3)
        xRange = 1 * np.tan(self.fov / 2 )
        yRange = float(height) / float(width) * xRange

        x, y = np.meshgrid(np.linspace(-xRange, xRange, width ),
                np.linspace(-yRange, yRange, height ) )

        y = np.flip(y, axis=0 )
        z = -np.ones( (height, width), dtype=np.float32 )

        pCoord = np.stack([x, y, z], axis = 0 )[np.newaxis, :]
        pCoord = pCoord.astype(np.float32 )
        pCoord = torch.from_numpy(pCoord )
        if self.isCuda:
            pCoord = pCoord.cuda()
        point = pCoord * depth

        return point

    def boxToSurface(self, axes, center ):
        centers = []
        normals = []
        xAxes = []
        yAxes = []

        centers.append( (center + axes[:, 0, :] * 0.5 ).unsqueeze(-1) )
        centers.append( (center - axes[:, 0, :] * 0.5 ).unsqueeze(-1) )
        centers.append( (center + axes[:, 1, :] * 0.5 ).unsqueeze(-1) )
        centers.append( (center - axes[:, 1, :] * 0.5 ).unsqueeze(-1) )
        centers.append( (center + axes[:, 2, :] * 0.5 ).unsqueeze(-1) )
        centers.append( (center - axes[:, 2, :] * 0.5 ).unsqueeze(-1) )
        centers = torch.cat(centers, dim=-1 )
        centers = centers.unsqueeze(-2 ).unsqueeze(-1 )

        normals.append( axes[:, 0, :].unsqueeze(-1 ) )
        normals.append( -axes[:, 0, :].unsqueeze(-1 ) )
        normals.append( axes[:, 1, :].unsqueeze(-1 ) )
        normals.append( -axes[:, 1, :].unsqueeze(-1 ) )
        normals.append( axes[:, 2, :].unsqueeze(-1 ) )
        normals.append( -axes[:, 2, :].unsqueeze(-1 ) )
        normals = torch.cat(normals, dim=-1 )
        normals = normals / torch.sqrt(
                torch.clamp(
                    torch.sum(normals * normals, dim=1, keepdim=True ),
                    min=1e-12
                    )
                )
        normals = normals.unsqueeze(-2 ).unsqueeze(-1 )

        xAxes.append(axes[:, 1, :].unsqueeze(-1) )
        xAxes.append(axes[:, 1, :].unsqueeze(-1) )
        xAxes.append(axes[:, 2, :].unsqueeze(-1) )
        xAxes.append(axes[:, 2, :].unsqueeze(-1) )
        xAxes.append(axes[:, 0, :].unsqueeze(-1) )
        xAxes.append(axes[:, 0, :].unsqueeze(-1) )
        xAxes = torch.cat(xAxes, dim=-1 )
        xAxes = xAxes.unsqueeze(-2 ).unsqueeze(-1 )

        yAxes.append(axes[:, 2, :].unsqueeze(-1) )
        yAxes.append(axes[:, 2, :].unsqueeze(-1) )
        yAxes.append(axes[:, 0, :].unsqueeze(-1) )
        yAxes.append(axes[:, 0, :].unsqueeze(-1) )
        yAxes.append(axes[:, 1, :].unsqueeze(-1) )
        yAxes.append(axes[:, 1, :].unsqueeze(-1) )
        yAxes = torch.cat(yAxes, dim=-1 )
        yAxes = yAxes.unsqueeze(-2 ).unsqueeze(-1 )

        return centers, normals, xAxes, yAxes

    def forward(
            self,
            lpt_axes, lpt_center,
            lpt_int,
            depth,
            pts_normal,
            isTest,
            pts_rough = None ):

        lpt_int = lpt_int.unsqueeze(-1 ).unsqueeze(-1 ).unsqueeze(-1)
        lpts, lpts_normal, lpts_xAxis, lpts_yAxis \
                = self.boxToSurface(lpt_axes, lpt_center )
        lpts_xAxisNorm = torch.sqrt(torch.clamp(
            torch.sum(lpts_xAxis * lpts_xAxis, dim=1, keepdim=True ), min=1e-12 ) )
        lpts_yAxisNorm = torch.sqrt(torch.clamp(
            torch.sum(lpts_yAxis * lpts_yAxis, dim=1, keepdim=True ), min=1e-12 ) )
        # Build the x and y axis for every place
        # Build the x and y axis for every place
        # pts:          batchSize x 3 x N x 1
        # pts_normal:   batchSize x 3 x N x 1

        # lpts:         batchSize x 3 x 1 x 6 x 1
        # lpts_normal:  batchSize x 3 x 1 x 6 x 1
        # lpts_xAxis:   batchSize x 3 x 1 x 6 x 1
        # lpts_yAxis:   batchSize x 3 x 1 x 6 x 1

        # lpt_int:      batchSize x 3 x 1 x 1 x 1
        pts = self.depthToPoint(depth )
        bn = pts.size(0 )
        height, width = pts.size(2), pts.size(3)
        N = width * height
        pts = pts.view(bn, 3, N, 1, 1)
        pts_normal = pts_normal.view(bn, 3, N, 1, 1)
        if not pts_rough is None:
            pts_rough = pts_rough.view(bn, 1, N, 1, 1 )

        # Sample Area
        seedArea = np.random.random( [bn, 2, N, 1, self.sampleNum ] ).astype(dtype = np.float32 ) - 0.5
        seedArea_x = torch.from_numpy(seedArea[:, 0:1, :] )
        seedArea_y = torch.from_numpy(seedArea[:, 1:2, :] )
        if self.isCuda:
            seedArea_x = seedArea_x.cuda()
            seedArea_y = seedArea_y.cuda()
        lpt_sampled = lpts + seedArea_x * lpts_xAxis + seedArea_y * lpts_yAxis

        pts_dir = lpt_sampled - pts
        pts_distL2 = torch.clamp(torch.sum(pts_dir * pts_dir, dim=1, keepdim=True ), min=1e-12 )
        pts_dir = pts_dir / torch.sqrt(pts_distL2 )

        pts_cos = torch.sum(pts_dir * pts_normal, dim=1, keepdim=True )
        lpt_cos = torch.clamp(torch.sum(pts_dir * lpts_normal, dim=1, keepdim=True ), -1, 1 )

        if isTest:
            pts_int = lpt_int * torch.clamp(pts_cos, min=0, max=1 ) \
                    * torch.clamp(lpt_cos, min=0, max=1 )
        else:
            pts_int = lpt_int * torch.clamp(pts_cos, min=0, max=1 ) \
                    * lpt_cos.abs()

        # Compute the possiblity of area light
        prob_area = (1 / lpts_xAxisNorm / lpts_yAxisNorm * pts_distL2 ).detach()

        pts_shading = pts_int / torch.clamp(prob_area, min=1e-12 )

        pts_shading = torch.sum(torch.mean(pts_shading, dim=-1), dim=-1)
        pts_shading = pts_shading.view(bn, 3, height, width )

        if not pts_rough is None:
            alpha = pts_rough * pts_rough
            k = (pts_rough + 1 ) * (pts_rough + 1 ) / 8.0
            alpha2 = alpha * alpha

            v = -pts
            v = v / torch.sqrt(torch.clamp(torch.sum(v * v, dim=1), min=1e-6 ) )
            l = pts_dir
            h = (l + v) / 2.0
            h = h / torch.sqrt(torch.clamp(torch.sum(h * h, dim=1, keepdim=True ), min=1e-6) )
            vdh = torch.sum(v * h, dim=1, keepdim=True )

            temp = (torch.zeros([1, 1, 1, 1, 1], dtype=torch.float32 ) + 2.0 )
            if self.isCuda:
                temp = temp.cuda()
            frac0 = self.F0 + (1 - self.F0) * torch.pow(temp, (-5.55472*vdh - 6.98326 )*vdh )

            ndv = torch.clamp(torch.sum(pts_normal * v, dim=1, keepdim=True ), 0, 1 )
            ndh = torch.clamp(torch.sum(pts_normal * h, dim=1, keepdim=True ), 0, 1 )
            ndl = torch.clamp(torch.sum(pts_normal * l, dim=1, keepdim=True ), 0, 1 )

            frac = alpha2 * frac0
            nom0 = ndh * ndh * (alpha2 -1 ) + 1
            nom1 = ndv * (1 - k) + k
            nom2 = ndl * (1 - k) + k
            nom = torch.clamp(4*np.pi * nom0*nom0*nom1*nom2, 1e-6, 4*np.pi )
            pts_specular = frac / nom * pts_int  / prob_area
            pts_specular = torch.mean(torch.mean(pts_specular, dim=-1 ), dim=-1 )

            pts_specular = pts_specular.view(bn, 3, height, width )

            return pts_shading, pts_specular
        else:
            return pts_shading


if __name__ == '__main__':
    import pickle
    import cv2
    import struct
    import scipy.ndimage as ndimage

    fov = 57.95 / 180.0 * np.pi

    with open('box0.dat', 'rb') as fIn:
        lightBox = pickle.load(fIn )

    with open('light0.dat', 'rb') as fIn:
        lightSrc = pickle.load(fIn )

    center = lightBox['box3D']['center']
    center = center.reshape(1, 3 ).astype(np.float32 )
    center = torch.from_numpy(center )

    xAxis = lightBox['box3D']['xAxis'] * lightBox['box3D']['xLen']
    yAxis = lightBox['box3D']['yAxis'] * lightBox['box3D']['yLen']
    zAxis = lightBox['box3D']['zAxis'] * lightBox['box3D']['zLen']

    xAxis = xAxis.astype(np.float32 ).reshape(1, 1, 3)
    yAxis = yAxis.astype(np.float32 ).reshape(1, 1, 3)
    zAxis = zAxis.astype(np.float32 ).reshape(1, 1, 3)
    axes = np.concatenate([xAxis, yAxis, zAxis], axis=1 )
    axes = torch.from_numpy(axes )

    intensity = lightSrc['intensity']
    intensity = intensity.reshape(1, 3).astype(np.float32 )
    intensity = torch.from_numpy(intensity )

    height, width = 120, 160

    normal = cv2.imread('imnormal_4.png' )[:, :, ::-1]
    normal = np.ascontiguousarray(normal )
    normal = cv2.resize(normal,(width, height), interpolation = cv2.INTER_AREA )

    normal = normal.astype(np.float32 )
    normal = normal / 127.5 - 1
    normal = normal.transpose(2, 0, 1)
    normal = normal / np.sqrt(
            np.maximum(np.sum(normal * normal, axis=0, keepdims=True ), 1e-12 ) )

    normal = normal.reshape(1, 3, height, width )
    normal = torch.from_numpy(normal )

    mask = cv2.imread('mask0.png' )[:, :, 0]
    mask = cv2.resize(mask, (width, height), interpolation = cv2.INTER_AREA )
    mask = (mask == 255)
    mask = ndimage.binary_erosion(mask, structure = np.ones((3, 3) ) )

    mask = mask.astype(np.float32 )

    mask = mask.reshape(1, 1, height, width )
    mask = torch.from_numpy(mask )

    with open('imdepth_4.dat', 'rb') as fIn:
        hBuffer = fIn.read(4 )
        dh = struct.unpack('i', hBuffer )[0]
        wBuffer = fIn.read(4 )
        dw = struct.unpack('i', wBuffer )[0]

        dBuffer = fIn.read()
        depth = struct.unpack('f' * dh * dw, dBuffer )
        depth = np.array(depth ).reshape(dh, dw ).astype(np.float32 )

    depth = cv2.resize(depth, (width, height), interpolation = cv2.INTER_AREA )

    depth = depth.reshape(1, 1, height, width )
    depth = torch.from_numpy(depth )

    renderer = renderDirecLighting(isCuda = False )
    shading = renderer.forward(axes, center, intensity,
            depth, normal, isTest=False )
    shading = shading.numpy().reshape(3, height, width )
    shading = shading.transpose(1, 2, 0)[:, :, ::-1]
    cv2.imwrite('shading.hdr', shading )


    # Visualize the geometry
    import open3d as o3d

    normal = normal.numpy()
    depth = depth.numpy()
    mask = mask.numpy()

    mask = mask.reshape(-1)

    xRange = 1 * np.tan(fov / 2 )
    yRange = float(height) / float(width) * xRange

    x, y = np.meshgrid(np.linspace(-xRange, xRange, width ),
            np.linspace(-yRange, yRange, height ) )

    y = np.flip(y, axis=0 )
    z = -np.ones( (height, width), dtype=np.float32 )

    pCoord = np.stack([x, y, z], axis = 0 )[np.newaxis, :]
    pCoord = pCoord.astype(np.float32 )
    pCoord = pCoord * depth

    pCoord = pCoord.reshape(3, width * height ).transpose(1, 0)
    normal = normal.reshape(3, width * height ).transpose(1, 0)
    normal = (normal + 1) * 0.5

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pCoord.astype(np.float32 )[mask == 0, :] )
    pcd.colors = o3d.utility.Vector3dVector(normal.astype(np.float32 )[mask == 0, :] )
    o3d.io.write_point_cloud('room.ply', pcd )

    center = center.numpy().squeeze()
    xAxis = xAxis.squeeze()
    yAxis = yAxis.squeeze()
    zAxis = zAxis.squeeze()

    corner_11 = center - yAxis / 2.0 - xAxis / 2.0 - zAxis / 2.0
    corner_12 = center - yAxis / 2.0 - xAxis / 2.0 + zAxis / 2.0
    corner_13 = center - yAxis / 2.0 + xAxis / 2.0 + zAxis / 2.0
    corner_14 = center - yAxis / 2.0 + xAxis / 2.0 - zAxis / 2.0

    corner_21 = center + yAxis / 2.0 - xAxis / 2.0 - zAxis / 2.0
    corner_22 = center + yAxis / 2.0 - xAxis / 2.0 + zAxis / 2.0
    corner_23 = center + yAxis / 2.0 + xAxis / 2.0 + zAxis / 2.0
    corner_24 = center + yAxis / 2.0 + xAxis / 2.0 - zAxis / 2.0

    vertices = []
    vertices.append(corner_11 )
    vertices.append(corner_12 )
    vertices.append(corner_13 )
    vertices.append(corner_14 )

    vertices.append(corner_21 )
    vertices.append(corner_22 )
    vertices.append(corner_23 )
    vertices.append(corner_24 )

    faces = []
    faces.append(np.array([0, 3, 2], dtype = np.int32 ) )
    faces.append(np.array([0, 2, 1], dtype = np.int32 ) )

    faces.append(np.array([0, 4, 1], dtype = np.int32 ) )
    faces.append(np.array([1, 4, 5], dtype = np.int32 ) )
    faces.append(np.array([0, 4, 3], dtype = np.int32 ) )
    faces.append(np.array([3, 4, 7], dtype = np.int32 ) )
    faces.append(np.array([2, 6, 1], dtype = np.int32 ) )
    faces.append(np.array([1, 6, 5], dtype = np.int32 ) )
    faces.append(np.array([2, 6, 3], dtype = np.int32 ) )
    faces.append(np.array([3, 6, 7], dtype = np.int32 ) )

    faces.append(np.array([4, 6, 7], dtype = np.int32 ) )
    faces.append(np.array([4, 5, 6], dtype = np.int32 ) )

    vertices = np.stack(vertices, axis=0 )
    faces = np.stack(faces, axis=0 )

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices )
    mesh.triangles = o3d.utility.Vector3iVector(faces )
    o3d.io.write_triangle_mesh('lampBox.ply', mesh )
