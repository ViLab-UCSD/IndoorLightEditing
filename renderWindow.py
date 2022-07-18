import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class renderDirecLighting:
    def __init__(self,
            fov=57.95,
            sampleNum = 180,
            isCuda = True ):

        self.fov = fov / 180.0 * np.pi
        self.sampleNum = sampleNum
        self.isCuda = isCuda
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


    def forward(self,
                pl_center,
                pl_normal,
                pl_xAxis,
                pl_yAxis,
                pl_para,
                pl_paraSky,
                pl_paraGrd,
                depth,
                pts_normal, pts_rough = None ):

        pl_int, pl_direc, pl_lamb = torch.split( pl_para, [3, 3, 1 ], dim=1 )
        pl_int = pl_int.unsqueeze(-1 ).unsqueeze(-1)
        pl_direc = pl_direc.unsqueeze(-1 ).unsqueeze(-1)
        pl_lamb = pl_lamb.unsqueeze(-1 ).unsqueeze(-1)

        pl_intSky, pl_direcSky, pl_lambSky = torch.split( pl_paraSky, [3, 3, 1 ], dim=1 )
        pl_intSky = pl_intSky.unsqueeze(-1 ).unsqueeze(-1)
        pl_direcSky = pl_direcSky.unsqueeze(-1 ).unsqueeze(-1)
        pl_lambSky = pl_lambSky.unsqueeze(-1 ).unsqueeze(-1)

        pl_intGrd, pl_direcGrd, pl_lambGrd = torch.split( pl_paraGrd, [3, 3, 1 ], dim=1 )
        pl_intGrd = pl_intGrd.unsqueeze(-1 ).unsqueeze(-1)
        pl_direcGrd = pl_direcGrd.unsqueeze(-1 ).unsqueeze(-1)
        pl_lambGrd = pl_lambGrd.unsqueeze(-1 ).unsqueeze(-1)

        pl_center = pl_center.unsqueeze(-1 ).unsqueeze(-1)
        pl_normal = pl_normal.unsqueeze(-1 ).unsqueeze(-1)
        pl_xAxis = pl_xAxis.unsqueeze(-1 ).unsqueeze(-1) * 0.5
        pl_yAxis = pl_yAxis.unsqueeze(-1 ).unsqueeze(-1) * 0.5
        # pts:          batchSize x 3 x N  x 1
        # pts_envMask:  batchSize x 1 x N x 1
        # pts_normal:   batchSize x 3 x N x 1

        # pl_center:    batchSize x 3 x 1 x 1
        # pl_normal:    batchSize x 3 x 1 x 1
        # pl_xAxis:     batchSize x 3 x 1 x 1
        # pl_yAxis      batchSize x 3 x 1 x 1

        # pl_int:       batchSize x 3 x 1 x 1
        # pl_direc:     batchSize x 3 x 1 x 1
        # pl_lamb:      batchSize x 1 x 1 x 1
        pts = self.depthToPoint(depth )
        bn = pts.size(0 )
        height, width = pts.size(2), pts.size(3)
        N = width * height
        pts = pts.view(bn, 3, N, 1 )
        pts_normal = pts_normal.view(bn, 3, N, 1 )
        if not pts_rough is None:
            pts_rough = pts_rough.view(bn, 1, N, 1 )

        up = np.array([0, 1, 0], dtype=np.float32 ).reshape([1, 3, 1, 1] )
        up = torch.from_numpy(up )

        right = np.array([0, 0, 1], dtype = np.float32 ).reshape([1, 3, 1, 1] )
        right = torch.from_numpy(right )

        if self.isCuda:
            up = up.cuda()
            right = right.cuda()

        xAxisNorm = torch.sqrt(torch.clamp(torch.sum(pl_xAxis * pl_xAxis, dim=1, keepdim=True ), min=1e-12 ) )
        yAxisNorm = torch.sqrt(torch.clamp(torch.sum(pl_yAxis * pl_yAxis, dim=1, keepdim=True ), min=1e-12 ) )
        axis_x = pl_xAxis / xAxisNorm
        axis_y = pl_yAxis / yAxisNorm

        dAxis1 = torch.cross(pl_direc, up.expand_as(pl_direc ), dim=1 )
        dAxis2 = torch.cross(pl_direc, right.expand_as(pl_direc ), dim=1 )

        dAxis1_norm = torch.sum(dAxis1 * dAxis1, dim=1, keepdim=True )
        dAxis2_norm = torch.sum(dAxis2 * dAxis2, dim=1, keepdim=True )
        dAxisInd = (dAxis1_norm > dAxis2_norm ).float()

        dAxis_x = dAxisInd * dAxis1 + (1 - dAxisInd ) * dAxis2
        dAxis_y = torch.cross(pl_direc, dAxis_x, dim=1 )

        dAxis_x = dAxis_x / torch.sqrt(
                torch.clamp(torch.sum(dAxis_x * dAxis_x, dim=1, keepdim=True ), min=1e-12) )
        dAxis_y = dAxis_y / torch.sqrt(
                torch.clamp(torch.sum(dAxis_y * dAxis_y, dim=1, keepdim=True ), min=1e-12) )

        prob_light_coef = pl_lamb / np.pi / 2 / (1 - torch.exp(-2 * pl_lamb ) )
        prob_area_coef = 1 / xAxisNorm / yAxisNorm / 4

        # Sample area
        seedArea = np.random.random( [bn, 2, N, self.sampleNum ] ).astype(dtype = np.float32 )
        seedArea_x = torch.from_numpy(seedArea[:, 0:1, :] ) * 2 - 1
        seedArea_y = torch.from_numpy(seedArea[:, 1:2, :] ) * 2 - 1
        if self.isCuda:
            seedArea_x = seedArea_x.cuda()
            seedArea_y = seedArea_y.cuda()
        seedArea_x = seedArea_x * pl_xAxis
        seedArea_y = seedArea_y * pl_yAxis
        lpt_sampled = seedArea_x + seedArea_y + pl_center

        pts_dir = lpt_sampled - pts
        pts_distL2 = torch.clamp(torch.sum(pts_dir * pts_dir, dim=1, keepdim=True ), min=1e-12 )
        pts_dir = pts_dir / torch.sqrt(pts_distL2 )

        pts_dir_reverse_ind = (torch.sum(pts_dir * pl_normal, dim=1, keepdim=True ) < 0).float()
        pts_dir = (1 - pts_dir_reverse_ind) * pts_dir - pts_dir_reverse_ind * pts_dir
        pts_dir = pts_dir.detach()

        pts_cos = torch.sum(pts_dir * pts_normal, dim=1, keepdim=True )
        lpt_cos = torch.abs(torch.sum(pts_dir * pl_normal, dim=1, keepdim=True ) )

        pts_int = pl_int * torch.exp(pl_lamb *
                torch.clamp(torch.sum(pl_direc * pts_dir, dim=1, keepdim=True ) - 1, max=0 ) )
        pts_int = pts_int * torch.clamp(pts_cos, 0, 1)

        pts_intSky = pl_intSky * torch.exp(pl_lambSky *
                torch.clamp(torch.sum(pl_direcSky * pts_dir, dim=1, keepdim=True ) - 1, max=0 ) )
        pts_intSky = pts_intSky * torch.clamp(pts_cos, 0, 1)

        pts_intGrd = pl_intGrd * torch.exp(pl_lambGrd *
                torch.clamp(torch.sum(pl_direcGrd * pts_dir, dim=1, keepdim=True ) - 1, max=0 ) )
        pts_intGrd = pts_intGrd * torch.clamp(pts_cos, 0, 1)

        # Compute the possiblity of area light
        prob_area_1 = ( prob_area_coef * pts_distL2 / torch.clamp(lpt_cos, min=1e-12 ) ).detach()
        prob_light_1 = ( prob_light_coef * torch.exp(pl_lamb *
                torch.clamp(torch.sum(pl_direc * pts_dir, dim=1, keepdim=True ) - 1, max=0 ) ) ).detach()

        # Sample SG
        seedDirec = np.random.random( [bn, 2, N, self.sampleNum ] ).astype(dtype = np.float32 )
        seedDirec_theta = torch.from_numpy(seedDirec[:, 0:1, :] )
        seedDirec_phi = torch.from_numpy(seedDirec[:, 1:2, :] )
        if self.isCuda:
            seedDirec_theta = seedDirec_theta.cuda()
            seedDirec_phi = seedDirec_phi.cuda()

        seedDirec_theta =  1 / torch.clamp(pl_lamb, min=1e-14 ) \
                * torch.log( torch.clamp(
                    1 - (1 - torch.exp(-2 * pl_lamb ) ) * seedDirec_theta,
                    min = 1e-14 ) ) + 1
        seedDirec_theta = torch.acos(torch.clamp(seedDirec_theta, min=-(1-1e-14), max=(1-1e-14 ) ) )
        seedDirec_phi = seedDirec_phi * np.pi * 2
        seedDirec_theta = seedDirec_theta
        seedDirec_phi = seedDirec_phi

        pts_dir_d = torch.sin(seedDirec_theta ) * torch.cos(seedDirec_phi ) * dAxis_x \
            + torch.sin(seedDirec_theta ) * torch.sin(seedDirec_phi ) * dAxis_y \
            + torch.cos(seedDirec_theta ) * pl_direc
        pts_dir_d = pts_dir_d.detach()
        lpt_cos_d = torch.sum(pts_dir_d * pl_normal, dim=1, keepdim=True )
        lpt_cos_mask = (lpt_cos_d > 0).float().detach()
        lpt_cos_d = torch.abs(lpt_cos_d )

        # Compute the intersection point
        t  = (torch.sum(pl_normal * (pl_center - pts), dim=1, keepdim=True) )  \
            / torch.clamp(
                torch.abs(
                    torch.sum(pl_normal * pts_dir_d, dim=1, keepdim=True )
                ),
                min=1e-12
            )
        t_reverse_ind = (torch.sum(pl_normal * pts_dir_d, dim=1, keepdim=True ) < 0).float()
        t = t * (1 - t_reverse_ind ) - t * t_reverse_ind
        t = t.detach()

        pl_intersect = pts + pts_dir_d * t
        pl_intersect_mask = torch.sum( (pl_intersect - pl_center) * pl_normal, dim=1, keepdim=True ).abs()
        pl_intersect_mask = (pl_intersect_mask < 1e-3 ).float()

        # Compute the intensity of the pixels
        pts_cos_d = torch.sum(pts_dir_d * pts_normal, dim=1, keepdim=True )
        lpt_interdist_x = torch.sum((pl_intersect - pl_center) * axis_x, dim=1, keepdim=True )
        lpt_interdist_y = torch.sum((pl_intersect - pl_center) * axis_y, dim=1, keepdim=True )
        lpt_interdist_x = torch.abs(lpt_interdist_x ) - xAxisNorm
        lpt_interdist_y = torch.abs(lpt_interdist_y ) - yAxisNorm
        lpt_coef = torch.sigmoid(-100 * lpt_interdist_x) * torch.sigmoid(-100 * lpt_interdist_y )
        lpt_coef = lpt_coef * pl_intersect_mask
        lpt_coef = lpt_coef.detach()

        pts_int_d = pl_int * torch.exp(pl_lamb *
            torch.clamp(torch.sum(pl_direc * pts_dir_d, dim=1, keepdim=True ) - 1, max=0 ) )
        pts_int_d = (lpt_coef * pts_int_d) * (t > 0).float() * torch.clamp(pts_cos_d, min=0, max=1) * lpt_cos_mask

        # Compute the probibility
        prob_light_2 = ( prob_light_coef * torch.exp(pl_lamb *
            torch.clamp(torch.sum(pl_direc * pts_dir_d, dim=1, keepdim=True ) - 1, max=0 ) ) ).detach()
        prob_area_2 = ( prob_area_coef * t * t / torch.clamp(lpt_cos_d, min=1e-12 ) ).detach()

        pts_intAll = (pts_intSky + pts_intGrd) / torch.clamp(prob_area_1, min=1e-12 ) \
                + pts_int * prob_area_1 / torch.clamp(prob_area_1 * prob_area_1 + prob_light_1 * prob_light_1, 1e-12 ) \
                + pts_int_d * prob_light_2 / torch.clamp(prob_area_2 * prob_area_2 + prob_light_2 * prob_light_2, 1e-12 )

        pts_shading = torch.mean(pts_intAll, dim=-1 )
        pts_shading = pts_shading.view(bn, 3, height, width )

        if not pts_rough is None:
            alpha = pts_rough * pts_rough
            k = (pts_rough + 1 ) * (pts_rough + 1 ) / 8.0
            alpha2 = alpha * alpha

            v = -pts
            v = v / torch.sqrt(torch.clamp(torch.sum(v * v, dim=1), min=1e-12 ) )
            l = pts_dir
            h = (l + v) / 2.0
            h = h / torch.sqrt(torch.clamp(torch.sum(h * h, dim=1, keepdim=True ), min=1e-12 ) )
            vdh = torch.sum(v * h, dim=1, keepdim=True )

            temp = torch.zeros([1, 1, 1, 1], dtype=torch.float32 ) + 2.0
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
            pts_specular = frac / nom * pts_intAll
            pts_specular = torch.mean(pts_specular, dim=-1 )

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

    with open('box1.dat', 'rb') as fIn:
        lightBox = pickle.load(fIn )
    lightBox = lightBox['box3D']
    center = lightBox['center'].reshape(1, 3)
    xAxis = (lightBox['xAxis'] * lightBox['xLen'] ).reshape(1, 3)
    yAxis = (lightBox['yAxis'] * lightBox['yLen'] ).reshape(1, 3)
    zAxisNormalized = lightBox['zAxis'].reshape(1, 3)
    zAxis = (lightBox['zAxis'] * lightBox['zLen'] ).reshape(1, 3)
    center = center - zAxis * 0.6

    center = torch.from_numpy(center.astype(np.float32 ) )
    zAxisNormalized = torch.from_numpy(zAxisNormalized.astype(np.float32 ) )
    xAxis = torch.from_numpy(xAxis.astype(np.float32 ) )
    yAxis = torch.from_numpy(yAxis.astype(np.float32 ) )

    with open('3SG_log2.dat', 'rb') as fIn:
        lightSrc = pickle.load(fIn )

    intensity = lightSrc['intensity'].reshape(1, 3)
    direction = lightSrc['axis'].reshape(1, 3)
    lamb = lightSrc['lamb'].reshape(1, 1)

    intensitySky = lightSrc['intensitySky'].reshape(1, 3)
    directionSky = lightSrc['axisSky'].reshape(1, 3)
    lambSky = lightSrc['lambSky'].reshape(1, 1)

    intensityGrd = lightSrc['intensityGrd'].reshape(1, 3)
    directionGrd = lightSrc['axisGrd'].reshape(1, 3)
    lambGrd = lightSrc['lambGrd'].reshape(1, 1)

    paras = np.concatenate([intensity, direction, lamb], axis=1 )
    paras = torch.from_numpy(paras.astype(np.float32 ) )

    parasSky = np.concatenate([intensitySky, directionSky, lambSky ], axis=1 )
    parasSky = torch.from_numpy(parasSky.astype(np.float32 ) )

    parasGrd = np.concatenate([intensityGrd, directionGrd, lambGrd ], axis=1 )
    parasGrd = torch.from_numpy(parasGrd.astype(np.float32 ) )

    height, width = 120, 160

    pixel_len = np.tan(fov / 2) / width * 2.0
    pixel_len = pixel_len * np.abs(center.numpy().squeeze()[2] )

    normal = cv2.imread('imnormal_1.png' )[:, :, ::-1]
    normal = np.ascontiguousarray(normal )
    normal = cv2.resize(normal,(width, height), interpolation = cv2.INTER_AREA )

    normal = normal.astype(np.float32 )
    normal = normal / 127.5 - 1
    normal = normal.transpose(2, 0, 1)
    normal = normal / np.sqrt(
            np.maximum(np.sum(normal * normal, axis=0, keepdims=True ), 1e-12 ) )

    normal = normal.reshape(1, 3, height, width )
    normal = torch.from_numpy(normal )

    mask = cv2.imread('mask1.png' )[:, :, 0]
    mask = cv2.resize(mask, (width, height), interpolation = cv2.INTER_AREA )
    mask = (mask == 255)
    mask = ndimage.binary_erosion(mask, structure = np.ones((3, 3) ) )

    mask = mask.astype(np.float32 )

    mask = mask.reshape(1, 1, height, width )
    mask = torch.from_numpy(mask )


    with open('imdepth_1.dat', 'rb') as fIn:
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

    renderer = renderDirecLighting(isCuda = False, sampleNum = 90 )
    shading = renderer.forward(center, zAxisNormalized,
            xAxis, yAxis, paras, parasSky, parasGrd, depth, normal )

    shading = shading.numpy().reshape(3, height, width )
    shading = shading.transpose(1, 2, 0)[:, :, ::-1 ]
    cv2.imwrite('shading3SG_MIS.hdr', shading )

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

    vertices, faces = [], []
    v1 = center - xAxis * 0.5 - yAxis * 0.5
    v2 = center - xAxis * 0.5 + yAxis * 0.5
    v3 = center + xAxis * 0.5 + yAxis * 0.5
    v4 = center + xAxis * 0.5 - yAxis * 0.5
    vertices.append(v1.squeeze() )
    vertices.append(v2.squeeze() )
    vertices.append(v3.squeeze() )
    vertices.append(v4.squeeze() )
    vertices = np.stack(vertices, axis=0 ).astype(np.float64 )

    faces.append(np.array([0, 1, 2], dtype = np.int32 ) )
    faces.append(np.array([0, 2, 3], dtype = np.int32 ) )
    faces = np.stack(faces, axis=0 )

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices )
    mesh.triangles = o3d.utility.Vector3iVector(faces )
    o3d.io.write_triangle_mesh('winPlane.ply', mesh )
