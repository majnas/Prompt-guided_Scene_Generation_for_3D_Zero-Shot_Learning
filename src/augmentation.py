import numpy as np
import math
import random
import os
import torch
import torch.nn as nn
from torchvision import transforms

class normalize_data(object):
    def __call__(self,sample:dict) -> dict:
        """ Normalize the batch data, use coordinates of the block centered at origin,
            Input:
                NxC array
            Output:
                NxC array
        """
        batch_data = sample["points"] 
        N, C = batch_data.shape
        normal_data = np.zeros((N, C))  
        centroid = np.mean(batch_data, axis=0)
        batch_data = batch_data - centroid
        m = np.max(np.sqrt(np.sum(batch_data ** 2, axis=1)))
        batch_data = batch_data / m
        sample["points"]=batch_data
        return sample
   

class shuffle_points(object):
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: dict) -> dict:
      """ Shuffle orders of points in each point cloud -- changes FPS behavior.
          Use the same shuffling idx for the entire batch.
          Input:
              sample["points"] -> NxC array
          Output:
              sample["points"] -> NxC array
      """
      if self.p > random.random():
        batch_data = sample["points"]
        idx = np.arange(batch_data.shape[0])
        np.random.shuffle(idx)
        sample["points"] = batch_data[idx,:]
      return sample


#####
class rotate_point_cloud(object):
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self,sample:dict) -> dict:
        """ Randomly rotate the point clouds to augument the dataset
            rotation is per shape based along up direction
            Input:
                sample["points"] -> NxC array
            Output:
                sample["points"] -> NxC array
        """
        if self.p > random.random():
            batch_data = sample["points"] 
            rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]])
            # handle standard + normalized data
            rotated_data[:,:3] = np.dot(batch_data[:,:3], rotation_matrix)
            if batch_data.shape[1] == 6:
                rotated_data[:,3:6] = np.dot(batch_data[:,3:6], rotation_matrix)
            sample["points"] = rotated_data
        return sample

class rotate_point_cloud_z(object):
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self,sample) ->dict:
        """ Randomly rotate the point clouds to augument the dataset
            rotation is per shape based along up direction
            Input:
                Nx3 array, original  point clouds
            Return:
                Nx3 array, rotated  point clouds
        """
        if self.p > random.random():
            batch_data = sample["points"]
            rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, sinval, 0],
                                        [-sinval, cosval, 0],
                                        [0, 0, 1]])
            # handle standard + normalized data
            rotated_data[:,:3] = np.dot(batch_data[:,:3], rotation_matrix)
            if batch_data.shape[1] == 6:
                rotated_data[:,3:6] = np.dot(batch_data[:,3:6], rotation_matrix)
            sample["points"]=rotated_data
        return sample  

class  rotate_point_cloud_with_normal(object):
    def __call__(self,sample):
        ''' Randomly rotate XYZ, normal point cloud.
            Input:
                batch_xyz_normal: N,6, first three channels are XYZ, last 3 all normal
            Output:
                N,6, rotated XYZ, normal point cloud
        '''
        batch_xyz_normal=sample["points"]
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_xyz_normal[:,0:3]
        shape_normal = batch_xyz_normal[:,3:6]
        batch_xyz_normal[:,0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        batch_xyz_normal[:,3:6] = np.dot(shape_normal.reshape((-1, 3)), rotation_matrix)
        sample["points"]=batch_xyz_normal
        return sample

class  rotate_perturbation_point_cloud_with_normal(object):
    def __init__(self,angle_sigma=0.06, angle_clip=0.18):
        self.angle_sigma=angle_sigma
        self.angle_clip=angle_clip
    def __call__(self,sample):
        """ Randomly perturb the point clouds by small rotations
            Input:
                Nx6 array, original batch o point clouds and point normals
            Return:
                Nx3 array, rotated  point clouds
        """
        batch_data=sample["points"]
        rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
        angles = np.clip(self.angle_sigma*np.random.randn(3), -self.angle_clip, self.angle_clip)
        Rx = np.array([[1,0,0],
                        [0,np.cos(angles[0]),-np.sin(angles[0])],
                        [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                        [0,1,0],
                        [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                        [np.sin(angles[2]),np.cos(angles[2]),0],
                        [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        shape_pc = batch_data[:,0:3]
        shape_normal = batch_data[:,3:6]
        rotated_data[:,0:3] = np.dot(shape_pc.reshape((-1, 3)), R)
        rotated_data[:,3:6] = np.dot(shape_normal.reshape((-1, 3)), R)
        sample["points"]=rotated_data
        return sample

class rotate_point_cloud_by_angle(object):
    def __init__(self,rotation_angle):
        self.rotation_angle=rotation_angle

    def __call__(self,sample):
        """ Rotate the point cloud along up direction with certain angle.
            Input:
                Nx3 array, original  point clouds
            Return:
                Nx3 array, rotated  point clouds
        """
        batch_data=sample["points"]
        rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
        cosval = np.cos(self.rotation_angle)
        sinval = np.sin(self.rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        
        rotated_data= np.dot(batch_data, rotation_matrix)
        sample["points"]=rotated_data
        return sample
class rotate_point_cloud_by_angle_with_normal(object):
    def __init__(self,rotation_angle):
        self.rotation_angle=rotation_angle
    def __call__(self,sample):
        """ Rotate the point cloud along up direction with certain angle.
            Input:
                Nx6 array, original point clouds with normal
                scalar, angle of rotation
            Return:
                Nx6 array, rotated point clouds iwth normal
        """
        batch_data=sample["points"]
        rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
        #self.rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(self.rotation_angle)
        sinval = np.sin(self.rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[:,0:3]
        shape_normal = batch_data[:,3:6]
        rotated_data[:,0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        rotated_data[:,3:6] = np.dot(shape_normal.reshape((-1,3)), rotation_matrix)
        sample["points"]=rotated_data
        return sample

class rotate_perturbation_point_cloud(object):
    def __init__(self,angle_sigma=0.06, angle_clip=0.18):
        self.angle_sigma=angle_sigma
        self.angle_clip=angle_clip
    def __call__(self,sample):
        """ Randomly perturb the point clouds by small rotations
            Input:
                Nx3 array, original point clouds
            Return:
                Nx3 array, rotated  point clouds
        """
        batch_data=sample["points"]
        rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
        angles = np.clip(self.angle_sigma*np.random.randn(3), -self.angle_clip, self.angle_clip)
        Rx = np.array([[1,0,0],
                        [0,np.cos(angles[0]),-np.sin(angles[0])],
                        [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                        [0,1,0],
                        [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                        [np.sin(angles[2]),np.cos(angles[2]),0],
                        [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        
        rotated_data= np.dot(batch_data, R)
        sample["points"]=rotated_data
        return sample

class jitter_point_cloud(object):
    def __init__(self, p: float = 0.5, sigma: float = 0.01, clip: float = 0.05):
        self.p = p
        self.sigma=sigma
        self.clip=clip
    def __call__(self,sample):
        """ Randomly jitter points. jittering is per point.
            Input:
                Nx3 array, original  point clouds
            Return:
                Nx3 array, jittered  point clouds
        """
        if self.p > random.random():
            batch_data = sample["points"]
            N, C = batch_data.shape
            assert(self.clip > 0)
            jittered_data = np.clip(self.sigma * np.random.randn(N, C), -1*self.clip, self.clip)
            jittered_data += batch_data
            sample["points"] = jittered_data
        return sample

class shift_point_cloud(object):
    def __init__(self, shift_range=0.1):
        self.shift_range=shift_range
    def __call__(self,sample):
        """ Randomly shift point cloud. Shift is per point cloud.
            Input:
                Nx3 array, original  point clouds
            Return:
                Nx3 array, shifted  point clouds
        """
        batch_data=sample["points"]
        N, C = batch_data.shape
        shifts = np.random.uniform(-self.shift_range, self.shift_range, (3))
        batch_data[:,:] += shifts[:]
        sample["points"]=batch_data
        return sample

class random_scale_point_cloud(object):
    def __init__(self, scale_low=0.8, scale_high=1.25):
        self.scale_low=scale_low
        self.scale_high=scale_high
    def __call__(self, sample: dict) -> dict:
        """ Randomly scale the point cloud. Scale is per point cloud.
            Input:
                Nx3 array, original  point clouds
            Return:
                Nx3 array, scaled  point clouds
        """
        batch_data=sample["points"]
        N, C = batch_data.shape
        scales = np.random.uniform(self.scale_low, self.scale_high)
        batch_data[:,:] *= scales
        sample["points"]=batch_data
        return sample

class random_point_dropout(object):
    def __init__(self, max_dropout_ratio=0.875):
        self.max_dropout_ratio=max_dropout_ratio
    def __call__(self,sample):
        ''' sample: Nx3 '''
        batch_data=sample["points"]
        dropout_ratio =  np.random.random()*self.max_dropout_ratio # 0~0.875
        drop_idx = np.where(np.random.random((batch_data.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            batch_data[drop_idx,:] = batch_data[0,:] # set to the first point
        sample["points"]=batch_data    
        return sample

class random_point_dropout_v2(object):
    def __init__(self, max_dropout_ratio=0.875):
        self.max_dropout_ratio=max_dropout_ratio
    def __call__(self,sample):
        ''' batch_data: Nx3 '''
        batch_data=sample["points"]
        dropout_ratio =  np.random.random()*self.max_dropout_ratio # 0~0.875
        drop_idx = np.where(np.random.random((batch_data.shape[1]))<=dropout_ratio)[0]
        keep_idx = np.where(np.random.random((batch_data.shape[1]))>dropout_ratio)[0]
        if len(keep_idx) > len(drop_idx):
            batch_data[drop_idx, :] = batch_data[keep_idx[:len(drop_idx)], :]
        else:
            batch_data[ drop_idx, :] = batch_data[:len(drop_idx), :]
        sample["points"]=batch_data
        return sample

class ToTensor(object):
    def __call__(self, sample):
        return torch.from_numpy(sample["points"])




